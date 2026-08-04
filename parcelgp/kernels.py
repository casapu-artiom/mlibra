"""A parcel-aware factor that multiplies onto any existing spatial kernel.

A stationary kernel makes covariance a function of ``||x - x'||`` and nothing
else, so it cannot express "these two voxels are close but on opposite sides of a
boundary, don't smooth between them" — nor the converse, "these two are far apart
but the same kind of tissue, do borrow". This kernel adds exactly that, and only
that::

    k(x, x') = k_base(x, x') * exp( -||z(x) - z(x')||^2 / 2 )
    z(x)     = m(x)^T B        B: (K, r) learnable

``m(x)`` is the sparse soft parcel membership from a :class:`~parcelgp.field.ParcelField`
— derived from the reference image alone, and defined at *every* voxel including
the unmeasured ones between sections, which is the entire point: the lipid
measurements cannot say what happens in a gap, the template can.

``B`` is learned from the lipid data. So the partition stays reference-only while
*which parcels behave alike chemically* is fitted, independently per task. That
split is what keeps the construction honest.

Design notes:

  * **PSD.** It is an RBF over a learned embedding, hence a valid kernel; the
    product with any PSD base kernel is PSD by the Schur product theorem.
  * **Collapses to a no-op.** ``B -> 0`` gives ``z == 0`` and a factor of exactly
    1, recovering the base kernel. ``B`` is initialized near zero, so training
    starts from the unmodified model and the ablation is honest — a lipid with no
    anatomical structure simply leaves ``B`` at zero.
  * **No separate lengthscale.** Only ``||z - z'||`` matters, and the scale of
    ``B`` already sets it, so an explicit depth lengthscale would be redundant
    with ``B`` and only add a flat direction to the optimization.
  * **Cheap.** Memberships are sparse (top-J), so ``z`` is an embedding-bag
    lookup: no dense (N, K) matrix and no K x K coregionalization matrix ever
    exists. Cost is O(n * J * r) per evaluation.

Not differentiable w.r.t. the input coordinates: ``m(x)`` comes from a nearest-node
KD-tree query. That is fine for fixed data and anchored inducing points, but it
means gradients will not reach *learned* inducing locations through this factor.
"""
from __future__ import annotations

import logging
from collections import OrderedDict

import gpytorch
import torch
from linear_operator import to_dense
from torch import nn

log = logging.getLogger(__name__)


class ParcelFactorKernel(gpytorch.kernels.Kernel):
    """Wrap ``base_kernel`` with a learned parcel-similarity factor.

    Args:
      base_kernel:  the spatial kernel to modulate (e.g. the ``MaternKernel``
                    inside a ``ScaleKernel``). Wrap the kernel *inside* the
                    ScaleKernel, not the ScaleKernel itself, so outputscale
                    priors and logging keep working.
      field:        a built :class:`~parcelgp.field.ParcelField`.
      num_tasks:    size of the task batch (lipids in the batch, or latent GPs).
      rank:         embedding width r. Small is the point: r=8 with K=128 is
                    1024 parameters per task, versus 8256 for a full K x K
                    coregionalization matrix.
      init_scale:   std of the ``B`` initialization. Keep it small so training
                    starts at (near) the unmodified base kernel.
      per_task:     one ``B`` per task (default) or a single shared ``B``.
    """

    def __init__(self, base_kernel, field, num_tasks: int = 1, rank: int = 8,
                 init_scale: float = 0.05, per_task: bool = True,
                 cache_size: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.field = field                      # plain object: not a submodule
        self.rank = int(rank)
        self.num_tasks = int(num_tasks)

        self.register_buffer("mem_idx", torch.as_tensor(field.mem_idx, dtype=torch.long))
        self.register_buffer("mem_val", torch.as_tensor(field.mem_val, dtype=torch.float32))
        n_b = self.num_tasks if per_task else 1
        self.register_parameter(
            "B", nn.Parameter(torch.randn(n_b, field.n_parcels, self.rank)
                              * float(init_scale)))

        # Nearest-node lookups are keyed by tensor identity, because the hot call
        # pattern is the SAME fixed coordinate tensor every optimizer step (the
        # inducing points, via K_uu). Strong ref to the tensor so its data_ptr
        # cannot be recycled underneath the key; _version invalidates the entry
        # if the tensor is mutated in place.
        self._idx_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._idx_cache_max = int(cache_size)

    # -- keep the wrapped kernel's lengthscale visible to priors / logging ----
    @property
    def raw_lengthscale(self):
        return getattr(self.base_kernel, "raw_lengthscale")

    @property
    def lengthscale(self):
        return getattr(self.base_kernel, "lengthscale", None)

    @lengthscale.setter
    def lengthscale(self, value):
        self.base_kernel.lengthscale = value

    # ------------------------------------------------------------------ lookup
    def _node_index(self, x: torch.Tensor) -> torch.Tensor:
        """Nearest template-node index per input point, shape ``x.shape[:-1]``."""
        key = (x.data_ptr(), getattr(x, "_version", 0), tuple(x.shape))
        hit = self._idx_cache.get(key)
        if hit is not None and hit[0] is x:
            self._idx_cache.move_to_end(key)
            return hit[1]
        flat = x.reshape(-1, x.shape[-1])[:, :3].detach().cpu().numpy()
        idx = torch.as_tensor(self.field.node_index(flat), dtype=torch.long,
                              device=x.device).reshape(x.shape[:-1])
        self._idx_cache[key] = (x, idx)
        if len(self._idx_cache) > self._idx_cache_max:
            self._idx_cache.popitem(last=False)
        return idx

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Parcel embedding ``z`` of every input point, shape ``(T, n, r)``.

        ``T`` is the broadcast of the input's task batch and ``B``'s, so a shared
        ``B`` works with batched inputs and vice versa.
        """
        idx = self._node_index(x)                       # (...,) node ids
        midx = self.mem_idx[idx]                        # (..., J)
        mval = self.mem_val[idx]                        # (..., J)
        Tb, K, r = self.B.shape

        if idx.dim() == 1:                              # (n,) -> (Tb, n, r)
            sel = self.B[:, midx]                       # (Tb, n, J, r)
            return (sel * mval.unsqueeze(0).unsqueeze(-1)).sum(-2)

        if idx.dim() != 2:
            raise NotImplementedError(
                f"ParcelFactorKernel expects 2-D or 3-D inputs, got shape {tuple(x.shape)}")

        Tx = idx.shape[0]
        if Tb == 1:                                     # shared B across tasks
            sel = self.B[0][midx]                       # (Tx, n, J, r)
        else:
            if Tb != Tx:
                raise ValueError(f"input task batch {Tx} != B task batch {Tb}")
            # Flatten (task, parcel) so one gather covers all tasks at once.
            flat = midx + (torch.arange(Tx, device=midx.device) * K).view(Tx, 1, 1)
            sel = self.B.reshape(Tb * K, r)[flat]       # (Tx, n, J, r)
        return (sel * mval.unsqueeze(-1)).sum(-2)       # (Tx, n, r)

    # ----------------------------------------------------------------- forward
    def forward(self, x1, x2, diag: bool = False, last_dim_is_batch: bool = False,
                **params):
        if last_dim_is_batch:
            raise NotImplementedError("ParcelFactorKernel: last_dim_is_batch unsupported")
        base = self.base_kernel.forward(x1, x2, diag=diag, **params)
        z1 = self._embed(x1)
        z2 = z1 if x1 is x2 or torch.equal(x1, x2) else self._embed(x2)
        if diag:
            factor = torch.exp(-0.5 * (z1 - z2).square().sum(-1))       # (T, n)
            return base * factor
        d2 = torch.cdist(z1, z2).square()                               # (T, n1, n2)
        return to_dense(base) * torch.exp(-0.5 * d2)

    # --------------------------------------------------------------- reporting
    def factor_stats(self) -> dict:
        """Diagnostics for the learned factor: how far it has moved from a no-op.

        ``offdiag_mean`` is the mean covariance multiplier between DIFFERENT
        parcels — 1.0 means the factor is doing nothing, smaller means the model
        has learned to stop smoothing across those borders.
        """
        with torch.no_grad():
            B = self.B                                       # (Tb, K, r)
            d2 = torch.cdist(B, B).square()                   # (Tb, K, K)
            f = torch.exp(-0.5 * d2)
            K = B.shape[1]
            off = ~torch.eye(K, dtype=torch.bool, device=B.device)
            return {
                "parcel/B_norm": float(B.norm(dim=-1).mean()),
                "parcel/offdiag_mean": float(f[:, off].mean()),
                "parcel/offdiag_min": float(f[:, off].min()),
            }


def wrap_scale_kernel(covar_module, field, num_tasks, rank=8, init_scale=0.05,
                      per_task=True):
    """Insert the parcel factor INSIDE a ``ScaleKernel``, returning the module.

    Wrapping the inner kernel rather than the ScaleKernel keeps
    ``covar_module`` a ``ScaleKernel``, so the runner's outputscale prior
    registration, lengthscale prior and W&B hyperparameter logging all keep
    finding what they expect.
    """
    if not isinstance(covar_module, gpytorch.kernels.ScaleKernel):
        raise TypeError(
            f"expected a ScaleKernel to wrap, got {type(covar_module).__name__}")
    covar_module.base_kernel = ParcelFactorKernel(
        covar_module.base_kernel, field, num_tasks=num_tasks, rank=rank,
        init_scale=init_scale, per_task=per_task,
    )
    log.info("parcel factor: K=%d parcels, rank=%d, per_task=%s (%d params)",
             field.n_parcels, rank, per_task,
             covar_module.base_kernel.B.numel())
    return covar_module
