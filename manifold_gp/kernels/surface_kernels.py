#!/usr/bin/env python
# encoding: utf-8
"""Surface-aware kernels: base spatial kernel × a depth (distance-to-surface) factor.

These are drop-in subclasses of the existing kernels that multiply the base
covariance by a 1-D factor over a per-point ``d_surface`` feature (depth into
tissue). By the Schur product theorem the product of two PSD kernels is PSD, so

    k_surface(x, x') = k_base(x_spatial, x'_spatial) · k_depth(d(x), d(x'))

is a valid kernel: two points covary only if they are close BOTH in the base
geometry (Euclidean for Matérn, graph-manifold for Riemann) AND at similar depth.
It's the depth analogue of the anatomy product kernel.

Two ways to supply the depth feature:

  * **sampler-in-kernel (recommended)** -- pass a ``depth_sampler`` (a
    ``surface_depth.DepthSampler``). The kernel derives ``d_surface`` from the
    (unchanged, spatial-only) input coords *inside* ``forward`` via a nearest-node
    lookup, so the coords stay 3-D everywhere: nothing else in the model/pipeline
    changes (inducing-point snapping, learnable inducing, and predict all keep
    working). This avoids appending a depth column at every coord entry point.

  * **column-append** -- carry ``d_surface`` as the **last input column**
    (``depth_dim=-1``) and the kernel reads it. Requires appending the column at
    every coord entry point, and fights the inducing-point snap invariant, so
    prefer the sampler.

``k_depth`` is a Gaussian (RBF) over the depth difference with its own learnable
``depth_lengthscale``. ``depth_lengthscale -> inf`` recovers the base kernel
exactly (depth factor -> 1), so the feature is opt-in and can be learned away.
"""
from typing import Optional
from collections import OrderedDict

import torch
from torch.nn.functional import softplus
from linear_operator import to_dense

import gpytorch

from .riemann_matern_kernel import RiemannMaternKernel


class _SurfaceDepthMixin:
    """Shared machinery for the depth factor. Mix in BEFORE the base kernel so the
    base ``forward`` stays reachable via ``super().forward`` (this mixin defines no
    ``forward``)."""

    def _init_surface(self, depth_dim: int = -1, depth_lengthscale: float = 1.0,
                      depth_sampler=None):
        self.depth_dim = depth_dim
        # A DepthSampler (surface_depth.DepthSampler) or None. When set, depth is
        # looked up from the coords inside forward instead of read from a column.
        # Plain object (not an nn.Module) -> not registered as a submodule/param.
        self._depth_sampler = depth_sampler
        # Precache: map a coord tensor's identity -> its (GPU) depth tensor, so a
        # FIXED input (the inducing points, used in K_uu every training step) skips
        # the CPU round-trip + nearest-node query on all but the first call. Keyed
        # by (data_ptr, _version, shape) with a strong ref to the tensor (so its
        # ptr can't be reused while cached) + an identity guard. Bounded LRU; the
        # hot inducing key is touched every step so it never evicts. The optimizer
        # bumps _version on in-place updates (--learn-inducing), invalidating it.
        self._depth_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._depth_cache_max = 8
        # softplus(raw) parameterisation -> positive lengthscale, no gpytorch
        # constraint plumbing needed. inv-softplus so the initial value is exact.
        inv = torch.tensor(float(depth_lengthscale)).expm1().clamp_min(1e-6).log()
        self.register_parameter("raw_depth_lengthscale",
                                torch.nn.Parameter(inv.reshape(1)))

    def __call__(self, *args, **kwargs):
        # gpytorch's debug-mode check compares ard_num_dims against the FULL input
        # width (spatial + the appended depth column), but the base kernel only
        # ever sees the spatial columns (we strip depth in forward). Scope the
        # debug checks off for our own evaluation so ard_num_dims can correctly
        # describe the SPATIAL dims.
        with gpytorch.settings.debug(False):
            return super().__call__(*args, **kwargs)

    @property
    def depth_lengthscale(self) -> torch.Tensor:
        return softplus(self.raw_depth_lengthscale)

    @depth_lengthscale.setter
    def depth_lengthscale(self, value: float):
        with torch.no_grad():
            v = torch.as_tensor(float(value), device=self.raw_depth_lengthscale.device)
            self.raw_depth_lengthscale.copy_(v.expm1().clamp_min(1e-6).log().reshape(1))

    def _spatial(self, x: torch.Tensor) -> torch.Tensor:
        """The base kernel's coordinates. In sampler mode the coords are already
        spatial-only (depth comes from the sampler), so pass through unchanged;
        in column mode drop the depth column."""
        if self._depth_sampler is not None:
            return x
        n = x.size(-1)
        dd = self.depth_dim % n
        if dd == n - 1:
            return x[..., :-1]
        keep = [i for i in range(n) if i != dd]
        return x[..., keep]

    def _depth_values(self, x: torch.Tensor) -> torch.Tensor:
        """Standardized d_surface per input point, shape x.shape[:-1]. From the
        sampler (nearest-node lookup on the spatial coords, cached by tensor
        identity) or the depth column."""
        if self._depth_sampler is None:
            return x[..., self.depth_dim]
        key = (x.data_ptr(), getattr(x, "_version", 0), tuple(x.shape))
        cached = self._depth_cache.get(key)
        if cached is not None and cached[0] is x:      # identity guard vs ptr reuse
            self._depth_cache.move_to_end(key)
            return cached[1]
        flat = x.reshape(-1, x.shape[-1]).detach().cpu().numpy()
        dz = self._depth_sampler.sample(flat)          # (P,) numpy, non-diff
        out = torch.as_tensor(dz, dtype=x.dtype, device=x.device).reshape(x.shape[:-1])
        self._depth_cache[key] = (x, out)              # strong ref -> ptr stays valid
        if len(self._depth_cache) > self._depth_cache_max:
            self._depth_cache.popitem(last=False)
        return out

    def _depth_factor(self, x1: torch.Tensor, x2: torch.Tensor,
                      diag: bool) -> torch.Tensor:
        """Gaussian factor exp(-(d1-d2)^2 / (2 ell_d^2)) over d_surface."""
        d1 = self._depth_values(x1)
        d2 = self._depth_values(x2)
        ls2 = self.depth_lengthscale.square().clamp_min(1e-12)
        if diag:
            return torch.exp(-(d1 - d2).square() / (2.0 * ls2))
        diff = d1.unsqueeze(-1) - d2.unsqueeze(-2)          # (..., n1, n2)
        return torch.exp(-diff.square() / (2.0 * ls2))


class SurfaceMaternKernel(_SurfaceDepthMixin, gpytorch.kernels.MaternKernel):
    """Euclidean Matérn (over the spatial columns) × Gaussian depth factor.

    Pass ``ard_num_dims`` equal to the number of SPATIAL dims (not counting the
    appended depth column). All standard ``MaternKernel`` kwargs pass through.
    """

    def __init__(self, *args, depth_dim: int = -1, depth_lengthscale: float = 1.0,
                 depth_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_surface(depth_dim, depth_lengthscale, depth_sampler)

    def forward(self, x1, x2, diag: bool = False, **kwargs):
        base = super().forward(self._spatial(x1), self._spatial(x2),
                               diag=diag, **kwargs)
        factor = self._depth_factor(x1, x2, diag)
        if diag:
            return base * factor
        return to_dense(base) * factor


class SurfaceRiemannMaternKernel(_SurfaceDepthMixin, RiemannMaternKernel):
    """Manifold (graph-Laplacian) Matérn × Gaussian depth factor.

    The base ``RiemannKernel.forward`` returns a (possibly low-rank) LinearOperator;
    the Hadamard product with the dense depth factor materialises it, so this is
    best used at the inducing-point / test scale (M×M, M×N_test), not the full
    N×N graph. depth_lengthscale->inf recovers the plain manifold kernel.
    """

    def __init__(self, *args, depth_dim: int = -1, depth_lengthscale: float = 1.0,
                 depth_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_surface(depth_dim, depth_lengthscale, depth_sampler)

    def forward(self, x1, x2, diag: bool = False, **kwargs):
        base = super().forward(self._spatial(x1), self._spatial(x2),
                               diag=diag, **kwargs)
        factor = self._depth_factor(x1, x2, diag)
        return to_dense(base) * factor
