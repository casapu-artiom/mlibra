#!/usr/bin/env python
# encoding: utf-8
"""
Laplacian eigensolver for Riemann manifold kernels.

Extracts the eigendecomposition of a graph Laplacian out of RiemannKernel into
a standalone, cacheable component. Two reasons for this:

  1. Eigendecomposition is the slowest part of building a manifold kernel
     (minutes to hours for >1M-node graphs). Computing it once and reusing
     across experiments saves a lot of wall time.

  2. The eigenvectors only depend on graph topology, the bandwidth, and the
     normalization choice -- they don't depend on the data, the GP
     hyperparameters, or which downstream model uses them. Tying them to the
     kernel's lifecycle made caching ad-hoc.

Usage
-----
A) One-shot compute (no cache):

       op = kernel.laplacian()                      # GraphLaplacianOperator
       solver = LaplacianEigensolver(num_modes=200)
       eigval, eigvec = solver.compute(op)
       kernel.attach_eigenpairs(eigval, eigvec)

B) Compute-or-load (with disk cache):

       solver = LaplacianEigensolver(num_modes=200)
       eigval, eigvec = solver.compute_or_load(
           kernel.laplacian(),
           cache_dir=Path("cache/eigvecs"),
           key="allenccf_stride4_knn5_modes200_sym",
           graphbandwidth=kernel.graphbandwidth.item(),
           laplacian_normalization=kernel.laplacian_normalization,
           extra={"stride": 4, "knn": 5, "template": "allen_25um"},
       )
       kernel.attach_eigenpairs(eigval, eigvec)

Cache identity
--------------
You own the `key`. Compose it from whatever uniquely identifies your run --
template name, stride, KNN method, K, threshold, num_modes, normalization,
etc. The helper `make_key({...})` will format a dict deterministically into
a filesystem-safe string.

Alongside the cached arrays we store a fingerprint:
    {n_nodes, n_edges, num_modes, normalization, graphbandwidth,
     edge_value_mean, edge_value_std, backend}
On load we verify it matches the operator passed at the call site. Mismatches
emit a warning and trigger recompute (or raise, with strict_fingerprint=True).
This catches the case where someone reuses a key across two runs that aren't
really the same graph.

If the cached file has MORE eigenvectors than requested, we just truncate.
If it has fewer, that's a mismatch -> recompute.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import uuid
import wandb

import numpy as np
import torch
from torch.nn.functional import normalize as l2_normalize


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
@dataclass
class EigenFingerprint:
    """Structural summary of the Laplacian we eigendecomposed.

    Stored next to the cached arrays, checked on load. Mismatches don't crash
    by default -- they emit a warning and trigger recompute. Pass
    strict_fingerprint=True to LaplacianEigensolver if you want hard failure.
    """
    n_nodes: int
    n_edges: int
    num_modes: int
    laplacian_normalization: str
    graphbandwidth: float
    edge_value_mean: float
    edge_value_std: float
    backend: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EigenFingerprint":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})

    def matches(self, other: "EigenFingerprint",
                rtol: float = 1e-4) -> Tuple[bool, str]:
        """Strict on shape/normalization, tolerant on floats.

        Note we ALLOW cached.num_modes >= requested.num_modes -- if the cache
        has 500 modes and you ask for 200, you get the first 200. We only
        fail if the cache is too small.
        """
        if self.n_nodes != other.n_nodes:
            return False, f"n_nodes {self.n_nodes} vs {other.n_nodes}"
        if self.n_edges != other.n_edges:
            return False, f"n_edges {self.n_edges} vs {other.n_edges}"
        if self.laplacian_normalization != other.laplacian_normalization:
            return False, (
                f"normalization {self.laplacian_normalization!r} vs "
                f"{other.laplacian_normalization!r}"
            )
        if self.num_modes < other.num_modes:
            return False, (
                f"cached has {self.num_modes} modes, requested {other.num_modes}"
            )
        for fld in ("graphbandwidth", "edge_value_mean", "edge_value_std"):
            a, b = getattr(self, fld), getattr(other, fld)
            if abs(a - b) > rtol * (abs(a) + abs(b) + 1e-12):
                return False, f"{fld} differs: {a:.6g} vs {b:.6g}"
        return True, ""

class _IterationTracker:
    """Counts work units (SpMVs on cupy, OPinv solves on scipy) and reports
    progress to tqdm + wandb. Same throttling for both backends so dashboards
    line up across runs.
 
    The unit of work differs by backend, but the wandb key names are kept
    identical (`solver/spmv_iterations`, `solver/iters_per_sec`) so cross-
    backend runs are directly comparable in the dashboard. The `unit` field
    just controls the tqdm description and is exposed on the wandb run
    config separately if you want to disambiguate.
    """
 
    def __init__(self, desc: str, verbose: bool, log_interval_seconds: float = 2.0):
        from tqdm import tqdm  # local import keeps the module import light
        self.pbar = tqdm(desc=desc, leave=False, disable=not verbose)
        self.iteration_count = 0
        self._iters_at_last_log = 0
        self._last_log_time = time.time()
        self._log_interval_seconds = log_interval_seconds
 
    def tick(self) -> None:
        self.iteration_count += 1
        self.pbar.update(1)
        now = time.time()
        elapsed = now - self._last_log_time
        if elapsed < self._log_interval_seconds:
            return
        # Rate over the *window* since the last log -- not the run-average,
        # which would smear over the whole solve. Cheap and accurate.
        delta = self.iteration_count - self._iters_at_last_log
        rate = delta / elapsed if elapsed > 0 else 0.0
        if wandb is not None and wandb.run is not None:
            wandb.log({
                "solver/spmv_iterations": self.iteration_count,
                "solver/iters_per_sec": rate,
            })
        self._iters_at_last_log = self.iteration_count
        self._last_log_time = now
 
    def close(self) -> None:
        self.pbar.close()

# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class LaplacianEigensolver:
    """Compute (or load) the smallest eigenpairs of a graph Laplacian.

    Backend
    -------
    'cupy'  -- GPU-accelerated CuPy Lanczos (cupyx.scipy.sparse.linalg.eigsh).
               Default. Matches the original RiemannKernel behavior bit-for-bit.
    'scipy' -- CPU fallback. Uses shift-invert at sigma=0, which CuPy doesn't
               support; results are equivalent up to numerical noise.

    Post-processing (the same regardless of backend, as it was inline in
    RiemannKernel.eval before):
      1. Truncate to num_modes (Lanczos sometimes overshoots).
      2. Force the smallest eigenvalue to 0 exactly. Theoretical value for the
         Laplacian on a connected graph; numerical noise pushes it to ~1e-12.
      3. De-normalize the symmetric Laplacian eigenvectors to recover the
         random-walk eigenvectors:  phi = D^(-1/2) phi_sym
      4. L2-normalize each eigenvector column for numerical stability.
    """

    def __init__(self,
                 num_modes: int = 200,
                 backend: str = "cupy",
                 tol: float = 1e-4,
                 ncv_min: int = 1500,
                 strict_fingerprint: bool = False,
                 verbose: bool = True):
        print(f"[eigensolver] using backend: {backend}")
        if backend not in ("cupy", "scipy"):
            raise ValueError(f"Unknown backend: {backend!r}")
        self.num_modes = num_modes
        self.backend = backend
        self.tol = tol
        self.ncv_min = ncv_min
        self.strict_fingerprint = strict_fingerprint
        self.verbose = verbose

    # -------------------------------------------------------------- compute
    def compute(self, laplacian_op) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve for the bottom `num_modes` eigenpairs.

        Args:
          laplacian_op: GraphLaplacianOperator -- exposes .idx,
                        .laplacian_triu, .laplacian_diag, .degree_mat,
                        .operator_dimension. (Same object that
                        RiemannKernel.laplacian() returns.)

        Returns:
          eigval: (num_modes,)   float on the operator's device
          eigvec: (N, num_modes) float on the operator's device, with all the
                  post-processing already applied -- the kernel just installs
                  these directly.
        """
        if self.backend == "cupy":
            evals, evecs = self._compute_cupy(laplacian_op)
        else:
            evals, evecs = self._compute_scipy(laplacian_op)
        return self._postprocess(evals, evecs, laplacian_op)

    def _compute_cupy(self, op) -> Tuple[torch.Tensor, torch.Tensor]:
        # Lazy imports so users without CuPy can still use the scipy backend.
        import cupy as cp
        import cupyx.scipy.sparse as cpsparse
        import cupyx.scipy.sparse.linalg as cplinalg
        import scipy.sparse as sp
        from tqdm import tqdm

        N = op.operator_dimension
        idx = op.idx.cpu()
        val = op.laplacian_triu.cpu()
        diag = op.laplacian_diag.cpu()

        # SciPy CSR is the easiest way to assemble the symmetric L from the
        # upper triangle + diagonal that GraphLaplacianOperator gives us.
        row, col = idx[0].numpy(), idx[1].numpy()
        L_scipy = sp.csr_matrix((-val.numpy(), (row, col)), shape=(N, N))
        L_scipy = L_scipy + L_scipy.T
        L_scipy.setdiag(diag.numpy())

        if self.verbose:
            nnz = L_scipy.nnz
            density = (nnz / (N * N)) * 100
            print(f"[eigensolver] N={N:,}  NNZ={nnz:,}  density={density:.6f}%")

        # Hand off to GPU
        L_cp = cpsparse.csr_matrix(L_scipy)

        # Wrap the SpMV with a tracker that reports to both tqdm and wandb
        class _Tracked(cplinalg.LinearOperator):
            def __init__(self, A, verbose):
                super().__init__(A.dtype, A.shape)
                self.A = A
                self.pbar = tqdm(desc="[eigensolver] CuPy Lanczos SpMV",
                                 leave=False, disable=not verbose)
                
                # Tracking state for W&B
                self.iteration_count = 0
                self.last_log_time = time.time()
                self.log_interval_seconds = 2.0  # Only ping W&B every 2 seconds

            def _matvec(self, x):
                self.iteration_count += 1
                self.pbar.update(1)
                
                # Throttled W&B logging
                current_time = time.time()
                if current_time - self.last_log_time > self.log_interval_seconds:
                    if wandb is not None and wandb.run is not None:
                        wandb.log({
                            "solver/spmv_iterations": self.iteration_count,
                            "solver/iters_per_sec": self.iteration_count / max(1e-5, current_time - self.last_log_time + self.log_interval_seconds) # rough speed
                        })
                    self.last_log_time = current_time
                    
                return self.A @ x

        op_tracked = _Tracked(L_cp, self.verbose)

        # ncv (Krylov subspace size) > 2k helps when eigenvalues cluster, which
        # they do for smooth manifold modes. Bumping to >=1500 was the original
        # tuned default.
        ncv = min(N, max(2 * self.num_modes + 1, self.ncv_min))

        evals_cp, evecs_cp = cplinalg.eigsh(
            op_tracked,
            k=self.num_modes,
            which="SA",   # smallest algebraic; CuPy doesn't support sigma=0
            tol=self.tol,
            ncv=ncv,
        )
        # Numerical noise can produce small negatives at the bottom of the
        # spectrum, especially with disconnected components. Clamp them.
        evals_cp = cp.clip(evals_cp, a_min=0.0, a_max=None)

        cp.cuda.Stream.null.synchronize()
        op_tracked.pbar.close()

        # Drop the host-side scratch matrix; the GPU operator hangs onto its
        # own copy until eigsh returns.
        del L_scipy, L_cp
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # CuPy -> torch (zero-copy via dlpack when possible). Pass the cupy array
        # itself, NOT array.toDlpack(): torch>=2.x consumes the object's __dlpack__
        # protocol and negotiates the DLPack version. The legacy raw-capsule path
        # (toDlpack()) fails on newer torch/cupy with "from_dlpack received an
        # invalid capsule ... consumed only once".
        if hasattr(evecs_cp, "__dlpack__"):
            evals = torch.from_dlpack(evals_cp).float()
            evecs = torch.from_dlpack(evecs_cp).float()
        else:
            evals = torch.from_numpy(cp.asnumpy(evals_cp)).float().cuda()
            evecs = torch.from_numpy(cp.asnumpy(evecs_cp)).float().cuda()

        return evals, evecs

    def _compute_scipy(self, op) -> Tuple[torch.Tensor, torch.Tensor]:
        import scipy.sparse as sp
        import scipy.sparse.linalg as sla
 
        N = op.operator_dimension
        idx = op.idx.cpu()
        val = op.laplacian_triu.cpu()
        diag = op.laplacian_diag.cpu()
 
        row, col = idx[0].numpy(), idx[1].numpy()
        L_scipy = sp.csr_matrix((-val.numpy(), (row, col)), shape=(N, N))
        L_scipy = L_scipy + L_scipy.T
        L_scipy.setdiag(diag.numpy())
 
        if self.verbose:
            print(f"[eigensolver] (scipy) N={N:,}  NNZ={L_scipy.nnz:,}")
 
        # Shift-invert at sigma=0 finds eigenvalues closest to zero efficiently.
        # This is the better choice on CPU; CuPy doesn't support it which is
        # why the GPU path uses 'SA' instead.
        #
        # In shift-invert mode ARPACK doesn't multiply by L per iteration -- it
        # solves L x = b against a pre-factorized splu(L). So the unit of work
        # we want to track is OPinv solves, not SpMVs of L. We factorize once
        # up front (the dominant cost for moderate N), then wrap splu.solve in
        # a LinearOperator that ticks the iteration tracker on every call.
 
        if self.verbose:
            print("[eigensolver] (scipy) factorizing L (shift-invert sigma=0)...")
        t_fact = time.time()
        splu_solver = sla.splu(L_scipy.tocsc())
        factorization_seconds = time.time() - t_fact
        if self.verbose:
            print(
                f"[eigensolver] (scipy) factorization done in "
                f"{factorization_seconds:.1f}s"
            )
        if wandb is not None and wandb.run is not None:
            wandb.log({"solver/factorization_seconds": factorization_seconds})
 
        # Wrap splu.solve so each ARPACK shift-invert step shows up in tqdm
        # and wandb. Same key names as the cupy path so dashboards line up.
        class _TrackedOPinv(sla.LinearOperator):
            def __init__(self, solver, dtype, shape, verbose):
                super().__init__(dtype, shape)
                self.solver = solver
                self.tracker = _IterationTracker(
                    desc="[eigensolver] SciPy Lanczos OPinv solve",
                    verbose=verbose,
                )
 
            def _matvec(self, x):
                self.tracker.tick()
                return self.solver.solve(x)
 
        op_tracked = _TrackedOPinv(
            splu_solver, L_scipy.dtype, L_scipy.shape, self.verbose,
        )
 
        try:
            evals_np, evecs_np = sla.eigsh(
                L_scipy,
                k=self.num_modes,
                sigma=0,
                which="LM",
                tol=self.tol,
                OPinv=op_tracked,
            )
        finally:
            op_tracked.tracker.close()
 
        evals_np = np.clip(evals_np, a_min=0.0, a_max=None)
 
        device = op.idx.device
        evals = torch.from_numpy(evals_np).float().to(device)
        evecs = torch.from_numpy(evecs_np).float().to(device)
        return evals, evecs

    def _postprocess(self, evals, evecs, op) -> Tuple[torch.Tensor, torch.Tensor]:
        evals = evals[: self.num_modes]
        evecs = evecs[:, : self.num_modes]
        evals[0] = 0.0

        # Both branches start from φ_sym (symmetric Laplacian eigvecs).
        # QR re-orthogonalizes Lanczos drift before any further transform.
        Q, R = torch.linalg.qr(evecs)
        signs = R.diagonal().sign()
        signs[signs == 0] = 1.0
        evecs = Q * signs.unsqueeze(0)   # φ_sym, now properly L2-orthonormal

        if op.normalization == "randomwalk":
            # φ_rw = D^(-1/2) φ_sym  →  D-orthonormal (φ^T D φ = I)
            degree_safe = torch.clamp(op.degree_mat, min=1e-8)
            evecs = evecs * degree_safe.pow(-0.5).view(-1, 1)

        return evals, evecs

    # ---------------------------------------------------------- fingerprint
    def fingerprint(self, op, graphbandwidth: float,
                    laplacian_normalization: str) -> EigenFingerprint:
        ev = op.laplacian_triu.detach()
        return EigenFingerprint(
            n_nodes=int(op.operator_dimension),
            n_edges=int(op.idx.shape[1]),
            num_modes=int(self.num_modes),
            laplacian_normalization=str(laplacian_normalization),
            graphbandwidth=float(graphbandwidth),
            edge_value_mean=float(ev.mean().item()),
            edge_value_std=float(ev.std().item()),
            backend=self.backend,
        )

    # ----------------------------------------------------------- save/load
    @staticmethod
    def _paths(cache_dir: Path, key: str) -> Tuple[Path, Path]:
        cache_dir = Path(cache_dir)
        return (
            cache_dir / f"{key}.eigpairs.npz",
            cache_dir / f"{key}.eigpairs.meta.json",
        )

    def save(self,
            cache_dir: Path, key: str,
            eigval: torch.Tensor, eigvec: torch.Tensor,
            fp: EigenFingerprint,
            extra: Optional[Dict[str, Any]] = None) -> Path:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        npz_path, meta_path = self._paths(cache_dir, key)

        # Stage the npz to local scratch, then copy. np.savez writes a zip and
        # zipfile.close() seeks back to patch the central directory -- FUSE-
        # backed S3 mounts (s3fs/goofys/mountpoint-s3) don't support seeks on
        # write-open files (Errno 95 EOPNOTSUPP), so we can't write directly
        # if cache_dir is on one. /tmp is always a real filesystem.
        import tempfile, shutil
        with tempfile.TemporaryDirectory(prefix="eigensolver_") as tmpd:
            tmp_local_npz = Path(tmpd) / npz_path.name
            np.savez(
                tmp_local_npz,
                eigval=eigval.detach().cpu().numpy(),
                eigvec=eigvec.detach().cpu().numpy(),
            )
            # 1. Create a unique temporary path on the FUSE S3 mount
            tmp_s3_npz = npz_path.with_name(f"{npz_path.name}.{uuid.uuid4().hex}.tmp")
            
            # 2. Copy the file to the unique FUSE path safely
            shutil.copyfile(tmp_local_npz, tmp_s3_npz)
            
            # 3. Atomically replace the target file
            os.replace(tmp_s3_npz, npz_path)
            # shutil.copyfile(tmp_npz, npz_path)

        # JSON is a single sequential write -- no seeks, fine to write direct.
        meta = {
            "key": key,
            "fingerprint": fp.to_dict(),
            "extra": extra or {},
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        if self.verbose:
            print(f"[eigensolver] saved -> {npz_path}")
        return npz_path

    def load(self, cache_dir: Path, key: str, device=None
             ) -> Optional[Tuple[torch.Tensor, torch.Tensor,
                                 EigenFingerprint, Dict]]:
        npz_path, meta_path = self._paths(cache_dir, key)
        if not npz_path.exists() or not meta_path.exists():
            return None
        with open(meta_path) as f:
            meta = json.load(f)
        fp = EigenFingerprint.from_dict(meta["fingerprint"])
        data = np.load(npz_path)
        evals = torch.from_numpy(data["eigval"]).float()
        evecs = torch.from_numpy(data["eigvec"]).float()
        if device is not None:
            evals = evals.to(device)
            evecs = evecs.to(device)
        return evals, evecs, fp, meta

    # -------------------------------------------------------- compute_or_load
    def compute_or_load(
        self,
        laplacian_op,
        cache_dir: Path,
        key: str,
        graphbandwidth: float,
        laplacian_normalization: str,
        extra: Optional[Dict[str, Any]] = None,
        force_recompute: bool = False,
        device=None,
        allow_larger_modes: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Try cache, fall back to compute, then save.

        Args:
          laplacian_op:    GraphLaplacianOperator from kernel.laplacian().
          cache_dir:       directory for the cache files.
          key:             stable name identifying this graph. You own the
                           naming scheme; e.g. "allen25_stride4_knn5_modes200_sym".
                           Use make_key({...}) below if you want a helper.
          graphbandwidth:  the bandwidth used inside laplacian_op (we read it
                           explicitly so the fingerprint includes it; the
                           operator itself stores it as a tensor that's
                           awkward to extract uniformly).
          laplacian_normalization: 'symmetric' / 'random_walk' / 'unnormalized'.
          extra:           free-form dict stored in the meta sidecar (stride,
                           knn method, template name, etc.). Documentation
                           only -- not used for matching.
          force_recompute: skip cache lookup, always recompute.
          device:          optional device to place loaded tensors on. If you
                           pass the kernel's device here, you avoid an extra
                           .to() at the call site.
          allow_larger_modes: on an exact-key miss, reuse a sibling cache with
                           the same graph but MORE modes (trimmed to num_modes).
                           Eigenpairs are nested, so this is exact, not an
                           approximation. Only affects the load path -- a true
                           miss still computes and saves under the requested key.
        """
        wanted = self.fingerprint(laplacian_op, graphbandwidth,
                                  laplacian_normalization)

        if not force_recompute:
            # Resolve the load key separately from `key`: if we end up computing,
            # we still save under the originally requested `key` (so we never
            # overwrite the larger sibling we borrowed from).
            load_key = key
            if allow_larger_modes:
                load_key = resolve_modes_key(cache_dir, key, self.num_modes)
                if load_key != key and self.verbose:
                    print(f"[eigensolver] reusing larger-modes cache "
                          f"{load_key} for request {key}")
            hit = self.load(cache_dir, load_key, device=device)
            if hit is not None:
                evals, evecs, cached_fp, _meta = hit
                ok, why = cached_fp.matches(wanted)
                if ok:
                    if self.verbose:
                        print(f"[eigensolver] HIT  {cache_dir}/{load_key}")
                    # Trim if the cache has more modes than we need now.
                    return evals[: self.num_modes], evecs[:, : self.num_modes]
                msg = f"[eigensolver] cache mismatch for key={load_key!r}: {why}"
                if self.strict_fingerprint:
                    raise RuntimeError(msg + " (strict_fingerprint=True)")
                logging.warning(msg + " -- recomputing.")

        if self.verbose:
            print(f"[eigensolver] MISS key={key} -- computing")
        t0 = time.time()
        evals, evecs = self.compute(laplacian_op)
        if self.verbose:
            print(f"[eigensolver] computed in {time.time() - t0:.1f}s")

        self.save(cache_dir, key, evals, evecs, wanted, extra=extra)
        # Honor the requested device for the freshly computed tensors too.
        if device is not None:
            evals = evals.to(device)
            evecs = evecs.to(device)
        return evals, evecs


# ---------------------------------------------------------------------------
# Helper: deterministic cache key from a config dict
# ---------------------------------------------------------------------------
def make_key(parts: Dict[str, Any]) -> str:
    """Compose a filesystem-safe cache key from a config dict.

    Sorted by key for stability. Floats are formatted with %g to round off
    the noise that turns 1.0 into 0.9999999.

    Example:
      make_key({"template": "allen_25um", "stride": 4, "knn": 5,
                "modes": 200, "norm": "sym"})
      -> "knn=5_modes=200_norm=sym_stride=4_template=allen_25um"
    """
    pieces = []
    for k in sorted(parts.keys()):
        v = parts[k]
        if isinstance(v, float):
            v = f"{v:.6g}"
        pieces.append(f"{k}={v}")
    return "_".join(pieces).replace("/", "-").replace(" ", "")


# make_key sorts fields alphabetically, so for the eigvec key
# ({bw, graph, modes, norm}) the modes field is always `_modes=<int>_norm=` --
# graph sorts before modes, norm after. That lets us swap just the mode count.
_MODES_FIELD_RE = re.compile(r"_modes=(\d+)_norm=")


def resolve_modes_key(cache_dir: Path, key: str, num_modes: int) -> str:
    """Find an existing eigvec cache that can serve `num_modes`, else return key.

    Eigenpairs are nested: a cache computed with M >= num_modes modes contains
    the requested ones as its leading columns, and both `EigenFingerprint.matches`
    (cached.num_modes >= requested) and `compute_or_load` (trims to num_modes on
    load) already handle that. The only thing standing in the way is that the
    mode count is baked into the cache *key* (the filename), so an exact-modes
    lookup walks right past a bigger, fully-usable cache sitting next to it.

    This returns the key of the *smallest* sibling cache (same graph / norm / bw)
    whose mode count is >= num_modes -- smallest so we load and trim the least.
    If the exact key already exists, or no suitable larger sibling is found, the
    original `key` is returned unchanged, so callers fall through to their normal
    compute-and-save path (saving under the requested key, as before).
    """
    cache_dir = Path(cache_dir)
    npz, meta = LaplacianEigensolver._paths(cache_dir, key)
    if npz.exists() and meta.exists():
        return key  # exact hit -- nothing to resolve

    m = _MODES_FIELD_RE.search(key)
    if m is None:
        return key  # unexpected key shape; leave it alone

    # The key with its mode count blanked out -- the "same graph" signature.
    neutral = _MODES_FIELD_RE.sub("_modes=*_norm=", key)
    best_key, best_modes = None, None
    for cand_npz in cache_dir.glob("*.eigpairs.npz"):
        cand = cand_npz.name[: -len(".eigpairs.npz")]
        cm = _MODES_FIELD_RE.search(cand)
        if cm is None or _MODES_FIELD_RE.sub("_modes=*_norm=", cand) != neutral:
            continue
        cand_modes = int(cm.group(1))
        if cand_modes < num_modes:
            continue
        if not (cache_dir / f"{cand}.eigpairs.meta.json").exists():
            continue  # need both halves to load
        if best_modes is None or cand_modes < best_modes:
            best_key, best_modes = cand, cand_modes
    return best_key if best_key is not None else key


# ---------------------------------------------------------------------------
# Helper: resolve the Lanczos Krylov subspace floor (ncv_min)
# ---------------------------------------------------------------------------
def resolve_ncv_min(num_modes: int, ncv_min: Optional[int] = None) -> int:
    """Pick the Krylov subspace floor for the Lanczos solver.

    Two regimes:

    * auto (``ncv_min`` is None or <= 0) -- return ``max(1500, 3*num_modes+20)``.
      The 1500 floor is robust insurance for the clustered low Laplacian
      spectrum and is cheap when N is small (stride >= 2: a few hundred
      thousand nodes).

    * explicit (``ncv_min`` > 0) -- return it verbatim, even if it drops *below*
      the auto floor. At stride=1 (tens of millions of nodes) the 1500 floor
      becomes the GPU-memory bottleneck (~ncv * N * 4 bytes for the basis), so
      a small explicit value (e.g. 100) is the way to make those runs fit. The
      solver still enforces the hard ARPACK minimum ``2*num_modes+1`` on top of
      whatever this returns, so an override that is too small is clamped up, not
      silently wrong.
    """
    if ncv_min is not None and int(ncv_min) > 0:
        return int(ncv_min)
    return max(1500, 3 * int(num_modes) + 20)