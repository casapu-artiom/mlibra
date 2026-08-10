#!/usr/bin/env python
# encoding: utf-8
"""
Compare three eigensolver approaches for the smallest graph-Laplacian modes.

Motivation
----------
The production GPU path (`LaplacianEigensolver` backend="cupy") runs plain
Lanczos with `which="SA"` because CuPy's eigsh has no shift-invert. The bottom
of the Laplacian spectrum is clustered with tiny gaps relative to ||L||, which
is exactly the regime where bare Lanczos converges slowly and resolves
individual eigenvectors poorly. This tool quantifies whether that actually
matters on a *real cached graph* by comparing:

  1. cupy_sa   -- current GPU default: cupyx eigsh, which="SA".
  2. scipy_si  -- CPU shift-invert at sigma=0 (the accurate reference path).
  3. lobpcg    -- LOBPCG for the smallest modes with a Jacobi preconditioner
                  (GPU via cupyx if available, else scipy).

against a reference (dense `eigh` when N is small enough, otherwise scipy
shift-invert), and reports per-mode residuals, eigenvalue errors, subspace
principal angles, orthonormality, and wall time.

It loads the graph from an existing KnnGraphCache directory, so you point it at
a cache you've already built rather than rebuilding the KNN graph.

Usage
-----
    python eigensolver_compare.py \
        --knn-dir /home/casap/mlibra/output/eigenvectors/knn \
        --modes 200 --bandwidth 1.0 --device cuda

If the cache dir holds exactly one graph the key is auto-detected; otherwise
pass --knn-key (or run with no key to list what's available).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from manifold_gp.utils.nearest_neighbors import KnnGraphCache, make_key as make_graph_key
from manifold_gp.utils.compute_eigenvectors import LaplacianEigensolver, resolve_ncv_min
from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator

# Moved from maldi/ to benchmarks/; the maldi sibling `utils` (imported inside a
# function below) lives in ../maldi, so put that on sys.path at module load.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "maldi"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "manifold"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "manifold" / "viz"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import psutil
except Exception:
    psutil = None


# ---------------------------------------------------------------------------
# Resource monitoring
# ---------------------------------------------------------------------------
def _nvidia_smi_available() -> bool:
    try:
        subprocess.check_output(["nvidia-smi", "--version"],
                                stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


class ResourceMonitor:
    """Sample host + GPU resource use around a block of work.

    A background thread polls every `interval` seconds:
      * host RSS via psutil,
      * GPU utilization % and GPU memory used via `nvidia-smi`.

    We use nvidia-smi (not torch.cuda.max_memory_allocated) for GPU memory
    because cupy allocates through its own pool, outside torch's caching
    allocator -- torch's counters would miss the cupy_sa / lobpcg-cupy paths
    entirely. The torch peak is still reported separately as a cross-check.

    Caveat: nvidia-smi `memory.used` is device-wide; the *delta* subtracts a
    baseline taken at __enter__, but a concurrent process on the same GPU adds
    noise. Run benchmarks on an otherwise-idle device.
    """

    def __init__(self, gpu_index: int = 0, interval: float = 0.1,
                 enable_gpu: bool = True):
        self.gpu_index = gpu_index
        self.interval = interval
        self.enable_gpu = enable_gpu and _nvidia_smi_available()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process() if psutil is not None else None
        self.host_rss: List[float] = []
        self.gpu_mem: List[float] = []
        self.gpu_util: List[float] = []
        self.base_rss = 0.0
        self.base_gpu_mem = 0.0
        self.torch_peak_mb = 0.0

    def _read_gpu(self) -> Tuple[float, float]:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-i", str(self.gpu_index)],
            text=True, stderr=subprocess.DEVNULL,
        )
        u, m = out.strip().split(",")
        return float(u), float(m)        # percent, MiB

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._proc is not None:
                self.host_rss.append(self._proc.memory_info().rss)
            if self.enable_gpu:
                try:
                    u, m = self._read_gpu()
                    self.gpu_util.append(u)
                    self.gpu_mem.append(m)
                except Exception:
                    pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "ResourceMonitor":
        if self._proc is not None:
            self.base_rss = float(self._proc.memory_info().rss)
        if self.enable_gpu:
            try:
                _, self.base_gpu_mem = self._read_gpu()
            except Exception:
                self.enable_gpu = False
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if torch.cuda.is_available():
            self.torch_peak_mb = torch.cuda.max_memory_allocated() / 1e6

    def stats(self) -> dict:
        pk = lambda xs: (max(xs) if xs else 0.0)
        mean = lambda xs: (sum(xs) / len(xs) if xs else 0.0)
        return {
            "host_rss_peak_mb": pk(self.host_rss) / 1e6,
            "host_rss_delta_mb": (pk(self.host_rss) - self.base_rss) / 1e6
            if self.host_rss else 0.0,
            "gpu_mem_peak_mb": pk(self.gpu_mem),
            "gpu_mem_delta_mb": (pk(self.gpu_mem) - self.base_gpu_mem)
            if self.gpu_mem else 0.0,
            "gpu_util_peak_pct": pk(self.gpu_util),
            "gpu_util_mean_pct": mean(self.gpu_util),
            "torch_peak_mb": self.torch_peak_mb,
            "n_samples": len(self.gpu_util) or len(self.host_rss),
        }


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------
def discover_keys(knn_dir: Path) -> List[str]:
    """All cached graph keys in a KnnGraphCache directory."""
    return sorted(p.name[: -len(".knngraph.npz")]
                  for p in Path(knn_dir).glob("*.knngraph.npz"))


def load_operator(knn_dir: Path, key: str, bandwidth: float,
                  normalization: str,
                  device: torch.device) -> GraphLaplacianOperator:
    """Rebuild the GraphLaplacianOperator from a cached graph.

    Note on `normalization`: laplacian_triu/_diag are the symmetric-normalized
    Laplacian entries regardless of this flag, and all three solvers operate on
    that symmetric L -- the randomwalk denormalization only happens in
    postprocess, which we deliberately bypass to isolate solver quality. So the
    flag is recorded for parity with the lgp experiments and threaded into the
    operator, but it does NOT change the eigenproblem being benchmarked.
    `bandwidth`, by contrast, scales L by 1/bw^2 and genuinely shifts the
    spectrum.
    """
    cache = KnnGraphCache(cache_dir=knn_dir, verbose=True)
    hit = cache.load(key, device=device)
    if hit is None:
        raise FileNotFoundError(f"No cached graph for key={key!r} in {knn_dir}")
    knn, edge_index, edge_value, fp, _meta = hit
    bw = torch.tensor(bandwidth, dtype=torch.float32, device=device)
    return GraphLaplacianOperator(
        x=edge_value,
        idx=edge_index,
        operator_dimension=knn.x.shape[0],
        graphbandwidth=bw,
        normalization=normalization,
        self_loops=True,
    )


def build_graph(knn_dir: Path, key: str, *, template: str, stride: int,
                threshold: float, knn_k: int, knn_method: str, nlist: int,
                reference_file: Optional[str], annotations_file: Optional[str],
                device: torch.device, verbose: bool = True) -> None:
    """Build + cache the KNN graph for `key` from the reference/annotation
    volumes, mirroring the lgp / compute_eigenvectors pipeline so the resulting
    cache key matches what those experiments produce.
    """
    from utils import crop_or_stride_volume, reference_ccf_from_subvolume

    if reference_file is None:
        raise SystemExit(
            "[compare] graph not cached and --reference-file not given, so it "
            "can't be built. Either pre-build the graph, or pass "
            "--reference-file (and --annotations-file for atlas methods)."
        )
    if verbose:
        print(f"[compare] building graph {key}\n"
              f"          stride={stride} threshold={threshold} k={knn_k} "
              f"method={knn_method} nlist={nlist}")

    reference_image = np.load(reference_file)
    annotation_volume = np.load(annotations_file) if annotations_file else None

    sub_volume, sub_atlas, voxel_offset, voxel_scale_mm = crop_or_stride_volume(
        reference_image, annotation_volume, stride)
    mm_coords = reference_ccf_from_subvolume(
        sub_volume, voxel_offset, voxel_scale_mm, threshold)
    # Normalize coords by this subvolume's own stats -- identical to how
    # compute_eigenvectors / lgp_manifold_experiment build the graph, so the
    # edge weights (squared distances) match.
    coord_mean = mm_coords.mean(axis=0)
    coord_std = mm_coords.std(axis=0)
    coords = torch.from_numpy(
        (mm_coords - coord_mean) / coord_std
    ).float().to(device).contiguous()
    if verbose:
        print(f"[compare] tissue nodes after threshold={threshold}: "
              f"{coords.shape[0]:,}")

    cache = KnnGraphCache(cache_dir=knn_dir, verbose=verbose)
    cache.train_or_load(
        key=key,
        method=knn_method,
        k=knn_k,
        nlist=nlist,
        coords=coords,
        volume=sub_volume,
        threshold=threshold,
        atlas_volume=sub_atlas,
        connectivity=1,
        extra={"template": template, "stride": stride, "thresh": threshold,
               "method": knn_method, "k": knn_k, "nlist": nlist, "bbox": None},
        device=device,
    )
    if verbose:
        print(f"[compare] graph built and cached -> {knn_dir}")


def assemble_L_csr(op: GraphLaplacianOperator):
    """Assemble the symmetric L as a SciPy CSR matrix (same recipe the solver
    uses internally), so all residual/metric math shares one matrix."""
    import scipy.sparse as sp
    N = op.operator_dimension
    idx = op.idx.cpu()
    val = op.laplacian_triu.cpu().numpy()
    diag = op.laplacian_diag.cpu().numpy()
    row, col = idx[0].numpy(), idx[1].numpy()
    L = sp.csr_matrix((-val, (row, col)), shape=(N, N))
    L = L + L.T
    L.setdiag(diag)
    return L.tocsr()


# ---------------------------------------------------------------------------
# Solvers -> raw (evals, evecs) as numpy, ascending by eigenvalue
# ---------------------------------------------------------------------------
def _to_sorted_numpy(evals, evecs) -> Tuple[np.ndarray, np.ndarray]:
    evals = evals.detach().cpu().numpy() if torch.is_tensor(evals) else np.asarray(evals)
    evecs = evecs.detach().cpu().numpy() if torch.is_tensor(evecs) else np.asarray(evecs)
    order = np.argsort(evals)
    return evals[order], evecs[:, order]


def run_cupy_sa(op, num_modes, ncv_min, tol) -> Tuple[np.ndarray, np.ndarray]:
    solver = LaplacianEigensolver(num_modes=num_modes, backend="cupy",
                                  tol=tol, ncv_min=ncv_min, verbose=True)
    evals, evecs = solver._compute_cupy(op)        # raw, pre-postprocess
    return _to_sorted_numpy(evals, evecs)


def run_scipy_si(op, num_modes, tol) -> Tuple[np.ndarray, np.ndarray]:
    solver = LaplacianEigensolver(num_modes=num_modes, backend="scipy",
                                  tol=tol, verbose=True)
    evals, evecs = solver._compute_scipy(op)       # raw, pre-postprocess
    return _to_sorted_numpy(evals, evecs)


def run_lobpcg(L_csr, num_modes, tol, maxiter, backend, seed=0
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Smallest `num_modes` modes via LOBPCG with a Jacobi preconditioner.

    backend: "cupy" runs on GPU (cupyx.scipy.sparse.linalg.lobpcg), "scipy" on
    CPU. The preconditioner is M = diag(L)^-1, the standard cheap choice that
    helps a lot on the clustered low end.
    """
    rng = np.random.default_rng(seed)
    N = L_csr.shape[0]
    X0 = rng.standard_normal((N, num_modes)).astype(np.float64)
    X0, _ = np.linalg.qr(X0)                        # orthonormal start block
    diag = L_csr.diagonal()
    inv_diag = np.where(np.abs(diag) > 1e-12, 1.0 / diag, 0.0)

    if backend == "cupy":
        import cupy as cp
        import cupyx.scipy.sparse as cps
        import cupyx.scipy.sparse.linalg as csl
        L = cps.csr_matrix(L_csr.astype(np.float64))
        Xg = cp.asarray(X0)
        invd = cp.asarray(inv_diag)
        M = csl.LinearOperator((N, N), matvec=lambda b: (b.T * invd).T)
        evals, evecs = csl.lobpcg(L, Xg, M=M, largest=False,
                                  tol=tol, maxiter=maxiter)
        evals = cp.asnumpy(evals)
        evecs = cp.asnumpy(evecs)
    else:
        import scipy.sparse.linalg as sla
        M = sla.LinearOperator((N, N), matvec=lambda b: (b.T * inv_diag).T)
        evals, evecs = sla.lobpcg(L_csr.astype(np.float64), X0, M=M,
                                  largest=False, tol=tol, maxiter=maxiter)
    return _to_sorted_numpy(evals, evecs)


def run_cupy_si(L_csr, num_modes, tol, sigma, ncv=None, verbose=True
                ) -> Tuple[np.ndarray, np.ndarray]:
    """GPU shift-invert: the accurate path scipy uses, done on the GPU.

    CuPy's eigsh has no `sigma`, so we build the shift-invert operator by hand:
    factorize M = (L - sigma*I) once with cupyx's GPU sparse LU, expose
    M^{-1} as a LinearOperator, and ask eigsh for the LARGEST eigenvalues of
    M^{-1} -- those map to the eigenvalues of L closest to sigma. We then undo
    the transform: if mu is an eigenvalue of M^{-1}, lambda = sigma + 1/mu.

    sigma is taken slightly BELOW 0 (the bottom of the Laplacian spectrum) so
    M = L + |sigma|*I is SPD and the factorization is stable, while the bottom
    modes still map to the largest |mu|. Each `lu.solve` call is one Lanczos
    step (the OPinv solve), so the progress bar counts Lanczos iterations.

    Unlike `cupy_sa` we use a LEAN ncv (~2k+1, not the 1500 floor): shift-invert
    converges in few restarts, so the big Krylov basis isn't needed -- that's
    the memory win this path is meant to show.

    PERFORMANCE CAVEAT (measured): cupyx's `splu` factorizes on the *CPU*
    (scipy SuperLU -- not GPU-accelerated; see its docstring) and only the
    triangular `solve` apply runs on the GPU, where the sequential dependency
    chain makes each single-RHS solve very slow (~0.8s even on a 24k graph).
    With hundreds-to-thousands of OPinv solves this dominates, so this path is
    the accuracy reference, NOT a fast solver. A genuinely fast GPU
    shift-invert needs a real GPU direct solver (cuDSS/cuSOLVER) or an iterative
    inner solve (CG/MINRES on the SPD M, GPU-SpMV-bound) with a preconditioner.
    """
    import cupy as cp
    import cupyx.scipy.sparse as cps
    import cupyx.scipy.sparse.linalg as csl
    from tqdm import tqdm

    N = L_csr.shape[0]
    L_cp = cps.csr_matrix(L_csr.astype(np.float64))
    ident = cps.identity(N, dtype=np.float64, format="csr")
    M = (L_cp - sigma * ident).tocsc()      # sigma<0 => M = L + |sigma| I (SPD)

    if verbose:
        print(f"[cupy_si] factorizing (L - sigma*I) on GPU, sigma={sigma:g} ...")
    t_fact = time.perf_counter()
    lu = csl.splu(M)
    fact_seconds = time.perf_counter() - t_fact
    if verbose:
        print(f"[cupy_si] GPU factorization done in {fact_seconds:.1f}s")

    pbar = tqdm(desc="[cupy_si] shift-invert OPinv solve",
                leave=False, disable=not verbose)
    counter = {"n": 0}

    def _solve(b):
        counter["n"] += 1
        pbar.update(1)
        return lu.solve(b)

    OP = csl.LinearOperator((N, N), matvec=_solve, dtype=np.float64)
    if ncv is None:
        ncv = min(N, max(2 * num_modes + 1, 20))

    mu, evecs = csl.eigsh(OP, k=num_modes, which="LA", tol=tol, ncv=ncv)
    cp.cuda.Stream.null.synchronize()
    pbar.close()
    if verbose:
        print(f"[cupy_si] OPinv solves (Lanczos iters): {counter['n']}")

    evals = cp.asnumpy(sigma + 1.0 / mu)
    evecs = cp.asnumpy(evecs)
    return _to_sorted_numpy(evals, evecs)


def run_cupy_si_cg(L_csr, num_modes, tol, sigma, inner_rtol=1e-6,
                   inner_maxiter=1000, precondition=True, ncv=None,
                   verbose=True) -> Tuple[np.ndarray, np.ndarray]:
    """Iterative (matrix-free) shift-invert -- GPU SpMV only, no factorization.

    Outer Lanczos (eigsh, which="LA") on OP = M^{-1}, but instead of
    factorizing M we apply M^{-1} with an INNER conjugate-gradient solve of
    M x = b, where M = L - sigma*I = L + |sigma|*I is SPD (sigma < 0). CG needs
    only SpMVs with M -- the GPU's best case -- so this avoids cupyx splu's CPU
    factorization and its pathologically slow GPU triangular solve. Eigenvalues
    map back via lambda = sigma + 1/mu.

    The knob is sigma: kappa(M) = (lambda_max + |sigma|)/|sigma|, so smaller
    |sigma| amplifies the low-mode gaps (faster OUTER convergence) but worsens
    M's conditioning (slower INNER CG). A Jacobi preconditioner on M cuts the
    inner count. `inner_rtol` must be tight enough not to corrupt the outer
    Lanczos (inexact shift-invert); 1e-6 is a safe default.

    Reports outer Lanczos steps and total/avg inner CG iterations so you can see
    where the SpMV budget goes.
    """
    import cupy as cp
    import cupyx.scipy.sparse as cps
    import cupyx.scipy.sparse.linalg as csl
    from tqdm import tqdm

    N = L_csr.shape[0]
    L_cp = cps.csr_matrix(L_csr.astype(np.float64))
    M = (L_cp - sigma * cps.identity(N, dtype=np.float64, format="csr")).tocsr()

    Mprec = None
    if precondition:
        d = M.diagonal()
        inv_d = cp.where(cp.abs(d) > 1e-30, 1.0 / d, 0.0)
        Mprec = csl.LinearOperator((N, N), matvec=lambda b: b * inv_d)

    pbar = tqdm(desc="[cupy_si_cg] outer Lanczos (inner CG SpMV)",
                leave=False, disable=not verbose)
    stats = {"outer": 0, "inner": 0, "nonconv": 0}

    def _opinv(b):
        stats["outer"] += 1
        pbar.update(1)
        inner = {"n": 0}
        x, info = csl.cg(M, b, rtol=inner_rtol, maxiter=inner_maxiter, M=Mprec,
                         callback=lambda xk: inner.__setitem__("n", inner["n"] + 1))
        stats["inner"] += inner["n"]
        if info != 0:
            stats["nonconv"] += 1
        return x

    OP = csl.LinearOperator((N, N), matvec=_opinv, dtype=np.float64)
    if ncv is None:
        ncv = min(N, max(2 * num_modes + 1, 20))

    mu, evecs = csl.eigsh(OP, k=num_modes, which="LA", tol=tol, ncv=ncv)
    cp.cuda.Stream.null.synchronize()
    pbar.close()
    if verbose:
        avg = stats["inner"] / max(1, stats["outer"])
        print(f"[cupy_si_cg] outer solves={stats['outer']}  "
              f"total inner CG SpMVs={stats['inner']}  "
              f"avg inner/outer={avg:.1f}  inner non-converged={stats['nonconv']}")
        if stats["nonconv"]:
            print(f"[cupy_si_cg] WARNING: {stats['nonconv']} inner solves hit "
                  f"maxiter={inner_maxiter} -- raise --si-cg-inner-maxiter or "
                  f"loosen sigma; the outer eigenpairs may be inaccurate.")

    evals = cp.asnumpy(sigma + 1.0 / mu)
    evecs = cp.asnumpy(evecs)
    return _to_sorted_numpy(evals, evecs)


def run_slepc(L_csr, num_modes, tol, *, shift_invert, target=0.0,
              factor_solver="mumps", ncv=None) -> Tuple[np.ndarray, np.ndarray]:
    """Bottom `num_modes` modes via SLEPc, in-process and serial (COMM_SELF).

    shift_invert=True factorizes (L - target*I) with a 64-bit parallel direct
    solver (MUMPS / SuperLU_DIST). That is the path that survives the 3D fill-in
    where scipy's 32-bit SuperLU raises "Can't expand MemType" -- so it is the
    feasible gold standard at large N. shift_invert=False is matrix-free
    Krylov-Schur on smallest-real (no factorization; cheap memory, slower to
    converge on the clustered low spectrum).

    Runs serially: PETSc COMM_SELF with one rank. The MUMPS advantage is the
    64-bit indexing + better ordering, not the parallelism, so a single process
    already fixes the overflow. For multi-rank speedups use slepc_eigensolve.py
    under mpirun.
    """
    import sys
    import petsc4py
    petsc4py.init(sys.argv)
    from petsc4py import PETSc          # noqa: E402
    from slepc4py import SLEPc          # noqa: E402

    comm = PETSc.COMM_SELF
    A = L_csr.tocsr()
    Amat = PETSc.Mat().createAIJWithArrays(
        size=(A.shape[0], A.shape[1]),
        csr=(A.indptr.astype(PETSc.IntType),
             A.indices.astype(PETSc.IntType),
             A.data.astype(PETSc.ScalarType)),
        comm=comm)
    Amat.assemble()

    E = SLEPc.EPS().create(comm=comm)
    E.setOperators(Amat)
    E.setProblemType(SLEPc.EPS.ProblemType.HEP)
    E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    if ncv is not None and ncv > 0:
        E.setDimensions(nev=num_modes, ncv=ncv)
    else:
        E.setDimensions(nev=num_modes)
    E.setTolerances(tol=tol)

    if shift_invert:
        E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
        E.setTarget(target)
        st = E.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        ksp = st.getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        try:
            pc.setFactorSolverType(factor_solver)
        except Exception:
            print(f"[slepc] factor solver {factor_solver!r} unavailable -- using "
                  "PETSc built-in LU (serial, may overflow on large fill).")
    else:
        E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)

    # Live progress: SLEPc's per-iteration monitor reports how many of the wanted
    # modes have converged. The low Laplacian spectrum is clustered, so nconv
    # sits flat then jumps in bursts -- the frontier residual (error of the next
    # mode about to converge, shrinking toward tol) gives motion during the flat
    # stretches. Defensive: a monitor error must never abort the solve.
    kind = "shift-invert" if shift_invert else "krylov-schur"
    bar = None
    try:
        from tqdm import tqdm
        bar = tqdm(total=num_modes, desc=f"[slepc] {kind} modes converged",
                   leave=False)
    except Exception:
        bar = None
    seen = {"n": 0}

    def _monitor(eps, its, nconv, eig, err):
        if bar is None:
            return
        try:
            if nconv > seen["n"]:
                bar.update(nconv - seen["n"])
                seen["n"] = nconv
            frontier = ""
            if err is not None and nconv < len(err):
                frontier = f" res={float(err[nconv]):.1e}"
            bar.set_postfix_str(f"iter={its}{frontier}")
        except Exception:
            pass

    try:
        E.setMonitor(_monitor)
    except Exception:
        pass

    E.setFromOptions()
    E.solve()
    if bar is not None:
        bar.close()
    nconv = E.getConverged()
    if nconv == 0:
        raise RuntimeError("slepc converged 0 eigenpairs -- raise --slepc-ncv-min "
                           "or loosen --tol")
    k = min(nconv, num_modes)
    if k < num_modes:
        print(f"[slepc] WARNING: only {k}/{num_modes} modes converged")
    evals = np.empty(k)
    evecs = np.empty((A.shape[0], k))
    xr = Amat.createVecRight()
    gather = range(k)
    try:
        from tqdm import tqdm
        gather = tqdm(gather, desc="[slepc] gathering eigenvectors", leave=False)
    except Exception:
        pass
    for i in gather:
        evals[i] = E.getEigenpair(i, xr).real
        evecs[:, i] = xr.getArray().real
    return _to_sorted_numpy(evals, evecs)


def run_reference(L_csr, op, num_modes, max_dense, tol, mode="auto", *,
                  slepc_target=0.0, slepc_factor_solver="mumps", slepc_ncv=None
                  ) -> Tuple[np.ndarray, np.ndarray, str]:
    """Gold-standard eigenpairs. 'auto' = dense eigh when N <= max_dense (exact),
    else scipy shift-invert. Explicit modes: 'dense', 'scipy_si', 'slepc_si'
    (MUMPS shift-invert -- the feasible truth at large N), 'slepc_ks'."""
    N = L_csr.shape[0]
    if mode == "dense" or (mode == "auto" and N <= max_dense):
        import scipy.linalg as la
        w, v = la.eigh(L_csr.toarray())
        return w[:num_modes], v[:, :num_modes], f"dense eigh (N={N})"
    if mode == "slepc_si":
        ev, evec = run_slepc(L_csr, num_modes, tol, shift_invert=True,
                             target=slepc_target,
                             factor_solver=slepc_factor_solver, ncv=slepc_ncv)
        return ev, evec, f"slepc shift-invert ({slepc_factor_solver})"
    if mode == "slepc_ks":
        ev, evec = run_slepc(L_csr, num_modes, tol, shift_invert=False,
                             ncv=slepc_ncv)
        return ev, evec, "slepc krylov-schur"
    # mode in ("auto" with large N, "scipy_si")
    evals, evecs = run_scipy_si(op, num_modes, tol)
    return evals, evecs, "scipy shift-invert"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def per_mode_residual(L_csr, evals, evecs) -> np.ndarray:
    """||L v_i - lambda_i v_i||_2 for each column (vectors L2-normalized first)."""
    V = evecs / (np.linalg.norm(evecs, axis=0, keepdims=True) + 1e-30)
    R = L_csr @ V - V * evals[None, :]
    return np.linalg.norm(R, axis=0)


def orthonormality_error(evecs) -> float:
    V = evecs / (np.linalg.norm(evecs, axis=0, keepdims=True) + 1e-30)
    G = V.T @ V
    return float(np.linalg.norm(G - np.eye(G.shape[0]), ord="fro")
                 / np.sqrt(G.shape[0]))


def principal_angles_deg(Va, Vb) -> np.ndarray:
    """Principal angles (degrees) between the two column spans.

    The fair eigenvector metric under clustering: individual vectors may rotate
    within a degenerate cluster, but the *subspace* should still match.

    Uses the Björck-Golub / Knyazev stable formulation: `arccos` loses all
    precision for small angles (cos near 1), so we compute those from the
    *sine* (the residual of Qb projected off Qa) via `arcsin`, and only use
    `arccos` for the large angles. Without this, identical subspaces report a
    spurious ~0.04 deg floor instead of 0.
    """
    Qa, _ = np.linalg.qr(Va)
    Qb, _ = np.linalg.qr(Vb)
    M = Qa.T @ Qb
    cos_s = np.clip(np.linalg.svd(M, compute_uv=False), -1.0, 1.0)
    # Singular values of the off-Qa residual of Qb are sin(theta); svd returns
    # them descending (largest angle first), so reverse to align with cos_s
    # (ascending angle).
    sin_s = np.clip(np.linalg.svd(Qb - Qa @ M, compute_uv=False), 0.0, 1.0)[::-1]
    small = cos_s ** 2 >= 0.5      # angle < 45 deg -> trust the sine
    angles = np.where(small, np.arcsin(sin_s), np.arccos(cos_s))
    return np.degrees(angles)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--knn-dir", required=True, type=Path,
                   help="KnnGraphCache directory (the 'knn' folder).")
    # --- graph selection: either pass the raw key, or compose it from the same
    #     options the lgp experiments use (template/stride/threshold/k/...) ---
    p.add_argument("--knn-key", default=None,
                   help="Graph key. If omitted, composed from the options below "
                        "(same scheme as the lgp experiments).")
    p.add_argument("--template", default="reference", help="Template name.")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--threshold", type=float, default=50.0)
    p.add_argument("--knn-k", type=int, default=15)
    p.add_argument("--knn-method", default="faiss")
    p.add_argument("--nlist", type=int, default=1)
    # --- graph building (on cache miss) ---
    p.add_argument("--build-if-missing", action="store_true",
                   help="If the graph isn't cached, build it from the volumes "
                        "below instead of erroring.")
    p.add_argument("--reference-file", default=None,
                   help="Reference volume .npy (required to build a missing graph).")
    p.add_argument("--annotations-file", default=None,
                   help="Annotations volume .npy (for atlas-based knn methods).")
    # --- Laplacian options ---
    p.add_argument("--bandwidth", type=float, default=1.0,
                   help="Graph bandwidth; scales L by 1/bw^2 (real spectral effect).")
    p.add_argument("--normalization", choices=["symmetric", "randomwalk"],
                   default="symmetric",
                   help="Recorded for lgp parity; affects only postprocess, not "
                        "the benchmarked symmetric eigenproblem.")
    p.add_argument("--modes", type=int, default=200)
    p.add_argument("--tol", type=float, default=1e-8,
                   help="Solver tolerance (matches production default).")
    p.add_argument("--ncv-min", type=int, default=-1,
                   help="Lanczos Krylov floor for cupy_sa; <=0 auto.")
    p.add_argument("--lobpcg-maxiter", type=int, default=200)
    p.add_argument("--lobpcg-backend", choices=["auto", "cupy", "scipy"],
                   default="auto")
    p.add_argument("--si-sigma", type=float, default=-1e-3,
                   help="Shift for cupy_si GPU shift-invert. Keep it slightly "
                        "below 0 so (L - sigma*I) is SPD.")
    p.add_argument("--si-ncv-min", type=int, default=-1,
                   help="Krylov floor for cupy_si / cupy_si_cg; <=0 uses a lean "
                        "2k+1 (shift-invert needs no big basis).")
    p.add_argument("--si-cg-inner-rtol", type=float, default=1e-6,
                   help="Inner CG tolerance for cupy_si_cg. Too loose corrupts "
                        "the outer Lanczos (inexact shift-invert).")
    p.add_argument("--si-cg-inner-maxiter", type=int, default=1000,
                   help="Inner CG iteration cap for cupy_si_cg.")
    p.add_argument("--si-cg-no-precond", action="store_true",
                   help="Disable the Jacobi preconditioner on the inner CG.")
    p.add_argument("--max-dense", type=int, default=6000,
                   help="Use exact dense reference when N <= this.")
    p.add_argument("--reference",
                   choices=["auto", "none", "dense", "scipy_si",
                            "slepc_si", "slepc_ks"],
                   default="auto",
                   help="'auto' = dense (small N) or scipy shift-invert; "
                        "'slepc_si' = MUMPS 64-bit shift-invert (the feasible "
                        "gold standard at large N where scipy's SuperLU "
                        "overflows); 'slepc_ks' = Krylov-Schur; 'none' = skip "
                        "the reference and report only the self-contained "
                        "residual / orthonormality / perf columns.")
    p.add_argument("--slepc-factor-solver", default="mumps",
                   choices=["mumps", "superlu_dist", "petsc"],
                   help="Direct solver for slepc_si shift-invert factorization. "
                        "mumps/superlu_dist are 64-bit (needed for large fill).")
    p.add_argument("--slepc-target", type=float, default=0.0,
                   help="slepc_si shift-invert target, near the spectrum bottom.")
    p.add_argument("--slepc-ncv-min", type=int, default=-1,
                   help="Krylov basis floor for slepc_si / slepc_ks; <=0 lets "
                        "SLEPc choose.")
    p.add_argument("--max-direct-nodes", type=int, default=2_000_000,
                   help="Above this N, skip the direct-factorization methods "
                        "(scipy_si, cupy_si) and any scipy reference -- their LU "
                        "fill-in overflows memory / SuperLU's 32-bit indexing.")
    p.add_argument("--reuse-reference", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="When the reference is scipy shift-invert (large N), "
                        "reuse it for the scipy_si row instead of solving the "
                        "same problem twice. --no-reuse-reference forces an "
                        "independent second solve (e.g. as a noise check).")
    p.add_argument("--approaches", default="cupy_sa,scipy_si,lobpcg",
                   help="Comma list subset of cupy_sa,cupy_si,cupy_si_cg,"
                        "scipy_si,lobpcg,slepc_si,slepc_ks.")
    p.add_argument("--monitor-interval", type=float, default=0.1,
                   help="Resource sampling interval (seconds).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path prefix for JSON + PNG report.")
    args = p.parse_args()

    device = torch.device(args.device)
    gpu_index = device.index if device.index is not None else (
        torch.cuda.current_device() if device.type == "cuda" else 0)
    cupy_available = False
    try:
        import cupy  # noqa: F401
        cupy_available = torch.cuda.is_available()
    except Exception:
        pass

    # ---- resolve key: explicit --knn-key wins, else compose from options ----
    if args.knn_key is None:
        args.knn_key = make_graph_key({
            "template": args.template,
            "stride": args.stride,
            "thresh": args.threshold,
            "method": args.knn_method,
            "k": args.knn_k,
            "nlist": args.nlist,
            "bbox": None,
        })
        print(f"[compare] composed key: {args.knn_key}")
    npz = Path(args.knn_dir) / f"{args.knn_key}.knngraph.npz"
    if not npz.exists():
        if args.build_if_missing:
            print(f"[compare] no cached graph at {npz} -- building")
            build_graph(
                args.knn_dir, args.knn_key, template=args.template,
                stride=args.stride, threshold=args.threshold, knn_k=args.knn_k,
                knn_method=args.knn_method, nlist=args.nlist,
                reference_file=args.reference_file,
                annotations_file=args.annotations_file, device=device)
        else:
            print(f"[compare] no cached graph at {npz}")
            print("[compare] available keys in this dir:")
            for k in discover_keys(args.knn_dir):
                print(f"    {k}")
            raise SystemExit("Adjust the graph options, pass --knn-key, or "
                             "use --build-if-missing with --reference-file.")

    # ---- load graph + assemble L ----
    print(f"[compare] loading graph {args.knn_key} from {args.knn_dir}")
    op = load_operator(args.knn_dir, args.knn_key, args.bandwidth,
                       args.normalization, device)
    L_csr = assemble_L_csr(op)
    N = L_csr.shape[0]
    lam_max_est = float(L_csr.diagonal().max()) * 2.0  # cheap upper-ish scale
    print(f"[compare] N={N:,}  nnz={L_csr.nnz:,}  modes={args.modes}")

    ncv_min = resolve_ncv_min(args.modes, args.ncv_min)
    lobpcg_backend = ("cupy" if (args.lobpcg_backend == "auto" and cupy_available)
                      else "scipy" if args.lobpcg_backend == "auto"
                      else args.lobpcg_backend)

    # ---- reference ----
    # The reference is the "truth" the eval-error and subspace-angle columns are
    # scored against; the residual / orthonormality / perf columns are
    # reference-free. At very large N there is no feasible gold standard (dense
    # is too big, scipy's LU overflows), so --reference none drops the
    # reference-dependent columns and reports only the self-contained metrics.
    want_reference = (args.reference != "none")
    # The direct-node ceiling only forces reference-free for the scipy/dense
    # gold standards (their LU overflows). slepc_si uses 64-bit MUMPS and stays
    # feasible at large N, so it is exempt from the ceiling.
    ref_is_overflow_prone = args.reference in ("auto", "scipy_si", "dense")
    if (want_reference and ref_is_overflow_prone
            and N > args.max_direct_nodes and N > args.max_dense):
        print(f"[compare] N={N:,} exceeds --max-direct-nodes "
              f"({args.max_direct_nodes:,}) and --max-dense ({args.max_dense:,}) "
              f"-- no feasible scipy/dense reference (LU would overflow). Pass "
              f"--reference slepc_si for a MUMPS gold standard, or proceed "
              f"reference-free.")
        want_reference = False

    if want_reference:
        print("[compare] computing reference eigenpairs ...")
        t0 = time.perf_counter()
        slepc_ncv = None if args.slepc_ncv_min <= 0 else args.slepc_ncv_min
        ref_evals, ref_evecs, ref_name = run_reference(
            L_csr, op, args.modes, args.max_dense, args.tol, mode=args.reference,
            slepc_target=args.slepc_target,
            slepc_factor_solver=args.slepc_factor_solver, slepc_ncv=slepc_ncv)
        ref_wall = time.perf_counter() - t0
        if ref_name.startswith("scipy"):
            ref_method = "scipy_si"
        elif ref_name.startswith("slepc shift"):
            ref_method = "slepc_si"
        elif ref_name.startswith("slepc krylov"):
            ref_method = "slepc_ks"
        else:
            ref_method = "dense"
        print(f"[compare] reference = {ref_name}  ({ref_wall:.1f}s)")
        lam_max = float(ref_evals[-1]) if ref_evals[-1] > 0 else lam_max_est
    else:
        ref_evals = ref_evecs = None
        ref_name = "none (reference-free metrics only)"
        ref_method = None
        ref_wall = 0.0
        # No reference eigenvalue to set the spectral scale; use a cheap
        # Gershgorin-style upper bound (2*max diagonal) to normalize residuals.
        lam_max = lam_max_est
        print(f"[compare] reference = {ref_name}")

    si_ncv = None if args.si_ncv_min <= 0 else args.si_ncv_min
    runners = {
        "cupy_sa": lambda: run_cupy_sa(op, args.modes, ncv_min, args.tol),
        "cupy_si": lambda: run_cupy_si(L_csr, args.modes, args.tol,
                                       args.si_sigma, ncv=si_ncv),
        "cupy_si_cg": lambda: run_cupy_si_cg(
            L_csr, args.modes, args.tol, args.si_sigma,
            inner_rtol=args.si_cg_inner_rtol,
            inner_maxiter=args.si_cg_inner_maxiter,
            precondition=not args.si_cg_no_precond, ncv=si_ncv),
        "scipy_si": lambda: run_scipy_si(op, args.modes, args.tol),
        "lobpcg": lambda: run_lobpcg(L_csr, args.modes, args.tol,
                                     args.lobpcg_maxiter, lobpcg_backend),
        "slepc_si": lambda: run_slepc(
            L_csr, args.modes, args.tol, shift_invert=True,
            target=args.slepc_target, factor_solver=args.slepc_factor_solver,
            ncv=(None if args.slepc_ncv_min <= 0 else args.slepc_ncv_min)),
        "slepc_ks": lambda: run_slepc(
            L_csr, args.modes, args.tol, shift_invert=False,
            ncv=(None if args.slepc_ncv_min <= 0 else args.slepc_ncv_min)),
    }
    wanted = [a.strip() for a in args.approaches.split(",") if a.strip()]
    for gpu_only in ("cupy_sa", "cupy_si", "cupy_si_cg"):
        if gpu_only in wanted and not cupy_available:
            print(f"[compare] cupy not available -- skipping {gpu_only}")
            wanted.remove(gpu_only)
    if any(s in wanted for s in ("slepc_si", "slepc_ks")):
        try:
            import slepc4py  # noqa: F401
            import petsc4py  # noqa: F401
        except Exception:
            for s in ("slepc_si", "slepc_ks"):
                if s in wanted:
                    print(f"[compare] slepc4py/petsc4py not available -- "
                          f"skipping {s}")
                    wanted.remove(s)
    # Direct-factorization methods (scipy_si, cupy_si) build an LU whose 3D
    # fill-in blows past memory / SuperLU's 32-bit indexing at large N. Skip
    # them above the node ceiling rather than crash or hang.
    if N > args.max_direct_nodes:
        for direct in ("scipy_si", "cupy_si"):
            if direct in wanted:
                print(f"[compare] N={N:,} > --max-direct-nodes "
                      f"({args.max_direct_nodes:,}) -- skipping {direct} "
                      f"(direct factorization infeasible at this size).")
                wanted.remove(direct)

    results: Dict[str, dict] = {}
    for name in wanted:
        print(f"\n[compare] === {name} (lobpcg backend={lobpcg_backend}) ===")
        # At large N the reference itself is scipy shift-invert, so running
        # scipy_si as a contestant just solves the same thing again (~minutes).
        # Reuse the reference solve for that row instead of recomputing.
        if (args.reuse_reference and ref_method is not None
                and name == ref_method):
            print(f"[compare] {name} == reference ({ref_name}) -- reusing the "
                  f"reference solve, skipping redundant recompute")
            evals, evecs = ref_evals, ref_evecs
            wall, perf = ref_wall, {}
        else:
            try:
                with ResourceMonitor(gpu_index=gpu_index,
                                     interval=args.monitor_interval,
                                     enable_gpu=(device.type == "cuda")) as mon:
                    t0 = time.perf_counter()
                    evals, evecs = runners[name]()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    wall = time.perf_counter() - t0
                perf = mon.stats()
            except Exception as e:
                print(f"[compare] {name} FAILED: {e}")
                results[name] = {"failed": str(e)}
                continue

        res = per_mode_residual(L_csr, evals, evecs)
        results[name] = {
            "wall_seconds": wall,
            **perf,
            "residual_abs_max": float(res.max()),
            "residual_abs_mean": float(res.mean()),
            "residual_rel_max": float(res.max() / lam_max),
            "orthonormality_err": orthonormality_error(evecs),
            "_per_mode_residual": res,
            "_evals": evals,
        }
        # Reference-dependent metrics only when we have a reference.
        if ref_evals is not None:
            angles = principal_angles_deg(ref_evecs, evecs)
            ev_err = np.abs(evals - ref_evals)
            results[name].update({
                "eval_abs_err_max": float(ev_err.max()),
                "eval_abs_err_mean": float(ev_err.mean()),
                "eval_rel_err_max": float((ev_err / lam_max).max()),
                "subspace_angle_max_deg": float(angles.max()),
                "subspace_overlap_mean": float(np.cos(np.radians(angles)).mean()),
                "_eval_abs_err": ev_err,
                "_angles": np.sort(angles)[::-1],
            })

    # ---- table ----
    # Reference-dependent columns are dropped when running reference-free.
    ref_dependent = {"eval_abs_err_max", "eval_rel_err_max",
                     "subspace_angle_max_deg"}
    cols = [c for c in (
        ("wall_seconds", "wall(s)", "{:.2f}"),
        ("eval_abs_err_max", "λ|err|max", "{:.2e}"),
        ("eval_rel_err_max", "λ relerr", "{:.2e}"),
        ("residual_abs_max", "resid max", "{:.2e}"),
        ("residual_rel_max", "resid/λmax", "{:.2e}"),
        ("orthonormality_err", "ortho err", "{:.2e}"),
        ("subspace_angle_max_deg", "maxAngle°", "{:.3f}"),
    ) if ref_evals is not None or c[0] not in ref_dependent]
    perf_cols = [
        ("wall_seconds", "wall(s)", "{:.2f}"),
        ("gpu_util_mean_pct", "gpu%mean", "{:.0f}"),
        ("gpu_util_peak_pct", "gpu%peak", "{:.0f}"),
        ("gpu_mem_delta_mb", "gpuΔMB", "{:.0f}"),
        ("torch_peak_mb", "torchMB", "{:.0f}"),
        ("host_rss_delta_mb", "hostΔMB", "{:.0f}"),
    ]

    def _print_table(title, columns):
        print(f"\n[compare] {title}")
        header = f"{'approach':<10} " + " ".join(f"{h:>11}" for _, h, _ in columns)
        print(header)
        print("-" * len(header))
        for name in wanted:
            r = results.get(name, {})
            if "failed" in r:
                print(f"{name:<10} FAILED: {r['failed'][:60]}")
                continue
            row = f"{name:<10} " + " ".join(
                f"{fmt.format(r.get(k, float('nan'))):>11}" for k, _, fmt in columns)
            print(row)

    print(f"\n[compare] reference: {ref_name}   N={N:,}  modes={args.modes}  "
          f"λmax={lam_max:.4g}  bw={args.bandwidth}  norm={args.normalization}")
    _print_table("accuracy", cols)
    _print_table("performance (GPU mem via nvidia-smi, device-wide delta)", perf_cols)

    # ---- optional report ----
    if args.out is not None:
        _write_report(args.out, args, ref_name, N, lam_max, results, wanted)


def _write_report(out: Path, args, ref_name, N, lam_max, results, wanted):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "knn_key": args.knn_key, "N": N, "modes": args.modes,
        "bandwidth": args.bandwidth, "normalization": args.normalization,
        "stride": args.stride, "threshold": args.threshold, "knn_k": args.knn_k,
        "knn_method": args.knn_method, "tol": args.tol,
        "reference": ref_name, "lam_max": lam_max,
        "approaches": {
            name: {k: v for k, v in r.items() if not k.startswith("_")}
            for name, r in results.items()
        },
    }
    json_path = out.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[compare] wrote {json_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[compare] matplotlib unavailable, skipping plot: {e}")
        return

    # The eval-error / principal-angle panels only exist with a reference.
    have_ref = any("_eval_abs_err" in results.get(n, {}) for n in wanted)
    n_panels = 3 if have_ref else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5.3 * n_panels, 4.5),
                             squeeze=False)
    axes = axes[0]
    for name in wanted:
        r = results.get(name, {})
        if "failed" in r:
            continue
        axes[0].semilogy(r["_per_mode_residual"], label=name, lw=1.2)
        if have_ref and "_eval_abs_err" in r:
            axes[1].semilogy(np.maximum(r["_eval_abs_err"], 1e-16), label=name, lw=1.2)
            axes[2].plot(r["_angles"], label=name, lw=1.2)
    axes[0].set(title="per-mode residual ||Lv-λv||", xlabel="mode", ylabel="residual")
    if have_ref:
        axes[1].set(title="per-mode |λ error| vs reference", xlabel="mode", ylabel="abs err")
        axes[2].set(title="principal angles vs reference", xlabel="rank", ylabel="degrees")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"eigensolver comparison  N={N:,}  modes={args.modes}  ref={ref_name}")
    fig.tight_layout()
    png_path = out.with_suffix(".png")
    fig.savefig(png_path, dpi=130)
    print(f"[compare] wrote {png_path}")


if __name__ == "__main__":
    main()
