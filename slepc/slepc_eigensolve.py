#!/usr/bin/env python
# encoding: utf-8
"""
MPI-parallel SLEPc eigensolver for the graph Laplacian.

Computes the bottom-`modes` eigenpairs of the symmetric graph Laplacian with
SLEPc (Krylov-Schur, or shift-invert via a parallel direct solver), distributed
across MPI ranks via PETSc. The result is gathered to rank 0, postprocessed
identically to ``manifold_gp.utils.compute_eigenvectors.LaplacianEigensolver``,
and saved in the SAME ``<key>.eigpairs.npz`` / ``.meta.json`` cache layout -- so
the existing GP / manifold code reuses these eigenvectors with no changes.

Run it under mpirun (one pod, ranks = CPU cores):

    mpirun -n <P> python slepc_eigensolve.py \
        --stride 1 --threshold 5 --knn-k 15 --modes 300 \
        --eigenvector-dir /s3/.../eigenvectors \
        --reference-file /s3/.../reference_image.npy \
        --annotations-file /s3/.../level_15annot.npy \
        --build-if-missing

For multi-node, launch the SAME script under a distributed MPI workload (the MPI
operator provides mpirun across pods); the code is partition-agnostic.

Why SLEPc here: Krylov-Schur restarts more memory-efficiently than ARPACK for
many modes, and the shift-invert path uses a 64-bit parallel direct solver
(MUMPS / SuperLU_DIST) that can factorize fill-in scipy's 32-bit SuperLU cannot
(the stride<=2 regime where scipy.splu raised "gstrf invalid arguments").

Requires: petsc4py, slepc4py (PETSc built --with-mumps for the shift-invert
path), mpi4py, faiss (for build-if-missing), torch, and the manifold_gp package.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc          # noqa: E402
from slepc4py import SLEPc          # noqa: E402

import torch                        # noqa: E402

from manifold_gp.utils.nearest_neighbors import (   # noqa: E402
    KnnGraphCache, make_key as make_graph_key,
)
from manifold_gp.utils.compute_eigenvectors import (  # noqa: E402
    LaplacianEigensolver, make_key as make_eig_key,
)
from manifold_gp.operators.graph_laplacian_operator import (  # noqa: E402
    GraphLaplacianOperator,
)

# Reuse the exact graph-build helpers the rest of the pipeline uses. They live
# in maldi/utils.py; add that dir to the path so build-if-missing matches the
# cache keys produced by compute_eigenvectors / lgp_manifold_experiment.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "maldi"))

COMM = PETSc.COMM_WORLD
RANK = COMM.getRank()
SIZE = COMM.getSize()


def log(msg: str) -> None:
    if RANK == 0:
        print(f"[slepc] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Graph: build (rank 0) if missing, then every rank loads the cache
# ---------------------------------------------------------------------------
def ensure_graph(knn_dir: Path, key: str, args) -> None:
    """Rank 0 builds + caches the KNN graph if it isn't present. Collective:
    all ranks wait on the barrier so they can load it afterwards."""
    npz = Path(knn_dir) / f"{key}.knngraph.npz"
    if RANK == 0 and not npz.exists():
        if not args.build_if_missing:
            raise SystemExit(
                f"[slepc] graph not cached at {npz} and --build-if-missing not "
                f"set. Pre-build it (e.g. with maldi/compute_eigenvectors) or "
                f"pass --build-if-missing with --reference-file."
            )
        if not args.reference_file:
            raise SystemExit(
                "[slepc] --build-if-missing needs --reference-file (and "
                "--annotations-file for atlas methods)."
            )
        from utils import crop_or_stride_volume, reference_ccf_from_subvolume
        log(f"building graph {key} (stride={args.stride}, thr={args.threshold}, "
            f"k={args.knn_k}, method={args.knn_method}) on rank 0 ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        reference_image = np.load(args.reference_file)
        annotation_volume = (np.load(args.annotations_file)
                             if args.annotations_file else None)
        sub_volume, sub_atlas, voxel_offset, voxel_scale_mm = crop_or_stride_volume(
            reference_image, annotation_volume, args.stride)
        mm_coords = reference_ccf_from_subvolume(
            sub_volume, voxel_offset, voxel_scale_mm, args.threshold)
        coords = torch.from_numpy(
            (mm_coords - mm_coords.mean(0)) / mm_coords.std(0)
        ).float().to(device).contiguous()
        KnnGraphCache(cache_dir=knn_dir, verbose=True).train_or_load(
            key=key, method=args.knn_method, k=args.knn_k, nlist=args.nlist,
            coords=coords, volume=sub_volume, threshold=args.threshold,
            atlas_volume=sub_atlas, connectivity=1,
            extra={"template": args.template, "stride": args.stride,
                   "thresh": args.threshold, "method": args.knn_method,
                   "k": args.knn_k, "nlist": args.nlist, "bbox": None},
            device=device)
        log("graph build complete.")
    COMM.barrier()


def load_operator(knn_dir: Path, key: str, bandwidth: float, normalization: str):
    """Every rank loads the cached graph (CPU) and builds the symmetric
    GraphLaplacianOperator. Returns (op, edge_index, edge_value, N)."""
    hit = KnnGraphCache(cache_dir=knn_dir, verbose=False).load(key, device="cpu")
    if hit is None:
        raise SystemExit(f"[slepc] graph {key} missing after ensure_graph()")
    knn, edge_index, edge_value, _fp, _meta = hit
    N = int(knn.x.shape[0])
    op = GraphLaplacianOperator(
        edge_value, edge_index, N,
        torch.tensor(bandwidth, dtype=torch.float32),
        normalization, self_loops=True)
    return op, edge_index, edge_value, N


# ---------------------------------------------------------------------------
# Build the distributed PETSc matrix for this rank's row block
# ---------------------------------------------------------------------------
def build_petsc_matrix(op, N: int):
    """Assemble the symmetric L (scipy CSR) and hand each rank a contiguous row
    block as a distributed PETSc AIJ matrix.

    Memory note: each rank materializes the full CSR (~12 bytes/nnz) before
    slicing its block. For very large N you'd pre-convert L to a PETSc binary
    file and MatLoad it; for the moderate node counts here, build-and-slice is
    simpler and fine.
    """
    import scipy.sparse as sp
    idx = op.idx.cpu()
    val = op.laplacian_triu.cpu().numpy()
    diag = op.laplacian_diag.cpu().numpy()
    row, col = idx[0].numpy(), idx[1].numpy()
    L = sp.csr_matrix((-val, (row, col)), shape=(N, N))
    L = L + L.T
    L.setdiag(diag)
    L = L.tocsr()

    # Contiguous block partition matching PETSc's default ownership.
    base, rem = divmod(N, SIZE)
    n_local = base + (1 if RANK < rem else 0)
    r0 = RANK * base + min(RANK, rem)
    r1 = r0 + n_local
    sub = L[r0:r1].tocsr()
    del L

    A = PETSc.Mat().createAIJWithArrays(
        size=((n_local, N), (n_local, N)),
        csr=(sub.indptr.astype(PETSc.IntType),
             sub.indices.astype(PETSc.IntType),
             sub.data.astype(PETSc.ScalarType)),
        comm=COMM,
    )
    A.assemble()
    return A


# ---------------------------------------------------------------------------
# SLEPc solve
# ---------------------------------------------------------------------------
def solve_slepc(A, num_modes, tol, *, shift_invert, target, factor_solver,
                ncv, mpd):
    E = SLEPc.EPS().create(comm=COMM)
    E.setOperators(A)
    E.setProblemType(SLEPc.EPS.ProblemType.HEP)
    E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    # nev/ncv/mpd: ncv<=0 lets SLEPc choose (~2*nev). For LARGE nev, mpd
    # (maximum projected dimension) caps the working subspace so the per-restart
    # orthogonalization cost is O(mpd^2 * N) instead of O(ncv^2 * N) -- the key
    # knob for computing many modes without the basis-orthogonalization blowup.
    # It trades more (cheap) restarts for far less work each; mpd<=0 disables it.
    dim_kw = {"nev": num_modes}
    if ncv is not None and ncv > 0:
        dim_kw["ncv"] = ncv
    if mpd is not None and mpd > 0:
        dim_kw["mpd"] = mpd
    E.setDimensions(**dim_kw)
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
            log(f"factor solver {factor_solver!r} unavailable -- using PETSc "
                "built-in LU (serial, may overflow on large fill).")
    else:
        E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)

    # Progress bar driven by SLEPc's per-iteration monitor: tracks how many of
    # the `num_modes` wanted eigenpairs have converged. Rank 0 only; registered
    # on all ranks (the callback is a no-op elsewhere). Defensive -- a monitor
    # error must never abort the solve.
    bar = None
    if RANK == 0:
        try:
            from tqdm import tqdm
            bar = tqdm(total=num_modes, desc="[slepc] modes converged",
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
            # Frontier residual = error of the next mode about to converge.
            # The low Laplacian spectrum is clustered, so nconv sits flat for
            # long stretches and jumps in bursts; this gives live progress
            # (shrinking toward tol) during the flat phases.
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

    # ---- shift-invert: make the (silent, expensive) factorization observable --
    # Shift-invert front-loads ONE big direct factorization of (L - target*I)
    # before the first Krylov iteration -- the long quiet gap that looks like a
    # hang. We force it here (EPSSetUp/PCSetUp triggers MUMPS' symbolic + numeric
    # factorization) and wrap it with three independent signals so it's never
    # dark again:
    #   1. a wall-clock heartbeat thread that ticks "still factorizing, Ns" on
    #      rank 0 (best-effort: only fires if the C call releases the GIL);
    #   2. MUMPS' own ICNTL(4) phase diagnostics (printed from C, GIL-independent,
    #      enabled where the solver is configured above); and
    #   3. a closing line with elapsed time + the factor's REAL RAM/flops from
    #      MUMPS INFOG -- ground-truth sizing, not an estimate.
    if shift_invert:
        log(f"factorizing (shift-invert via {factor_solver}) -- silent phase; "
            "watch for MUMPS ICNTL(4) stats + the heartbeat below. No "
            "convergence bar until this completes ...")
        stop_beat = threading.Event()
        if RANK == 0:
            t_beat = time.perf_counter()

            def _heartbeat():
                # wait() returns True when set -> exits promptly on completion.
                while not stop_beat.wait(30.0):
                    print(f"[slepc]   ...still factorizing, "
                          f"{time.perf_counter()-t_beat:.0f}s elapsed",
                          flush=True)

            threading.Thread(target=_heartbeat, daemon=True).start()

        t_fact = time.perf_counter()
        E.setUp()
        # Idempotent: forces PCSetUp (the factorization) if EPSSetUp deferred it,
        # so the whole cost is guaranteed to land inside this timed bracket.
        E.getST().getKSP().getPC().setUp()
        stop_beat.set()
        dt_fact = time.perf_counter() - t_fact

        stats = ""
        if factor_solver == "mumps":
            try:
                F = E.getST().getKSP().getPC().getFactorMatrix()
                # INFOG(21)=peak RAM (MB) on the heaviest rank; (22)=total across
                # ranks; RINFOG(3)=factorization flops. Real numbers for MEM sizing.
                stats = (f"  factor RAM: {F.getMumpsInfog(21)} MB/rank peak, "
                         f"{F.getMumpsInfog(22)} MB total; "
                         f"{F.getMumpsRinfog(3) / 1e9:.1f} GFLOP")
            except Exception:
                pass
        log(f"factorization complete in {dt_fact:.1f}s{stats}")

    t0 = time.perf_counter()
    E.solve()
    if bar is not None:
        bar.close()
    nconv = E.getConverged()
    kind = "shift-invert" if shift_invert else "Krylov-Schur SR"
    # For shift-invert this is the iteration phase only (factorization timed
    # separately above); for Krylov-Schur SR it's the whole matrix-free solve.
    log(f"{kind}: iters={E.getIterationNumber()} converged={nconv}/{num_modes} "
        f"({time.perf_counter()-t0:.1f}s)")
    return E, nconv


def gather_eigenpairs(E, A, k):
    """Collect k eigenvalues (collective) and gather eigenvectors to rank 0.
    Returns (evals, evecs) on rank 0; (evals, None) elsewhere."""
    evals = np.array([E.getEigenvalue(i).real for i in range(k)])
    xr = A.createVecRight()
    scatter, seq = PETSc.Scatter.toZero(xr)
    evecs = np.empty((A.getSize()[0], k)) if RANK == 0 else None
    for i in range(k):
        E.getEigenpair(i, xr)
        scatter.scatter(xr, seq, addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD)
        if RANK == 0:
            evecs[:, i] = seq.getArray().real
    return evals, evecs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # graph selection (same scheme as the lgp experiments)
    p.add_argument("--template", default="reference")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--threshold", type=float, default=5.0)
    p.add_argument("--knn-k", type=int, default=15)
    p.add_argument("--knn-method", default="faiss")
    p.add_argument("--nlist", type=int, default=1)
    p.add_argument("--bandwidth", type=float, default=1.0)
    p.add_argument("--normalization", choices=["symmetric", "randomwalk"],
                   default="randomwalk")
    p.add_argument("--modes", type=int, default=300)
    # solver
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--ncv", type=int, default=-1, help="<=0 lets SLEPc choose.")
    p.add_argument("--mpd", type=int, default=-1,
                   help="Maximum projected dimension; caps the working subspace "
                        "to bound per-restart orthogonalization at O(mpd^2*N). "
                        "Set it (e.g. ~400-600) when modes is large; <=0 disables.")
    p.add_argument("--shift-invert", action="store_true",
                   help="Shift-invert (target) via a parallel direct solver; "
                        "needs PETSc --with-mumps. Default is Krylov-Schur SR.")
    p.add_argument("--target", type=float, default=0.0,
                   help="Shift-invert target (near the bottom of the spectrum).")
    p.add_argument("--factor-solver", default="mumps",
                   choices=["mumps", "superlu_dist", "petsc"])
    # paths
    p.add_argument("--eigenvector-dir", required=True, type=Path,
                   help="Cache root; graph in <dir>/knn, eigvecs in <dir>/eigvecs.")
    p.add_argument("--build-if-missing", action="store_true")
    p.add_argument("--reference-file", default=None)
    p.add_argument("--annotations-file", default=None)
    p.add_argument("--force-recompute", action="store_true",
                   help="Recompute even if the eigvec cache already exists.")
    args = p.parse_args()

    eig_dir = Path(args.eigenvector_dir)
    knn_dir = eig_dir / "knn"
    eigvec_dir = eig_dir / "eigvecs"

    # Cache keys -- identical scheme to the rest of the pipeline so the output
    # is a drop-in cache hit downstream.
    graph_key = make_graph_key({
        "template": args.template, "stride": args.stride,
        "thresh": args.threshold, "method": args.knn_method,
        "k": args.knn_k, "nlist": args.nlist, "bbox": None})
    eigvec_key = make_eig_key({
        "graph": graph_key, "norm": args.normalization,
        "bw": args.bandwidth, "modes": args.modes})

    log(f"ranks={SIZE}  graph_key={graph_key}")
    log(f"eigvec_key={eigvec_key}")

    # The LaplacianEigensolver helper (for postprocess/save) lives on rank 0
    # only -- it's where the gathered eigenpairs land, and this avoids its
    # constructor printing once per rank. Short-circuit if already computed.
    solver = None
    exists = False
    if RANK == 0:
        solver = LaplacianEigensolver(num_modes=args.modes, backend="scipy",
                                      tol=args.tol, verbose=False)
        solver.backend = "slepc"      # cosmetic; not part of the fingerprint match
        if not args.force_recompute:
            exists = solver.load(eigvec_dir, eigvec_key) is not None
    exists = COMM.tompi4py().bcast(exists, root=0) if hasattr(COMM, "tompi4py") else exists
    if exists:
        log(f"eigvec cache already present for {eigvec_key} -- nothing to do "
            "(use --force-recompute to overwrite).")
        return

    ensure_graph(knn_dir, graph_key, args)
    op, edge_index, edge_value, N = load_operator(
        knn_dir, graph_key, args.bandwidth, args.normalization)
    log(f"N={N:,}  edges={edge_index.shape[1]:,}")

    A = build_petsc_matrix(op, N)
    E, nconv = solve_slepc(
        A, args.modes, args.tol, shift_invert=args.shift_invert,
        target=args.target, factor_solver=args.factor_solver, ncv=args.ncv,
        mpd=args.mpd)
    if nconv == 0:
        raise SystemExit("[slepc] converged 0 eigenpairs -- raise --ncv / --tol.")
    k = min(nconv, args.modes)

    # Require FULL convergence. A partial run is worse than useless: the saved
    # fingerprint records the converged count, so downstream (asking for the
    # full `modes`) sees fingerprint.num_modes < requested and silently
    # recomputes on GPU -- wasting this whole CPU job. So fail loudly and delete
    # any stale/partial cache for this key, leaving nothing behind to masquerade
    # as complete. nconv is collective, so every rank takes this branch together.
    if k < args.modes:
        if RANK == 0:
            for p in LaplacianEigensolver._paths(eigvec_dir, eigvec_key):
                try:
                    if p.exists():
                        p.unlink()
                        log(f"deleted partial/stale cache {p}")
                except OSError as e:
                    log(f"WARNING: could not delete {p}: {e}")
        COMM.barrier()
        raise SystemExit(
            f"[slepc] only {k}/{args.modes} modes converged -- refusing to save "
            f"a partial eigvec cache. Increase --ncv (env NCV) or relax --tol "
            f"and rerun.")

    evals, evecs = gather_eigenpairs(E, A, k)

    if RANK == 0:
        # Postprocess + save in the LaplacianEigensolver layout so downstream
        # reuses it. _postprocess applies eval[0]=0, QR re-orthonormalization,
        # and the randomwalk denormalization (phi = D^-1/2 phi_sym).
        order = np.argsort(evals)
        evals_t = torch.from_numpy(evals[order]).float()
        evecs_t = torch.from_numpy(evecs[:, order]).float()
        # _postprocess truncates to num_modes; guard if fewer converged.
        solver.num_modes = k
        evals_t, evecs_t = solver._postprocess(evals_t, evecs_t, op)
        fp = solver.fingerprint(op, args.bandwidth, args.normalization)
        solver.save(eigvec_dir, eigvec_key, evals_t, evecs_t, fp,
                    extra={"graph_key": graph_key, "ranks": SIZE,
                           "method": "slepc",
                           "spectral_transform":
                               "shift-invert" if args.shift_invert else "krylov-schur",
                           "stride": args.stride, "threshold": args.threshold,
                           "knn_k": args.knn_k, "normalization": args.normalization,
                           "bandwidth": args.bandwidth})
        log(f"saved {k} eigenpairs -> {eigvec_dir}/{eigvec_key}.eigpairs.npz")


if __name__ == "__main__":
    main()
