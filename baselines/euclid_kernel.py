"""euclid_kernel.py — run EUCLID's `anatomical_interpolation`, unmodified.

Nothing here reimplements the estimator. `load_euclid_code()` lifts three
functions out of the cloned EUCLID checkout and execs them:

    fill_array_interpolation   the structure-gated exp(-d) interpolation kernel
    normalize_to_255           their per-lipid rescale
    anatomical_interpolation   the driver: log -> per-voxel mean -> rescale ->
                               w-clip -> reference<4 zeroing -> interpolate ->
                               10^3 box fill -> save volume

The whole transform is theirs, `anatomical_interpolation` included. We supply
only the two things it cannot get by itself: an AnnData-shaped object for it to
read, and a sampler that reads the resulting volumes at held-out pixels. No copy
or transliteration of their arithmetic exists in this file.

Their kernel is single-threaded and costs ~242 s per lipid on the 132x80x114
grid, so speed comes from `n_jobs`: one process per lipid, each running their
kernel untouched. 25-way takes 173 lipids from 11.6 h to ~30 min.

Fitting on a fold's TRAIN split is just a matter of what goes into that object —
the donor voxels are built from the rows we hand over, and the held-out rows are
never passed in.

Two accommodations, both documented at their call sites:

* **Row reduction.** Their driver runs `data.iterrows()` *inside* the per-lipid
  loop, so raw pixels would cost 173 x 4.97M row visits. We group to their voxel
  key first and pass the geometric mean, so their `np.log` followed by their
  `np.nanmean` recovers the identical per-voxel value from one row instead of
  ~18. Verified exactly: `verify_row_reduction()` gives max|diff| 0.0 over
  1,203,840 voxels against their unreduced path.
* **Output space.** Their volumes are in `normalize_to_255(log(x))` units, not
  the harness's (log - mean)/std. We fit a per-lipid affine on the TRAIN voxels
  to bring them into harness space. Correlation is invariant to that, so the
  headline metric is untouched; it only makes r2/rmse readable.

Geometry: EUCLID works on the 100um Allen CCF grid and indexes it with
`int(ccf*10) - 1`. Both are their conventions and are kept as-is here — the
shipped `reference_image100um.npy` / `annotation_image100um.npy` are used
directly, so no atlas argument is needed.
"""

from __future__ import annotations

import ast
import functools
import logging
import os
import textwrap
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy import ndimage
from tqdm import tqdm

DEFAULT_EUCLID_REPO = Path(__file__).resolve().parents[1] / "euclid"
_SRC = "src/euclid_msi/postprocessing.py"
_WANTED = ("normalize_to_255", "fill_array_interpolation", "anatomical_interpolation")


# ---------------------------------------------------------------------------
# Lift EUCLID's code out of the checkout
# ---------------------------------------------------------------------------

def _extract_source(tree: ast.Module, text: str, name: str) -> str:
    """Source of function `name` (module-level or method), decorators included."""
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            return textwrap.dedent("\n".join(lines[start - 1:node.end_lineno]))
    raise LookupError(f"{name} not found in EUCLID's {_SRC}")


def load_euclid_code(euclid_repo=None) -> dict:
    """Exec EUCLID's interpolation code as-is and return its namespace.

    Source extraction rather than `import euclid_msi.postprocessing` because that
    module pulls in mofapy2 / rdkit / umap / xgboost at import time. The
    namespace we exec into supplies exactly the globals their code references.
    """
    repo = Path(euclid_repo) if euclid_repo else DEFAULT_EUCLID_REPO
    src = repo / _SRC
    if not src.exists():
        raise FileNotFoundError(
            f"EUCLID checkout not found at {src}. Clone it:\n"
            f"    git clone https://github.com/lamanno-epfl/EUCLID.git {repo}\n"
            f"or pass --euclid-repo."
        )
    text = src.read_text()
    tree = ast.parse(text)
    ns = {"np": np, "pd": pd, "os": os, "njit": njit, "tqdm": tqdm,
          "defaultdict": defaultdict, "ndimage": ndimage}
    for name in _WANTED:
        exec(compile(_extract_source(tree, text, name), str(src), "exec"), ns)
    return ns


class _Adata:
    """The slice of AnnData that `anatomical_interpolation` reads: `.X`,
    `.obs` (index + xccf/yccf/zccf) and `.var_names`."""

    def __init__(self, X, obs, var_names):
        self.X = X
        self.obs = obs
        self.var_names = list(var_names)


class _Postproc:
    """The slice of `Postprocessing` that `anatomical_interpolation` reads."""

    def __init__(self, adata, reference_image, annotation_image):
        self.adata = adata
        self.reference_image = reference_image
        self.annotation_image = annotation_image
        self.analysis_name = "euclid"


# ---------------------------------------------------------------------------
# Driving it
# ---------------------------------------------------------------------------

def euclid_voxel(coords_mm: np.ndarray) -> np.ndarray:
    """EUCLID's pixel -> voxel map: `int(ccf * 10) - 1`, from
    `anatomical_interpolation`."""
    return (coords_mm.astype(np.float64) * 10.0).astype(np.int64) - 1


def _reduce_rows(coords_mm, values_raw, shape):
    """One row per EUCLID voxel, carrying the geometric mean.

    Their driver logs, then averages per voxel with `np.nanmean`. Passing
    `exp(nanmean(log(x)))` for a voxel makes their own log+nanmean return the
    same number from a single row, cutting the `iterrows()` cost ~18x with no
    change to the result (zeros survive as `exp(-inf) = 0`, which their `np.log`
    turns straight back into `-inf`).
    """
    vox = euclid_voxel(coords_mm)
    ok = np.ones(len(vox), bool)
    for a in range(3):
        ok &= (vox[:, a] >= 0) & (vox[:, a] < shape[a])
    vox, values_raw = vox[ok], values_raw[ok]
    key = (vox[:, 0] * shape[1] + vox[:, 1]) * shape[2] + vox[:, 2]

    uniq, inv = np.unique(key, return_inverse=True)
    p = values_raw.shape[1]
    reduced = np.empty((uniq.size, p), dtype=np.float32)
    # Chunk over lipids: the log of the full (n, 173) matrix would be ~7 GB.
    for lo in range(0, p, 32):
        hi = min(lo + 32, p)
        with np.errstate(divide="ignore", invalid="ignore"):
            logv = np.log(values_raw[:, lo:hi].astype(np.float32))
        keep = ~np.isnan(logv)               # their nanmean skips NaN, keeps -inf
        sums = np.zeros((uniq.size, hi - lo), dtype=np.float32)
        cnts = np.zeros((uniq.size, hi - lo), dtype=np.float32)
        np.add.at(sums, inv, np.where(keep, logv, 0.0))
        np.add.at(cnts, inv, keep)
        with np.errstate(divide="ignore", invalid="ignore"):
            reduced[:, lo:hi] = np.exp(np.where(cnts > 0, sums / np.maximum(cnts, 1),
                                                np.nan))

    vz = uniq % shape[2]
    vy = (uniq // shape[2]) % shape[1]
    vx = uniq // (shape[2] * shape[1])
    # ccf coords that land back on the same voxel under int(ccf*10)-1
    ccf = (np.stack([vx, vy, vz], 1) + 1.5) / 10.0
    logging.info(f"[euclid] {len(values_raw):,} pixels -> {uniq.size:,} voxel rows")
    return ccf, reduced


def _report_donor_survival(values_raw, w, euclid_repo=None):
    """How many measured voxels survive `normalize_to_255` + the `w` clip.

    Worth a loud line: `w` is a threshold on THEIR 0-255 rescale of the log
    intensities, so it is only meaningful on the intensity scale it was tuned
    for. On this dataset their default w=50 leaves ~7% of donors standing and
    the interpolation ends up reconstructing mostly background, which shows up
    as a collapsed held-out correlation rather than as an error.
    """
    norm255 = load_euclid_code(euclid_repo)["normalize_to_255"]
    kept = []
    with np.errstate(divide="ignore", invalid="ignore"):
        logv = np.log(np.asarray(values_raw, dtype=np.float64))
    for j in range(logv.shape[1]):
        a = norm255(logv[:, j].copy())
        kept.append(float(np.mean(np.isfinite(a) & (a >= w))))
    med = float(np.median(kept))
    msg = (f"[euclid] w={w}: {med * 100:.1f}% of measured voxels survive as donors "
           f"(median over {len(kept)} lipids; range "
           f"{min(kept) * 100:.1f}-{max(kept) * 100:.1f}%)")
    if med < 0.5:
        logging.warning(msg + " — most of the signal is being discarded before "
                              "interpolation; w is calibrated to EUCLID's own "
                              "intensity scale, consider --euclid-w 0")
    else:
        logging.info(msg)
    return med


def _run_one(lipid, coords_mm, values_col, w, out_dir, euclid_repo):
    """One lipid through EUCLID's driver. Module-level and self-contained so it
    can be shipped to a worker process (the exec'd function itself won't pickle,
    so each worker re-lifts it)."""
    # print(), not logging: loky workers inherit the parent's stdout but not its
    # logging config, and on a cluster these lines are how you see a hung worker.
    t0 = time.monotonic()
    print(f"[euclid][worker {os.getpid()}] START  {lipid}", flush=True)
    ns = load_euclid_code(euclid_repo)
    # Their driver wraps the lipid loop in tqdm; with one lipid per worker that is
    # just a progress bar redrawn into the log file. Same iteration, no bar.
    ns["tqdm"] = functools.partial(tqdm, disable=True)
    repo = Path(euclid_repo) if euclid_repo else DEFAULT_EUCLID_REPO
    post = _Postproc(
        _Adata(np.asarray(values_col, dtype=np.float64).reshape(-1, 1),
               pd.DataFrame(coords_mm, columns=["xccf", "yccf", "zccf"]), [lipid]),
        np.load(repo / "reference_image100um.npy"),
        np.load(repo / "annotation_image100um.npy"),
    )
    ns["anatomical_interpolation"](post, [lipid], output_dir=str(out_dir), w=w)
    el = time.monotonic() - t0
    print(f"[euclid][worker {os.getpid()}] DONE   {lipid}  {el:.0f}s", flush=True)
    return lipid, el


def run_anatomical_interpolation(coords_mm, values_raw, lipid_names, out_dir,
                                 w=50, euclid_repo=None, reduce_rows=True,
                                 n_jobs=1):
    """Run EUCLID's `anatomical_interpolation` over `lipid_names`.

    coords_mm : (n, 3) xccf/yccf/zccf in mm — TRAIN rows only.
    values_raw: (n, p) raw intensities (their code applies the log itself).
    Returns {lipid: 3D volume}; volumes are also left on disk in `out_dir`.

    Their kernel is whole-volume and single-lipid: measured at ~242 s per lipid
    on the 132x80x114 grid, so 173 lipids is 11.6 h serially. Their driver loops
    lipids one at a time and the `@njit` holds the GIL, so `n_jobs` fans out over
    *processes*, one lipid each — 25-way takes it to ~30 min. The arithmetic is
    untouched either way.
    """
    repo = Path(euclid_repo) if euclid_repo else DEFAULT_EUCLID_REPO
    shape = np.load(repo / "reference_image100um.npy").shape
    if reduce_rows:
        coords_mm, values_raw = _reduce_rows(coords_mm, values_raw, shape)
    values_raw = np.asarray(values_raw)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"[euclid] anatomical_interpolation: {len(lipid_names)} lipid(s), "
                 f"w={w}, n_jobs={n_jobs} (~242 s/lipid; their kernel is "
                 f"single-threaded, so n_jobs is the only lever)")

    _report_donor_survival(values_raw, w, euclid_repo)

    todo = [(j, lip) for j, lip in enumerate(lipid_names)
            if not (out_dir / f"{lip}_interpolation_log.npy").exists()]
    if len(todo) < len(lipid_names):
        logging.info(f"[euclid] {len(lipid_names) - len(todo)} volume(s) already on "
                     f"disk in {out_dir}; recomputing {len(todo)}")
    if todo:
        t0 = time.monotonic()
        n = len(todo)

        secs_seen = []

        def _progress(k, lip, lipid_secs):
            """One line per finished lipid: count, wall clock, ETA.

            The ETA has to model the concurrency, not extrapolate throughput.
            Lipids run `n_jobs` at a time and each costs about the same, so the
            work left is ceil(remaining / n_jobs) more waves. Naive
            elapsed/k*(n-k) reports ~25x too long until the first wave lands,
            which is exactly when you are watching the log.
            """
            secs_seen.append(lipid_secs)
            el = time.monotonic() - t0
            per = sum(secs_seen) / len(secs_seen)
            # Estimate the TOTAL and subtract elapsed, rather than extrapolating
            # from what is left. Lipids run n_jobs at a time and each costs about
            # the same, so the job is ceil(n / n_jobs) waves of `per` seconds.
            # Counting remaining work instead gets the tail wrong either way:
            # elapsed/k*(n-k) over-reports ~n_jobs-fold before the first wave
            # lands, and counting queued-only under-reports by a whole wave.
            total = -(-n // max(n_jobs, 1)) * per          # ceil division
            eta = max(0.0, total - el)
            queued = max(0, n - k - n_jobs)
            logging.info(
                f"[euclid] {k:>4}/{n} lipids | {lip} took {lipid_secs:.0f}s "
                f"(mean {per:.0f}s) | elapsed {el / 60:.1f} min | "
                f"eta {eta / 60:.1f} min ({queued} queued, "
                f"{min(n_jobs, n - k)} in flight)"
            )

        if n_jobs == 1:
            for k, (j, lip) in enumerate(todo, 1):
                _, secs = _run_one(lip, coords_mm, values_raw[:, j], w, out_dir,
                                   euclid_repo)
                _progress(k, lip, secs)
        else:
            from joblib import Parallel, delayed
            # return_as="generator_unordered" hands each result back the moment it
            # lands, so the log shows progress live instead of one dump at the end.
            gen = Parallel(n_jobs=n_jobs, backend="loky",
                           return_as="generator_unordered")(
                delayed(_run_one)(lip, coords_mm, values_raw[:, j], w, out_dir,
                                  euclid_repo)
                for j, lip in todo
            )
            for k, (lip, secs) in enumerate(gen, 1):
                _progress(k, lip, secs)
        logging.info(f"[euclid] interpolation finished: {n} lipids in "
                     f"{(time.monotonic() - t0) / 60:.1f} min on {n_jobs} process(es)")

    volumes, missing = {}, []
    for lip in lipid_names:
        f = out_dir / f"{lip}_interpolation_log.npy"
        if f.exists():
            volumes[lip] = np.load(f)
        else:
            missing.append(lip)
    if missing:
        raise RuntimeError(
            f"anatomical_interpolation produced no volume for {len(missing)} "
            f"lipid(s) (it catches and prints its own errors): {missing[:5]}"
        )
    return volumes


def sample_volumes(volumes, lipid_names, coords_mm):
    """Read the interpolated volumes at `coords_mm`, using EUCLID's index map.

    Out-of-grid pixels and voxels the pipeline left as NaN come back as NaN.
    """
    any_vol = volumes[lipid_names[0]]
    vox = euclid_voxel(coords_mm)
    ok = np.ones(len(vox), bool)
    for a in range(3):
        ok &= (vox[:, a] >= 0) & (vox[:, a] < any_vol.shape[a])
    out = np.full((len(vox), len(lipid_names)), np.nan, dtype=np.float32)
    x, y, z = vox[ok, 0], vox[ok, 1], vox[ok, 2]
    for j, lip in enumerate(lipid_names):
        out[ok, j] = volumes[lip][x, y, z]
    n_nan = int(np.isnan(out).any(axis=1).sum())
    if n_nan:
        logging.info(f"[euclid] {n_nan:,}/{len(vox):,} sampled pixels are NaN "
                     f"(outside the grid or never filled)")
    return out


def verify_row_reduction(coords_mm, values_raw, lipid, out_dir, w=50,
                         euclid_repo=None, n_rows=200_000, seed=0):
    """Run one lipid with and without the row reduction and compare volumes.

    Uses a subsample so the unreduced path (their `iterrows()`) is affordable.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(coords_mm), min(n_rows, len(coords_mm)), replace=False)
    c, v = coords_mm[idx], values_raw[idx]
    a = run_anatomical_interpolation(c, v, [lipid], Path(out_dir) / "reduced",
                                     w=w, euclid_repo=euclid_repo, reduce_rows=True)[lipid]
    b = run_anatomical_interpolation(c, v, [lipid], Path(out_dir) / "raw",
                                     w=w, euclid_repo=euclid_repo, reduce_rows=False)[lipid]
    both = np.isfinite(a) & np.isfinite(b)
    rep = {"n_compared": int(both.sum()),
           "max_abs_diff": float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0,
           "nan_mismatch": int((np.isfinite(a) != np.isfinite(b)).sum())}
    logging.info(f"[euclid] row-reduction parity: {rep}")
    return rep
