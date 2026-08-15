#!/usr/bin/env python
"""Graph visualization utilities — the numeric engine behind the explorers.

Runs the border-budget analysis on ANY kernel, manual or trained. This is the
shared back end for `manifold_vs_euclidean_explorer.py`; it is also usable
standalone via the CLI at the bottom of this module.

It answers, for whatever kernel you hand it:

    (1) THE PRIZE      how much is the anatomical border worth AT ALL?  Measured from
                       the empirical lipid covariance, with NO kernel fitted -- so it
                       is an upper bound on what any border-aware kernel can add.
    (2) AT THE BORDER  does this kernel reproduce the covariance drop the data shows
                       across a region border (~19-24%)?  A Euclidean kernel must give
                       0 -- it only sees distance.
    (3) INSIDE         what does it cost inside the regions?  (the manifold kernels
                       decay 2-3x too fast, which is where they lose.)
    (4) PREDICTION     held-out R^2 on the 178-lipid vector, split interior / near-border,
                       with the noise (and optionally the lengthscale) tuned.

Two ways to specify a kernel
----------------------------
    # A. by hand
    spec = KernelSpec(kind="manifold", knn_method="faiss_atlas_weighted",
                      inflation=50, denoise=3, prune=0.97, num_modes=300,
                      bandwidth=0.1, nu=2, lengthscale=1.0)

    # B. from a trained run (reads config.json + the LEARNED lengthscale from model.pth)
    spec = KernelSpec.from_run("/home/casap/mlibra/output/lgp/FOLD-2-MANIFOLD-...")

    # C. the reference baselines
    KernelSpec.euclidean(nu=0.5, lengthscale=2.0)
    KernelSpec.euclidean(nu=0.5, lengthscale=2.0, border_downweight=0.10)

Then
----
    pool = CellPool()                       # pooled MALDI cells + anchors (cached to disk)
    report(pool, [KernelSpec.euclidean(), spec])

NEVER solves an eigensystem. If a config's eigenpairs are not in the cache it tells you
the `manifold/compute_eigenvectors.py` command that would populate them.

CLI
---
    python manifold/visualizations/graph_visualization_utils.py --run <run_dir> [--run <run_dir2>]
    python manifold/visualizations/graph_visualization_utils.py --manual knn_method=faiss_atlas_weighted,prune=0.97,num_modes=300
"""
from __future__ import annotations

import argparse, hashlib, json, logging, os, sys, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.special import kv, gamma as gamma_fn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "maldi")); sys.path.insert(0, str(REPO / "manifold"))
logging.basicConfig(level=logging.ERROR)
log = logging.getLogger("graph_visualization_utils")

from manifold_kernel_builder import build_manifold_graph
from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.compute_eigenvectors import (
    LaplacianEigensolver, make_key as make_eig_key, resolve_modes_key)
from manifold_gp.utils.anatomical_knn import (
    labels_for_nodes_from_sub_atlas, dissolve_root_labels)
from utils import (crop_or_stride_volume, reference_ccf_from_subvolume,
                   coord_norm_from_reference)

# On-disk cache for the pooled cells / lipid fields. Override with
# GRAPH_VIZ_CACHE to keep it out of the source tree.
CACHE = Path(os.environ.get("GRAPH_VIZ_CACHE",
                            REPO / "manifold" / "visualizations" / ".cache"))
CACHE.mkdir(parents=True, exist_ok=True)

DATA = dict(
    reference   = "/home/casap/mlibra/mlibra_data/reference_image.npy",
    annotations = "/home/casap/mlibra/mlibra_data/level_15annot.npy",
    maldi       = "/home/casap/mlibra/mlibra_data/maindata_minimal.parquet",
    eigdir      = "/home/casap/mlibra/output/eigenvectors",
)
SAMPLES = ["ReferenceAtlas", "SecondAtlas", "Male1", "Male2", "Male3",
           "Female1", "Female2", "Female3"]

# Trained runs store S3 paths; map them onto this box.
PATH_REMAP = {
    "/s3/mlibra/mlibra-data/artiom/eigenvectors": DATA["eigdir"],
    "/s3/mlibra/mlibra-data": "/home/casap/mlibra/mlibra_data",
}
def _local(p):
    p = str(p)
    for a, b in PATH_REMAP.items():
        if p.startswith(a):
            return p.replace(a, b, 1)
    return p


def snap_points_to_nodes(coords_mm: np.ndarray, node_ccf: np.ndarray,
                         axis_order=(0, 1, 2), max_mm: float = 1.0):
    """Nearest template node for each MALDI voxel (KD-tree in mm space).

    Returns (node_idx, valid_mask) where valid_mask drops points whose
    nearest node is farther than `max_mm` (outside the tissue mask).
    """
    from scipy.spatial import cKDTree
    n = node_ccf.shape[0]
    pts = coords_mm[:, list(axis_order)].astype(np.float32)
    tree = cKDTree(node_ccf)

    # Pre-filter non-finite query rows: some scipy versions RAISE on NaN, others
    # return the sentinel idx == n. Query only finite rows, and treat the rest
    # (plus any sentinel idx >= n) as unmatched. This keeps every returned index
    # in [0, n), so it can never index an N-sized array out of bounds — the
    # cause of "index N out of bounds for axis 0 with size N".
    finite = np.isfinite(pts).all(axis=1)
    idx = np.zeros(pts.shape[0], dtype=np.int64)
    dist = np.full(pts.shape[0], np.inf, dtype=np.float64)
    if finite.any():
        d, i = tree.query(pts[finite], k=1)
        dist[finite] = np.asarray(d)
        idx[finite] = np.asarray(i)
    oob = (~finite) | ~np.isfinite(dist) | (idx >= n) | (idx < 0)
    idx = np.where(oob, 0, idx).astype(np.int64)
    valid = np.isfinite(dist) & (dist <= max_mm) & ~oob
    if oob.any():
        log.info("snap: %d/%d query points unmatchable (non-finite coords or "
                 "no neighbour) — dropped", int(oob.sum()), oob.size)
    return idx, valid


# ===========================================================================
# 1.  The cell pool -- pooled MALDI voxels, anchors, region labels
# ===========================================================================
class CellPool:
    """MALDI voxels from all wild-type brains, pooled into small grid cells.

    A single voxel from a single brain is far too noisy to correlate; the mean of
    ~100 close voxels is not. `region` / `d_boundary` come from the ANALYSIS atlas
    (default level_15, root-dissolved) and define what counts as "a border" -- the
    same definition for every kernel compared, regardless of which atlas that
    kernel's own graph was built on.
    """
    def __init__(self, stride=4, threshold=5, cell=2, min_voxels=20,
                 n_anchors=3000, seed=0, annotations=None, verbose=True):
        self.stride, self.threshold, self.cell = stride, threshold, cell
        self.annotations = annotations or DATA["annotations"]
        # The pooled lipid field (Z / anchors) is ATLAS-INDEPENDENT, so the atlas is
        # NOT in the cache key -- switching level_5 <-> level_15 only recomputes the
        # cheap region labels (use_atlas), never re-pools the MALDI voxels.
        key = hashlib.md5(json.dumps(
            [stride, threshold, cell, min_voxels, n_anchors, seed],
            sort_keys=True).encode()).hexdigest()[:10]
        f = CACHE / f"pool_{key}.npz"

        ref = np.load(DATA["reference"])
        sv, _sa, off, vs = crop_or_stride_volume(ref, None, stride)
        self.mask = sv > threshold
        self._vs = vs
        self.sub_volume = sv                       # strided template (napari backdrop)
        self.node_idx = np.argwhere(self.mask).astype(np.int32)
        self.node_mm = reference_ccf_from_subvolume(sv, off, vs, threshold).astype(np.float32)
        self.coord_mean, self.coord_std = coord_norm_from_reference(ref)
        self.N = self.node_mm.shape[0]

        if f.exists():
            d = np.load(f, allow_pickle=True)
            self.Z, self.anchors = d["Z"], d["anchors"]
            self.lipids = [str(x) for x in d["lipids"]]
            self.voxels_per_cell = float(d["vpc"])
            if verbose:
                print(f"[pool] HIT {f.name}  {len(self.anchors):,} anchors, "
                      f"{self.Z.shape[1]} lipids, ~{self.voxels_per_cell:.0f} voxels/cell")
        else:
            self._build(f, sv, min_voxels, n_anchors, seed, verbose)

        self.xyz = self.node_mm[self.anchors].astype(np.float64)
        self._pairs()                              # I, J, d, S  (atlas-independent)
        self.use_atlas(self.annotations, verbose=verbose)   # region_full, region, d_boundary, cross

    # ------------------------------------------------- atlas (region labels)
    def use_atlas(self, annotations, verbose=True):
        """(Re)compute the region labels / boundary distances from an atlas volume.

        Cheap — just strides the annotation, dissolves root, and runs the boundary
        EDT. Lets the explorer switch level_5 <-> level_15 without re-pooling."""
        self.annotations = annotations
        ann = np.load(annotations)
        sub_atlas = ann[::self.stride, ::self.stride, ::self.stride]
        lab = dissolve_root_labels(
            labels_for_nodes_from_sub_atlas(self.sub_volume, sub_atlas, self.threshold),
            self.node_mm)
        self.region_full = lab                     # (N,) region of EVERY template node
        self.region = lab[self.anchors]
        self.d_boundary = self._boundary_distance(
            lab, self.mask, self.sub_volume.shape, self._vs)[self.anchors]
        self.cross = self.region[self.I] != self.region[self.J]
        if verbose:
            print(f"[pool] atlas = {Path(annotations).stem}: "
                  f"{len(np.unique(lab)):,} regions, "
                  f"{int((self.d_boundary <= 0.2).mean() * 100)}% anchors near a border")
        return self

    # ---------------------------------------------------------------- build
    def _build(self, f, sv, min_voxels, n_anchors, seed, verbose):
        import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
        _, cell_of_node = np.unique(self.node_idx // self.cell, axis=0, return_inverse=True)
        C = int(cell_of_node.max()) + 1
        schema = pq.read_schema(DATA["maldi"])
        # keep only real lipid channels: drop coordinates, indices, and metadata
        # (x/y/z*, *_index, __index*, Sample/Section/Allen*) -- coordinate columns
        # are trivially predicted by ANY spatial kernel and would inflate R².
        meta = {"xccf", "yccf", "zccf", "x", "y", "z", "Sample", "Section",
                "SectionID", "AllenName", "AllenID"}
        def is_lipid(c):
            cl = c.lower()
            if c in meta or cl.endswith("_index") or cl.startswith("__index"):
                return False
            return pa.types.is_floating(schema.field(c).type)
        lipids = [c for c in schema.names if is_lipid(c)]
        ssum = np.zeros((C, len(lipids))); cnt = np.zeros((C, len(lipids)), np.int32)
        for s in SAMPLES:
            t0 = time.time()
            df = pd.read_parquet(DATA["maldi"], columns=["xccf", "yccf", "zccf", *lipids],
                                 filters=[("Sample", "==", s)])
            ni, ok = snap_points_to_nodes(
                df[["xccf", "yccf", "zccf"]].to_numpy(np.float32), self.node_mm, (0, 1, 2), 1.0)
            v = df[lipids].to_numpy(np.float32)[ok]
            ci = cell_of_node[ni[ok]]; fin = np.isfinite(v)
            np.add.at(ssum, ci, np.where(fin, v, 0.0)); np.add.at(cnt, ci, fin.astype(np.int32))
            if verbose:
                print(f"[pool]   {s:<15} {int(ok.sum()):>9,} voxels ({time.time()-t0:.0f}s)", flush=True)
            del df, v, fin
        good = (cnt >= min_voxels).all(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            Y = np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan)
        cells = np.where(good)[0]
        vpc = float(cnt[cells].mean())

        rep = np.zeros(C, np.int64)
        order = np.argsort(cell_of_node, kind="stable")
        st = np.searchsorted(cell_of_node[order], np.arange(C))
        en = np.searchsorted(cell_of_node[order], np.arange(C), side="right")
        for k in cells:
            m = order[st[k]:en[k]]
            rep[k] = m[np.argmin(((self.node_mm[m] - self.node_mm[m].mean(0)) ** 2).sum(1))]

        rng = np.random.default_rng(seed)
        pick = np.sort(rng.choice(cells, min(n_anchors, len(cells)), replace=False))
        self.anchors = rep[pick]
        self.Z = ((Y[pick] - np.nanmean(Y[cells], 0)) /
                  (np.nanstd(Y[cells], 0) + 1e-8)).astype(np.float32)
        self.lipids, self.voxels_per_cell = lipids, vpc
        np.savez_compressed(f, Z=self.Z, anchors=self.anchors,
                            lipids=np.array(lipids), vpc=vpc)
        if verbose:
            print(f"[pool] built -> {f.name}  {len(cells):,} usable cells, "
                  f"{len(self.anchors):,} anchors, ~{vpc:.0f} voxels/cell")

    @staticmethod
    def _boundary_distance(labels, mask, shape, vs):
        from scipy.ndimage import distance_transform_edt
        lab = np.zeros(shape, np.int64); lab[mask] = labels + 1
        out = np.full(shape, np.nan, np.float32)
        for r in np.unique(labels) + 1:
            tgt = mask & (lab != r)
            d = distance_transform_edt(~tgt, sampling=(vs,) * 3)
            sel = mask & (lab == r); out[sel] = d[sel]
        return out[mask]

    def _pairs(self, max_mm=2.0):
        I, J = np.triu_indices(len(self.anchors), 1)
        d = np.linalg.norm(self.xyz[I] - self.xyz[J], axis=1)
        k = d <= max_mm
        self.I, self.J, self.d = I[k], J[k], d[k]
        Zf = self.Z.astype(np.float64)
        self.S = (Zf[self.I] * Zf[self.J]).mean(1)     # EMPIRICAL covariance of each pair
        self.EDG = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0])
        # self.cross is set by use_atlas (it depends on the region labels)


# ===========================================================================
# 2.  KernelSpec -- manual, or lifted straight out of a trained run
# ===========================================================================
@dataclass
class KernelSpec:
    kind: str = "manifold"                 # euclidean | manifold | heat
    name: str | None = None
    # euclidean / manifold shared
    nu: float = 2.0
    lengthscale: float = 0.1
    outputscale: float = 1.0               # ScaleKernel multiplier on K
    noise: float = 0.0                     # ABSOLUTE observation-noise variance (z-units²);
                                            # 0 = auto-tune (grid-search, current behaviour).
                                            # >0 = use verbatim, no search -- for reproducing a
                                            # specific trained model's learned noise exactly.
    ard: object = None                     # euclidean: optional per-axis lengthscales (z-units)
    border_downweight: float = 0.0         # k *= (1-c) on cross-region pairs
    # manifold: RiemannMaternKernel bump / OOS knobs (used by the REAL kernel)
    bump_scale: float = 3.0
    bump_decay: float = 0.05
    # manifold / heat: the graph
    knn_method: str = "faiss_atlas_weighted"
    inflation: float = 50.0
    root_handling: str = "dissolve"
    denoise: int = 0
    prune: float = 0.0
    knn_k: int = 15
    n_list: str = "sqrt"
    n_probe: int = 8
    stride: int = 4
    threshold: int = 5
    template: str = "reference"
    annotations: str = DATA["annotations"]
    # manifold / heat: the spectrum
    num_modes: int = 1000
    bandwidth: float = 0.1
    laplacian_norm: str = "randomwalk"
    diffusion_time: float = 0.7            # heat only
    eigdir: str = DATA["eigdir"]

    def __post_init__(self):
        if self.name is None:
            if self.kind == "euclidean":
                self.name = f"euclidean (nu={self.nu:g}, l={self.lengthscale:g})"
            else:
                g = self.knn_method.replace("faiss_atlas_weighted", f"atlas x{self.inflation:g}")
                if self.prune: g += f" + prune {self.prune:g}"
                self.name = f"{self.kind} · {g} (K={self.num_modes}, l={self.lengthscale:g})"
            if self.border_downweight:
                self.name += f" + {100*self.border_downweight:.0f}% border"
            if self.noise:
                self.name += f" [noise={self.noise:.3g} FIXED]"

    # ------------------------------------------------------------ builders
    @staticmethod
    def euclidean(nu=0.5, lengthscale=2.0, border_downweight=0.0, outputscale=1.0, noise=0.0):
        return KernelSpec(kind="euclidean", nu=nu, lengthscale=lengthscale,
                          outputscale=outputscale, border_downweight=border_downweight,
                          noise=noise)

    @classmethod
    def from_run(cls, run_dir, lipid=None, use_learned_lengthscale=True, verbose=True):
        """Lift a MANIFOLD spec out of a trained run: config.json for the graph/
        spectrum, and the LEARNED lengthscale/outputscale/noise out of the
        checkpoint. See `_from_run` for the two checkpoint layouts (whole-brain
        vs per-lipid) and the noise-detection caveat."""
        return cls._from_run(run_dir, "manifold", lipid=lipid,
                             use_learned_lengthscale=use_learned_lengthscale, verbose=verbose)

    @classmethod
    def euclidean_from_run(cls, run_dir, lipid=None, use_learned_lengthscale=True, verbose=True):
        """Lift a EUCLIDEAN spec out of a trained run. See `_from_run`."""
        return cls._from_run(run_dir, "euclidean", lipid=lipid,
                             use_learned_lengthscale=use_learned_lengthscale, verbose=verbose)

    @classmethod
    def _from_run(cls, run_dir, kind, lipid=None, use_learned_lengthscale=True, verbose=True):
        """Shared loader behind `from_run`/`euclidean_from_run`.

        Two checkpoint layouts (see manifold/model_hyperparams.py, whose loader +
        constraint-transform logic this reuses):
          * whole-brain  <run>/model.pth               -- one shared kernel;
            lengthscale/outputscale averaged across whatever raw_* tensors are
            found (unchanged behaviour from before `noise` existed).
          * per-lipid    <run>/checkpoints/batch_*.pt   -- REQUIRES `lipid`: each
            lipid has its OWN outputscale/noise (MultitaskGaussianLikelihood
            task noise), so loading without picking one would silently average
            across lipids that don't actually share a kernel.

        `noise` (see KernelSpec.noise) is populated -- verbatim, no auto-tune --
        whenever a recognizable likelihood noise parameter is found. CAVEATS,
        both inherited from model_hyperparams.py and not fixed here:
          * older l3di.lgp / lgp_manifold / GPLFR checkpoints store noise as a
            bare log_var_n / log_sigma tensor (not a gpytorch Likelihood) and are
            NOT auto-detected -- noise silently stays 0 (auto-tune) for those.
          * a per-lipid ARD euclidean lengthscale IS handled (see below) by
            going back to the raw checkpoint tensor -- model_hyperparams.py's
            per-lipid TABLE drops it (only assigns scalars), but the tensor
            itself is right there in the checkpoint.
          * a whole-brain (model.pth) ARD lengthscale -- one vector PER LATENT
            DIM, not per lipid -- is NOT handled; that branch still averages
            everything down to a single scalar (pre-existing behaviour).
        """
        import model_hyperparams as MH        # manifold/ is already on sys.path
        run = Path(run_dir)
        cfg = json.load(open(run / "config.json"))
        ls = float(cfg.get("lengthscale_init") or 1.0)
        os_, noise, ard = 1.0, 0.0, None

        ckpts = MH.discover(run)
        if not ckpts:
            raise FileNotFoundError(f"no model.pth or checkpoints/batch_*.pt under {run}")
        per_lipid = any(c.parent.name == "checkpoints" for c in ckpts)

        if use_learned_lengthscale and per_lipid:
            if lipid is None:
                names = sorted({r.get("lipid") for c in ckpts for r in MH.extract_rows(c)
                                if r.get("lipid")})
                raise ValueError(
                    f"{run.name} is a per-lipid run (checkpoints/batch_*.pt) -- pass "
                    f"lipid=<name> to pick whose hyperparameters to load.\n"
                    f"available lipids ({len(names)}): {names}")
            row = c_match = None
            for c in ckpts:
                hit = next((r for r in MH.extract_rows(c) if r.get("lipid") == lipid), None)
                if hit is not None:
                    row, c_match = hit, c
                    break
            if row is None:
                raise ValueError(f"lipid {lipid!r} not found under {run.name}")

            def pick(suffix, default, warn_missing=True):
                hits = sorted(k for k, v in row.items()
                              if isinstance(v, (int, float)) and k.endswith(suffix))
                if not hits:
                    if verbose and warn_missing:
                        dflt = "n/a" if default is None else f"{default:g}"
                        print(f"[spec]   ! no {suffix!r} column for {lipid!r} "
                              f"-- keeping default {dflt}")
                    return default
                if len(hits) > 1 and verbose:
                    print(f"[spec]   multiple {suffix!r} columns {hits} -- using {hits[0]!r}")
                return float(row[hits[0]])

            os_   = pick("outputscale", os_)
            noise = pick("task_noises", None, warn_missing=False)
            if noise is None:
                noise = pick("noise", 0.0)

            # Lengthscale: a scalar column covers a shared/isotropic kernel
            # (manifold's RiemannMaternKernel, or non-ARD euclidean). An ARD
            # euclidean kernel stores a PER-LIPID VECTOR
            # (n_tasks, 1, ard_dims) that extract_rows' per-lipid table drops
            # entirely (it only assigns when numel()==1 or numel()==n_tasks;
            # an ARD vector matches neither) -- go back to the raw tensor.
            ls_found = pick("lengthscale", None, warn_missing=False)
            if ls_found is not None:
                ls = ls_found
            elif kind == "euclidean":
                sd_c, meta_c = MH.load_merged(c_match)
                names_c = [str(n) for n in (meta_c.get("lipid_names") or [])]
                bounds_c = MH.collect_bounds(sd_c)
                idx = names_c.index(lipid) if lipid in names_c else None
                if idx is not None:
                    for k, v in sd_c.items():
                        if not (k.endswith("raw_lengthscale") and "constraint" not in k
                                and torch.is_tensor(v) and names_c
                                and v.numel() % len(names_c) == 0):
                            continue
                        per_task = v.numel() // len(names_c)
                        if per_task <= 1:
                            continue                    # not ARD -- extract_rows already caught it
                        val = MH.transform(v, bounds_c.get(k))[0].reshape(len(names_c), per_task)
                        ard = val[idx].tolist()
                        if verbose:
                            print(f"[spec]   ARD lengthscale for {lipid!r} ({k}): "
                                  f"{['%.4f' % a for a in ard]}")
                        break
                if ard is None and verbose:
                    print(f"[spec]   ! no lengthscale/ARD column for {lipid!r} "
                          f"-- keeping default {ls:g}")
            elif verbose:
                print(f"[spec]   ! no 'lengthscale' column for {lipid!r} "
                      f"-- keeping default {ls:g}")

            if verbose:
                ls_str = ("ARD" + str(["%.3f" % a for a in ard])) if ard else f"{ls:.3f}"
                print(f"[spec] {run.name[:56]} · lipid={lipid!r}\n"
                      f"       -> l={ls_str}  outputscale={os_:.4f}  "
                      f"noise={noise:.4f} (FIXED, verbatim)")

        elif use_learned_lengthscale:
            sd, _ = MH.load_merged(ckpts[0])
            bounds = MH.collect_bounds(sd)
            raws = [v for k, v in sd.items()
                    if k.endswith("raw_lengthscale") and "constraint" not in k]
            if raws:
                # Positive() constraint == softplus
                vals = [float(torch.nn.functional.softplus(r).flatten()[0]) for r in raws]
                ls = float(np.mean(vals))
                if verbose:
                    print(f"[spec] learned lengthscales {['%.3f' % v for v in vals]}"
                          f"  -> using mean {ls:.3f}  (init was {cfg.get('lengthscale_init')})")
            os_raws = [v for k, v in sd.items()
                       if k.endswith("raw_outputscale") and "constraint" not in k]
            if os_raws:
                ov = [float(torch.nn.functional.softplus(r).flatten().mean()) for r in os_raws]
                os_ = float(np.mean(ov))
                if verbose:
                    print(f"[spec] learned outputscale -> {os_:.3f}")
            noise_raws = [(k, v) for k, v in sd.items()
                          if ("raw_noise" in k or "raw_task_noises" in k) and "constraint" not in k]
            if noise_raws:
                nv = [float(MH.transform(v, bounds.get(k))[0].mean()) for k, v in noise_raws]
                noise = float(np.mean(nv))
                if verbose:
                    print(f"[spec] learned noise -> {noise:.4f} (FIXED, verbatim)")
            elif verbose:
                print("[spec]   ! no gpytorch likelihood noise param found -- older "
                      "log_var_n/log_sigma checkpoints aren't auto-detected; noise stays auto-tune")

        tag = f" · {lipid}" if lipid else ""
        if kind == "euclidean":
            spec = cls(kind="euclidean", name=(run.name[:50] + tag),
                      nu=float(cfg.get("nu", 0.5)), lengthscale=ls, ard=ard,
                      outputscale=os_, noise=noise)
            if verbose:
                ls_str = ("ARD" + str(["%.3f" % a for a in ard])) if ard else f"{spec.lengthscale:.3f}"
                print(f"[spec] {spec.name}\n       -> nu={spec.nu:g} l={ls_str} "
                      f"outputscale={spec.outputscale:.4f} noise={spec.noise:.4f}")
            return spec

        spec = cls(
            kind="manifold",
            name=run.name[:58] + tag,
            nu=float(cfg.get("nu", 2.0)),
            lengthscale=ls,
            outputscale=os_,
            noise=noise,
            bump_scale=float(cfg.get("bump_scale", 3.0)),
            bump_decay=float(cfg.get("bump_decay", 0.05)),
            knn_method=cfg.get("knn_method", "faiss"),
            inflation=float(cfg.get("cross_region_inflation", 50.0)),
            root_handling=cfg.get("root_handling", "dissolve"),
            denoise=int(cfg.get("denoise_labels", 0) or 0),
            prune=float(cfg.get("prune_cross_region", 0.0) or 0.0),
            knn_k=int(cfg.get("knn_k", 15)),
            n_list=str(cfg.get("n_list", "sqrt")),
            stride=int(cfg.get("stride", 4)),
            threshold=int(cfg.get("threshold", 5)),
            template=cfg.get("template_name", "reference"),
            annotations=_local(cfg.get("annotations_file", DATA["annotations"])),
            num_modes=int(cfg.get("num_modes", 1000)),
            bandwidth=float(cfg.get("graphbandwidth_init", 0.1)),
            laplacian_norm=cfg.get("laplacian_norm", "randomwalk"),
            eigdir=_local(cfg.get("eigenvector_dir", DATA["eigdir"])),
        )
        if verbose:
            print(f"[spec] {run.name[:70]}\n       -> {spec.knn_method} infl={spec.inflation:g} "
                  f"prune={spec.prune:g} dn={spec.denoise} K={spec.num_modes} "
                  f"bw={spec.bandwidth:g} nu={spec.nu:g} l={spec.lengthscale:.3f} "
                  f"noise={spec.noise:.4f} atlas={Path(spec.annotations).stem}")
        return spec


# ===========================================================================
# 3.  Eigenpairs at the anchors -- LOAD ONLY, never solve
# ===========================================================================
def _spec_eig_key(spec: KernelSpec) -> str:
    """Config hash identifying this spec's graph+spectrum (ignores kernel-only knobs)."""
    return hashlib.md5(json.dumps(
        {k: v for k, v in asdict(spec).items()
         if k not in ("name", "kind", "nu", "lengthscale", "outputscale", "noise", "ard",
                      "border_downweight", "bump_scale", "bump_decay", "diffusion_time")},
        sort_keys=True).encode()).hexdigest()[:12]


_FULL_EIG = {}   # config_key -> (lam (K,), vec_full (N, K));  kept for at most one config


def full_eigenpairs(spec: KernelSpec, pool: CellPool, verbose=True, allow_solve=False):
    """FULL-node eigenpairs (lam (K,), vec (N, K)) for this spec's graph.

    Load-only by default: raises with the ``compute_eigenvectors.py`` command if
    the config is not in the pipeline cache; ``allow_solve=True`` solves it once on
    the GPU. Kept in RAM for the most recent config only (the full matrix is large),
    so the interactive viz can look up covariance to any template node cheaply.
    """
    key = _spec_eig_key(spec)
    if key in _FULL_EIG:
        return _FULL_EIG[key]

    cfg = SimpleNamespace(template_name=spec.template, reference_file=DATA["reference"],
                          annotations_file=spec.annotations,
                          device="cuda" if torch.cuda.is_available() else "cpu")
    args = dict(eigenvector_dir=spec.eigdir, threshold=spec.threshold, stride=spec.stride,
                knn_k=spec.knn_k, n_list=spec.n_list, n_probe=spec.n_probe,
                knn_method=spec.knn_method, cross_region_inflation=spec.inflation,
                root_handling=spec.root_handling, denoise_labels=spec.denoise,
                prune_cross_region=spec.prune, force_recompute_graph=False)
    knn, ei, ev, gkey = build_manifold_graph(args, cfg, pool.coord_mean, pool.coord_std)
    lap = GraphLaplacianOperator(ev, ei, knn.x.shape[0],
                                 torch.tensor(spec.bandwidth, device=cfg.device),
                                 spec.laplacian_norm)
    kp = {"graph": gkey, "norm": spec.laplacian_norm, "bw": spec.bandwidth,
          "modes": spec.num_modes}
    edir = Path(spec.eigdir) / "eigvecs"
    ekey = resolve_modes_key(edir, make_eig_key(kp), spec.num_modes)
    from manifold_gp.utils.compute_eigenvectors import resolve_ncv_min
    ncv = resolve_ncv_min(spec.num_modes, -1)
    solver = LaplacianEigensolver(num_modes=spec.num_modes, backend="cupy",
                                  ncv_min=ncv, verbose=verbose)
    hit = solver.load(edir, ekey, device="cpu")
    if hit is None:
        if not allow_solve:
            raise FileNotFoundError(
                f"\nNo cached eigenpairs for {spec.name!r}.\n  key: {make_eig_key(kp)}\n\n"
                f"This script never solves. Populate the cache with:\n\n"
                f"  GPU (CUDA/LOBPCG):\n{_eig_cmd(spec)}\n\n"
                f"  CPU/MPI (SLEPc, shift-invert):\n{_slepc_cmd(spec)}\n")
        # An eigensolve takes minutes — ALWAYS show the solver's tqdm in the console,
        # even when the caller asked to stay quiet, so the GUI never looks frozen.
        solver = LaplacianEigensolver(num_modes=spec.num_modes, backend="cupy",
                                      ncv_min=ncv, verbose=True)
        print(f"\n[eig ] SOLVING {spec.name[:48]}\n"
              f"       N={knn.x.shape[0]:,} nodes, K={spec.num_modes} modes, "
              f"bw={spec.bandwidth:g}, {spec.laplacian_norm}\n"
              f"       one-time — result is written to the shared pipeline cache. "
              f"Progress:", flush=True)
        evals, evecs = solver.compute_or_load(
            lap, cache_dir=edir, key=make_eig_key(kp), graphbandwidth=spec.bandwidth,
            laplacian_normalization=spec.laplacian_norm, extra=kp,
            force_recompute=False, device=cfg.device, allow_larger_modes=True)
        lam = evals[:spec.num_modes].detach().cpu().numpy().astype(np.float64)
        vec = evecs[:, :spec.num_modes].detach().cpu().numpy().astype(np.float32)
    else:
        lam, vec, _fp, _m = hit
        lam = lam[:spec.num_modes].numpy().astype(np.float64)
        vec = vec[:, :spec.num_modes].numpy().astype(np.float32)
    _FULL_EIG.clear()                       # bound memory: keep only this config
    _FULL_EIG[key] = (lam, vec)
    if verbose:
        print(f"[eig ] {spec.name[:44]:<46} K={spec.num_modes:<5} "
              f"N={vec.shape[0]:,}  lam in [0, {lam.max():.2f}]")
    return lam, vec


def eig_at_anchors(spec: KernelSpec, pool: CellPool, verbose=True, allow_solve=False):
    """Eigenvectors of this spec's graph, restricted to the pool's anchors.

    Small per-anchor cache on disk; falls back to ``full_eigenpairs`` (load-only by
    default, ``allow_solve=True`` to solve once) and subsamples to the anchors.
    """
    f = CACHE / f"eig_{_spec_eig_key(spec)}_{len(pool.anchors)}.npz"
    if f.exists():
        d = np.load(f)
        return d["lam"].astype(np.float64), d["vec"].astype(np.float64)
    lam, vec_full = full_eigenpairs(spec, pool, verbose=verbose, allow_solve=allow_solve)
    vec = vec_full[pool.anchors].astype(np.float64)
    np.savez_compressed(f, lam=lam, vec=vec)
    return lam, vec


def _eig_cmd(spec: KernelSpec):
    a = ["python manifold/compute_eigenvectors.py",
         f"--reference-volume {DATA['reference']}",
         f"--annotations-volume {spec.annotations}",
         f"--output-path {spec.eigdir}",
         f"--stride {spec.stride} --threshold {spec.threshold}",
         f"--knn-k {spec.knn_k} --modes {spec.num_modes}",
         f"--nlist {spec.n_list} --nprobe {spec.n_probe}",
         f"--bandwidth {spec.bandwidth:g} --normalization {spec.laplacian_norm}",
         f"--knn-method {spec.knn_method}"]
    if spec.knn_method == "faiss_atlas_weighted":
        a.append(f"--cross-region-inflation {spec.inflation:g} "
                 f"--root-handling {spec.root_handling}")
        if spec.denoise: a.append(f"--denoise-labels {spec.denoise}")
        if spec.prune:   a.append(f"--prune-cross-region {spec.prune:g}")
    return " \\\n    ".join(a)


def _slepc_cmd(spec: KernelSpec, nproc=16, target=-0.01):
    """The CPU/MPI alternative to ``_eig_cmd`` -- shift-invert SLEPc, for graphs
    too big (or too slow) for the CUDA/LOBPCG path in compute_eigenvectors.py.
    Same graph key, so it drops into the same eigvec cache.

    Goes through ``slepc_eigensolve.sh`` (not ``slepc_eigensolve.py`` directly):
    the wrapper preloads ``libmkl_rt.so`` under a conda env, which is what a
    bare ``mpirun ... python slepc_eigensolve.py`` needs to avoid
    ``Intel MKL FATAL ERROR: Cannot load libmkl_def.so`` (kernel-lib skew
    between conda's MKL and the one mpirun's ranks pick up), and also resolves
    ``mpirun``/``python`` by absolute path for the same reason."""
    env = {"NPROC": nproc, "SHIFT_INVERT": 1, "TARGET": target,
           "FACTOR_SOLVER": "mumps",
           "TEMPLATE": spec.template,
           "REFERENCE_FILE": DATA["reference"], "ANNOTATIONS_FILE": spec.annotations,
           "EIGENVECTOR_DIR": spec.eigdir,
           "STRIDE": spec.stride, "THRESHOLD": spec.threshold,
           "KNN_K": spec.knn_k, "MODES": spec.num_modes,
           "NLIST": spec.n_list, "NPROBE": spec.n_probe,
           "BANDWIDTH": f"{spec.bandwidth:g}", "NORMALIZATION": spec.laplacian_norm,
           "KNN_METHOD": spec.knn_method}
    if spec.knn_method == "faiss_atlas_weighted":
        env["CROSS_REGION_INFLATION"] = f"{spec.inflation:g}"
        env["ROOT_HANDLING"] = spec.root_handling
        if spec.denoise: env["DENOISE_LABELS"] = spec.denoise
        if spec.prune:   env["PRUNE_CROSS_REGION"] = f"{spec.prune:g}"
    a = [f"{k}={v}" for k, v in env.items()] + ["bash local_run/slepc_eigensolve.sh"]
    return " \\\n    ".join(a)


# ===========================================================================
# 4.  Kernels
# ===========================================================================
def make_kernel(spec: KernelSpec, pool: CellPool, allow_solve=False):
    """Returns K(idx_a, idx_b) over anchor indices, with the border down-weight applied.

    Manifold kernels go through the REAL ``RiemannMaternKernel.features()`` (features
    at the anchors are computed once and reused)."""
    if spec.kind == "euclidean":             # the REAL gpytorch MaternKernel
        anc_mm = pool.node_mm[pool.anchors]
        def base(a, b):
            return euclidean_gram(spec, pool, anc_mm[a], anc_mm[b])
    else:
        F = manifold_features(spec, pool, pool.anchors, allow_solve=allow_solve)  # (A, K)
        def base(a, b):
            return F[a] @ F[b].T

    c = spec.border_downweight
    if not c:
        return base
    def kern(a, b):
        return base(a, b) * (1 - c * (pool.region[a][:, None] != pool.region[b][None, :]))
    return kern


def implied_rho(spec, pool, allow_solve=False):
    """K(i,j)/sqrt(K_ii K_jj) on the pool's pairs -- comparable across kernels
    (outputscale cancels, so this is a pure correlation)."""
    I, J = pool.I, pool.J
    if spec.kind == "euclidean":
        rho = _matern_1d(pool.d, spec.nu, spec.lengthscale)
    else:                                    # real RiemannMaternKernel features
        F = manifold_features(spec, pool, pool.anchors, allow_solve=allow_solve)
        dg = (F * F).sum(1)
        rho = (F[I] * F[J]).sum(1) / np.sqrt(dg[I] * dg[J])
    if spec.border_downweight:
        rho = rho * (1 - spec.border_downweight * pool.cross)
    return rho


def euclidean_gram(spec: KernelSpec, pool: CellPool, A_mm, B_mm):
    """The REAL Euclidean Matérn — gpytorch ``MaternKernel`` (the deployed baseline
    kernel), times ``outputscale``. Isotropic scalar lengthscale by default; if
    ``spec.ard`` (a 3-vector, in z-units) is set it runs per-axis ARD in the
    training coordinate frame. Returns (|A|,|B|)."""
    import torch, gpytorch
    A_mm = np.atleast_2d(np.asarray(A_mm, np.float64))
    B_mm = np.atleast_2d(np.asarray(B_mm, np.float64))
    if spec.ard is not None:
        cm = pool.coord_mean.numpy().astype(np.float64); cs = float(pool.coord_std)
        A = (A_mm - cm) / cs; B = (B_mm - cm) / cs
        k = gpytorch.kernels.MaternKernel(nu=spec.nu, ard_num_dims=3)
        k.lengthscale = torch.as_tensor(np.asarray(spec.ard, np.float32)).reshape(1, 3)
    else:
        A, B = A_mm, B_mm
        k = gpytorch.kernels.MaternKernel(nu=spec.nu)
        k.lengthscale = float(spec.lengthscale)
    with torch.no_grad():
        K = k(torch.as_tensor(A, dtype=torch.float32),
              torch.as_tensor(B, dtype=torch.float32)).to_dense().numpy().astype(np.float64)
    return spec.outputscale * K


def _matern_1d(d, nu, ls):
    """Matérn correlation as a function of distance (unit variance). Equals a
    gpytorch MaternKernel to float precision; used for the 234k-pair correlogram
    (implied_rho) where materialising a coord gram would be wasteful."""
    if nu == 0.5:
        return np.exp(-d / ls)
    if nu == 1.5:
        r = np.sqrt(3) * d / ls; return (1 + r) * np.exp(-r)
    if nu == 2.5:
        r = np.sqrt(5) * d / ls; return (1 + r + r * r / 3) * np.exp(-r)
    z = np.sqrt(2 * nu) * np.maximum(np.asarray(d, float), 1e-12) / ls
    return np.where(np.asarray(d, float) == 0, 1.0,
                    (2 ** (1 - nu) / gamma_fn(nu)) * z ** nu * kv(nu, z))


_REAL_KERNEL = {}   # config_key -> (RiemannMaternKernel, device);  most recent config only


def real_manifold_kernel(spec: KernelSpec, pool: CellPool, allow_solve=False, verbose=True):
    """Construct the ACTUAL ``RiemannMaternKernel`` for this spec's graph+spectrum.

    Same object the training pipeline deploys (bump/Nyström/spectral-density and
    all), built from the cached graph + eigenpairs. Lengthscale is set per call;
    ``outputscale`` and ``bump_*`` are applied by the callers / on the instance.
    Cached for the most recent config (holds the full eigvec on the GPU).
    """
    import torch
    from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
    key = _spec_eig_key(spec)
    if key in _REAL_KERNEL:
        return _REAL_KERNEL[key]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SimpleNamespace(template_name=spec.template, reference_file=DATA["reference"],
                          annotations_file=spec.annotations, device=dev)
    args = dict(eigenvector_dir=spec.eigdir, threshold=spec.threshold, stride=spec.stride,
                knn_k=spec.knn_k, n_list=spec.n_list, n_probe=spec.n_probe,
                knn_method=spec.knn_method, cross_region_inflation=spec.inflation,
                root_handling=spec.root_handling, denoise_labels=spec.denoise,
                prune_cross_region=spec.prune, force_recompute_graph=False)
    knn, ei, ev, _gkey = build_manifold_graph(args, cfg, pool.coord_mean, pool.coord_std)
    lam, vec = full_eigenpairs(spec, pool, verbose=verbose, allow_solve=allow_solve)
    kernel = RiemannMaternKernel(
        nu=int(spec.nu), knn=knn, edge_index=ei, edge_value=ev,
        eigval=torch.as_tensor(lam, dtype=torch.float32, device=dev),
        eigvec=torch.as_tensor(vec, dtype=torch.float32, device=dev),
        nearest_neighbors=spec.knn_k, num_modes=spec.num_modes,
        bump_scale=spec.bump_scale, bump_decay=spec.bump_decay,
        laplacian_normalization=spec.laplacian_norm,
        graphbandwidth_init=spec.bandwidth).to(dev)
    kernel.eval()
    _REAL_KERNEL.clear(); _REAL_KERNEL[key] = (kernel, dev)
    return kernel, dev


def manifold_features(spec: KernelSpec, pool: CellPool, node_idx, allow_solve=False,
                      chunk=40000):
    """The REAL ``RiemannMaternKernel.features(x)`` at the given template nodes.

    Returns ``sqrt(outputscale) * features`` (n, num_modes) so that
    ``F @ F.T`` reproduces the deployed kernel ``outputscale · k(i,j)`` exactly
    (including the spectral-density normalisation and the ×N feature scaling that
    the class applies internally). Nodes are on-graph, so features() takes the
    exact-eigenvector branch (no Nyström).
    """
    import torch
    kernel, dev = real_manifold_kernel(spec, pool, allow_solve=allow_solve)
    kernel.lengthscale = torch.tensor(float(spec.lengthscale))
    idx = np.asarray(node_idx)
    cm = pool.coord_mean.to(torch.float32); cs = pool.coord_std.to(torch.float32)
    zc = (torch.as_tensor(pool.node_mm[idx], dtype=torch.float32) - cm) / cs
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), chunk):
            f = kernel.features(zc[s:s + chunk].to(dev).contiguous())
            out.append(f.detach().cpu().numpy())
    F = np.concatenate(out, 0).astype(np.float64)
    return F * np.sqrt(max(spec.outputscale, 0.0))


def node_covariance(spec: KernelSpec, pool: CellPool, test_node: int,
                    target_nodes: np.ndarray, allow_solve=False):
    """k(test_node, target_nodes) over the FULL template-node set — for the
    interactive slice heatmap (unlike make_kernel, which is anchor-indexed).

    Euclidean uses the node mm-coordinates; manifold uses the full eigenvectors.
    The border down-weight is applied using the analysis region labels. Values are
    NOT normalised (they are covariances k(test, ·)); the caller scales for display.
    """
    tn = int(test_node)
    tgt = np.asarray(target_nodes)
    if spec.kind == "euclidean":             # the REAL gpytorch MaternKernel
        k = euclidean_gram(spec, pool, pool.node_mm[tn], pool.node_mm[tgt])[0]
    else:                                    # the REAL RiemannMaternKernel.features()
        F = manifold_features(spec, pool, np.r_[tn, tgt], allow_solve=allow_solve)
        k = F[1:] @ F[0]
    if spec.border_downweight:
        cross = pool.region_full[tgt] != pool.region_full[tn]
        k = k * (1 - spec.border_downweight * cross)
    return k


def node_gram(spec: KernelSpec, pool: CellPool, A, B, allow_solve=False):
    """K over arbitrary FULL template-node index sets A, B -> (|A|, |B|).

    The node-indexed companion to make_kernel (which is anchor-indexed). Used by
    the explorer to fit/predict a held-out error field at dense slice voxels.
    """
    A = np.asarray(A); B = np.asarray(B)
    if spec.kind == "euclidean":             # the REAL gpytorch MaternKernel
        K = euclidean_gram(spec, pool, pool.node_mm[A], pool.node_mm[B])
    else:                                    # the REAL RiemannMaternKernel.features()
        if np.array_equal(A, B):
            F = manifold_features(spec, pool, A, allow_solve=allow_solve); K = F @ F.T
        else:
            FA = manifold_features(spec, pool, A, allow_solve=allow_solve)
            FB = manifold_features(spec, pool, B, allow_solve=allow_solve)
            K = FA @ FB.T
    if spec.border_downweight:
        cross = pool.region_full[A][:, None] != pool.region_full[B][None, :]
        K = K * (1 - spec.border_downweight * cross)
    return K


def node_lipid_field(pool: CellPool, lipid_name: str, verbose=True):
    """Per-NODE value of one lipid, pooled over all wild-type samples (cached).

    Returns ``(z (N,), covered (N,))`` where z is the lipid aggregated onto every
    template node and z-scored over the covered nodes (NaN where unmeasured).
    Cheap (one parquet column) and cached to ``.cache/nodelip_<hash>.npz``.
    """
    key = hashlib.md5(f"{lipid_name}|{pool.stride}|{pool.threshold}".encode()).hexdigest()[:10]
    f = CACHE / f"nodelip_{key}.npz"
    if f.exists():
        d = np.load(f); return d["z"], d["covered"]
    import pandas as pd
    ssum = np.zeros(pool.N); cnt = np.zeros(pool.N, np.int64)
    for s in SAMPLES:
        df = pd.read_parquet(DATA["maldi"], columns=["xccf", "yccf", "zccf", lipid_name],
                             filters=[("Sample", "==", s)])
        ni, ok = snap_points_to_nodes(
            df[["xccf", "yccf", "zccf"]].to_numpy(np.float32), pool.node_mm, (0, 1, 2), 1.0)
        v = df[lipid_name].to_numpy(np.float32)[ok]; ni = ni[ok]
        fin = np.isfinite(v)
        np.add.at(ssum, ni[fin], v[fin]); np.add.at(cnt, ni[fin], 1)
    covered = cnt > 0
    with np.errstate(invalid="ignore"):
        mean = np.where(covered, ssum / np.maximum(cnt, 1), np.nan)
    mu, sd = np.nanmean(mean), np.nanstd(mean) + 1e-8
    z = (mean - mu) / sd
    np.savez_compressed(f, z=z.astype(np.float32), covered=covered)
    if verbose:
        print(f"[lipid] {lipid_name}: {int(covered.sum()):,}/{pool.N:,} nodes covered")
    return z.astype(np.float32), covered


# ===========================================================================
# 4b.  Per-lipid comparison + a lengthscale health check (used by the explorer)
# ===========================================================================
def lengthscale_health(spec: KernelSpec, pool: CellPool, allow_solve=False):
    """Is the Matérn lengthscale even identifiable for this manifold config?

    S(λ)=(2ν/ℓ²+λ)^(−ν) only 'bites' when its corner λ_c=2ν/ℓ² falls inside the
    retained spectrum [0, λ_max]. If 2ν/ℓ² > λ_max the filter is ~flat, ℓ is
    degenerate with the outputscale, and a 'learned' ℓ never actually moved off
    its init. Returns a dict; ``ok`` is False when the current ℓ is in the dead
    zone. (Euclidean kernels have no spectral corner -> always ok.)
    """
    if spec.kind == "euclidean":
        return dict(ok=True, msg="euclidean: lengthscale always identifiable")
    lam, _ = eig_at_anchors(spec, pool, verbose=False, allow_solve=allow_solve)
    lam_max = float(lam.max())
    lam_c = 2 * spec.nu / spec.lengthscale ** 2
    ls_min = float(np.sqrt(2 * spec.nu / lam_max))
    taper = ((lam_c) / (lam_c + lam_max)) ** (-spec.nu)
    ok = lam_c < lam_max
    if ok:
        msg = (f"OK  λ_c={lam_c:.2f} < λ_max={lam_max:.2f}; S tapers {taper:.1f}× "
               f"across the spectrum")
    else:
        msg = (f"⚠ INERT  λ_c=2ν/ℓ²={lam_c:.2f} > λ_max={lam_max:.2f}: the filter is "
               f"flat ({taper:.2f}×) and ℓ is degenerate with the outputscale. "
               f"Use ℓ ≥ {ls_min:.2f} (or lower the bandwidth / raise num_modes).")
    return dict(ok=ok, lam_max=lam_max, lam_c=lam_c, ls_min=ls_min, taper=taper, msg=msg)


def per_lipid_r2(pool: CellPool, specs, seed=0, tune_alpha=True, allow_solve=False):
    """Held-out per-lipid R² for each spec, on a shared train/val/test split.

    Noise is tuned PER LIPID (its own val-R²-best alpha off the shared grid),
    matching how the real training pipeline (``lgp_experiment_per_lipid.py``,
    ``MultitaskGaussianLikelihood(has_task_noise=True)``) learns one observation
    noise per lipid rather than a single value shared across all of them —
    a manifold kernel doing badly on a couple of noisy lipids no longer forces
    everyone else onto the same (too-high) noise floor.

    Returns ``dict(name -> (n_lipids,) R²)`` plus ``te`` (test indices), ``FAR``,
    ``NEAR`` masks, so callers can build both the summary and the per-lipid
    manifold-vs-euclidean comparison from one fit.
    """
    Z = pool.Z.astype(np.float64)
    rng = np.random.default_rng(seed); perm = rng.permutation(len(pool.anchors))
    n = len(perm); tr, va, te = perm[:n // 2], perm[n // 2:3 * n // 4], perm[3 * n // 4:]
    NEAR, FAR = pool.d_boundary <= 0.20, pool.d_boundary > 0.40
    ALPH = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0] if tune_alpha else [0.1]

    def col_r2(pred, truth):                         # per-column (per-lipid) R²
        return 1 - ((truth - pred) ** 2).sum(0) / ((truth - truth.mean(0)) ** 2).sum(0)

    def solve(Kt, a):
        return np.linalg.solve(Kt + a * (np.trace(Kt) / len(tr)) * np.eye(len(tr)), Z[tr])

    out, preds, alphas = {}, {}, {}
    for s in specs:
        K = make_kernel(s, pool, allow_solve=allow_solve)
        Ktt, Kvt, Ket = K(tr, tr), K(va, tr), K(te, tr)
        n_lip = Z.shape[1]
        if s.noise:                    # FIXED absolute noise -- reproduce a trained model
            sol = np.linalg.solve(Ktt + s.noise * np.eye(len(tr)), Z[tr])
            best_test = Ket @ sol
            best_alpha = np.full(n_lip, s.noise)
        else:
            best_val = np.full(n_lip, -np.inf)
            best_alpha = np.full(n_lip, ALPH[0])
            best_test = np.zeros((len(te), n_lip))
            for a in ALPH:                            # pick noise PER LIPID by val R²
                sol = solve(Ktt, a)
                val_r2 = col_r2(Kvt @ sol, Z[va])
                better = np.nan_to_num(val_r2, nan=-np.inf) > best_val
                best_val[better] = val_r2[better]
                best_alpha[better] = a
                best_test[:, better] = (Ket @ sol)[:, better]
        out[s.name] = col_r2(best_test, Z[te]); preds[s.name] = best_test
        alphas[s.name] = best_alpha                   # (n_lipids,) -- one noise per lipid
    return dict(r2=out, preds=preds, alpha=alphas, tr=tr, va=va, test=te,
                FAR=FAR, NEAR=NEAR, lipids=pool.lipids)


def compare_two(pl, name_a, name_b):
    """Manifold-vs-euclidean per-lipid comparison from a per_lipid_r2 result.

    `name_a` is the challenger (e.g. the manifold kernel), `name_b` the baseline
    (e.g. euclidean). Returns win count, mean/median Δ, and the biggest movers.
    """
    a, b = pl["r2"][name_a], pl["r2"][name_b]
    ok = np.isfinite(a) & np.isfinite(b)
    d = a - b
    order = np.argsort(-d)
    lip = np.array(pl["lipids"])
    return dict(
        n=int(ok.sum()), n_a_wins=int((d[ok] > 0).sum()),
        mean_delta=float(np.nanmean(d[ok])), median_delta=float(np.nanmedian(d[ok])),
        mean_a=float(np.nanmean(a[ok])), mean_b=float(np.nanmean(b[ok])),
        top_wins=[(lip[i], float(d[i])) for i in order[:6]],
        top_losses=[(lip[i], float(d[i])) for i in order[::-1][:6]],
    )


# ===========================================================================
# 5.  The four analyses
# ===========================================================================
def prize(pool: CellPool, n_bins=24, verbose=True):
    """(1) The upper bound, with NO kernel fitted."""
    S, d, X = pool.S, pool.d, pool.cross
    q = np.quantile(d, np.linspace(0, 1, n_bins + 1)); q[-1] += 1e-9
    b = np.clip(np.digitize(d, q) - 1, 0, n_bins - 1)
    C_d = np.zeros(len(S)); C_b = np.zeros(len(S))
    for k in range(n_bins):
        m = b == k
        C_d[m] = S[m].mean()
        for g in (False, True):
            mm = m & (X == g)
            if mm.sum() > 5: C_b[mm] = S[mm].mean()
    SST = ((S - S.mean()) ** 2).sum()
    SSd = ((S - C_d) ** 2).sum(); SSb = ((S - C_b) ** 2).sum()
    out = dict(distance=1 - SSd / SST, both=1 - SSb / SST, border=(SSd - SSb) / SST)
    if verbose:
        print("\n(1) THE PRIZE -- empirical covariance only, NO kernel fitted\n")
        print(f"{'d (mm)':<11}{'n same':>9}{'n cross':>9}{'% cross':>9}"
              f"{'C_same':>9}{'C_cross':>9}{'gap':>8}")
        E = pool.EDG
        for i in range(len(E) - 1):
            m = (d >= E[i]) & (d < E[i + 1])
            ns, nc = int((m & ~X).sum()), int((m & X).sum())
            if ns + nc == 0: continue
            print(f"{f'{E[i]}-{E[i+1]}':<11}{ns:>9,}{nc:>9,}{100*nc/(ns+nc):>8.0f}%"
                  f"{S[m & ~X].mean():>9.3f}{S[m & X].mean():>9.3f}"
                  f"{S[m & ~X].mean() - S[m & X].mean():>8.3f}")
        print(f"\n   Var(S) explained by DISTANCE alone      : {100*out['distance']:>5.2f}%"
              "   <- ceiling for ANY stationary kernel")
        print(f"   Var(S) explained by DISTANCE + border    : {100*out['both']:>5.2f}%")
        print(f"   >>> THE BORDER IS WORTH                  : {100*out['border']:>5.2f}%"
              f"   ({100*out['border']/out['distance']:.0f}% on top of distance)")
    return out


def border_table(pool, specs, verbose=True):
    """(2) Does each kernel reproduce the covariance drop across a border?"""
    E = pool.EDG; rows = {}
    emp = pool.S
    for nm, v in [("EMPIRICAL (the data)", emp)] + [(s.name, implied_rho(s, pool)) for s in specs]:
        r, ref = [], None
        for i in range(len(E) - 1):
            m = (pool.d >= E[i]) & (pool.d < E[i + 1])
            a, c = v[m & ~pool.cross].mean(), v[m & pool.cross].mean()
            if ref is None:
                ref = abs(a)
            # a relative drop is meaningless once the kernel's own correlation has
            # decayed into the noise -- it divides by ~0. Report NaN there instead.
            r.append(100 * (a - c) / a if abs(a) > 0.05 * ref else np.nan)
        rows[nm] = np.array(r)
    if verbose:
        print("\n(2) AT THE BORDER -- relative drop, cross vs same, at MATCHED distance (%)\n")
        print(f"{'':<44}" + "".join(f"{f'{E[i]}-{E[i+1]}':>9}" for i in range(len(E) - 1)))
        for nm, r in rows.items():
            print(f"{nm[:43]:<44}" + "".join(
                f"{x:>9.1f}" if np.isfinite(x) else f"{'--':>9}" for x in r))
        print("\n   a Euclidean kernel MUST give ~0 -- it only sees distance.")
        print("   '--' = the kernel's own correlation has already decayed into the noise,")
        print("          so a relative drop is undefined there (it divides by ~0).")
    return rows


def inside_table(pool, specs, verbose=True):
    """(3) The same-region correlogram -- what the kernel costs you inside a region."""
    E = pool.EDG; rows = {}
    for nm, v in [("EMPIRICAL (the data)", pool.S)] + \
                 [(s.name, implied_rho(s, pool)) for s in specs]:
        r = np.array([v[(pool.d >= E[i]) & (pool.d < E[i + 1]) & ~pool.cross].mean()
                      for i in range(len(E) - 1)])
        rows[nm] = r / r[0]
    if verbose:
        print("\n(3) INSIDE A REGION -- same-region correlogram, normalised to 1.0 at 0.3 mm\n")
        print(f"{'':<44}" + "".join(f"{f'{E[i]}-{E[i+1]}':>9}" for i in range(len(E) - 1)))
        for nm, r in rows.items():
            print(f"{nm[:43]:<44}" + "".join(f"{x:>9.3f}" for x in r))
        print("\n   a kernel that decays FASTER than the data here is losing the long tail,")
        print("   which is where 78% of the short-range pairs live.")
    return rows


def predict_table(pool, specs, tune_lengthscale=False, seed=0, verbose=True):
    """(4) Held-out R^2 on the 178-lipid vector, noise (and optionally l) tuned."""
    Z = pool.Z.astype(np.float64)
    rng = np.random.default_rng(seed); perm = rng.permutation(len(pool.anchors))
    n = len(perm); tr, va, te = perm[:n//2], perm[n//2:3*n//4], perm[3*n//4:]
    NEAR, FAR = pool.d_boundary <= 0.20, pool.d_boundary > 0.40
    ALPH = [1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    LS = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]

    def r2(p, t): return 1 - ((t - p) ** 2).sum() / ((t - t.mean(0)) ** 2).sum()
    def gp(Kt, Kq, a):
        return Kq @ np.linalg.solve(Kt + a * (np.trace(Kt) / len(tr)) * np.eye(len(tr)), Z[tr])

    rows = {}
    for s in specs:
        grid = LS if tune_lengthscale else [s.lengthscale]
        best = (-9, None, None)
        for ls in grid:
            sp = KernelSpec(**{**asdict(s), "lengthscale": ls, "name": s.name})
            K = make_kernel(sp, pool)
            Kt, Kv = K(tr, tr), K(va, tr)
            for a in ALPH:
                v = r2(gp(Kt, Kv, a), Z[va])
                if v > best[0]: best = (v, ls, a)
        _, ls, a = best
        sp = KernelSpec(**{**asdict(s), "lengthscale": ls, "name": s.name})
        K = make_kernel(sp, pool)
        pr = gp(K(tr, tr), K(te, tr), a)
        rows[s.name] = dict(all=r2(pr, Z[te]),
                            interior=r2(pr[FAR[te]], Z[te][FAR[te]]),
                            near=r2(pr[NEAR[te]], Z[te][NEAR[te]]),
                            ls=ls, alpha=a)
    # region-mean baseline
    mu = {r: Z[tr][pool.region[tr] == r].mean(0) for r in np.unique(pool.region[tr])}
    base = np.stack([mu.get(r, Z[tr].mean(0)) for r in pool.region[te]])
    rows["region mean (baseline)"] = dict(
        all=r2(base, Z[te]), interior=r2(base[FAR[te]], Z[te][FAR[te]]),
        near=r2(base[NEAR[te]], Z[te][NEAR[te]]), ls=np.nan, alpha=np.nan)

    if verbose:
        print(f"\n(4) HELD-OUT PREDICTION ({len(te)} test cells, {Z.shape[1]} lipids)\n")
        print(f"{'kernel':<44}{'l':>7}{'noise':>7}{'R2 all':>9}{'R2 inter':>10}{'R2 near':>9}")
        print("-" * 86)
        for nm, r in rows.items():
            print(f"{nm[:43]:<44}{r['ls']:>7.2f}{r['alpha']:>7.2f}"
                  f"{r['all']:>9.3f}{r['interior']:>10.3f}{r['near']:>9.3f}")
    return rows


def report(pool: CellPool, specs, tune_lengthscale=False, plot=True, save=None):
    """Everything, in the notebook's order."""
    print("=" * 88)
    print(f"PLAYROOM   {len(pool.anchors):,} anchor cells "
          f"({pool.cell*0.1:.1f} mm, ~{pool.voxels_per_cell:.0f} voxels each) "
          f"| {len(pool.lipids)} lipids | border = {Path(pool.annotations).stem}")
    print("=" * 88)
    p = prize(pool)
    b = border_table(pool, specs)
    i = inside_table(pool, specs)
    r = predict_table(pool, specs, tune_lengthscale=tune_lengthscale)
    if plot:
        _plot(pool, b, i, r, p, save)
    return dict(prize=p, border=b, inside=i, predict=r)


def _plot(pool, border, inside, predict, pz, save=None):
    import matplotlib as mpl, matplotlib.pyplot as plt
    mpl.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "axes.edgecolor": "#e2e1dd", "axes.grid": True,
                         "grid.color": "#e2e1dd", "grid.linewidth": .6,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9, "legend.frameon": False})
    PAL = ["#0d366b", "#6a7079", "#1baf7a", "#e34948", "#4a3aa7", "#eda100", "#eb6834"]
    ctr = .5 * (pool.EDG[1:] + pool.EDG[:-1])
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.5))
    for k, (nm, v) in enumerate(border.items()):
        emp = nm.startswith("EMPIRICAL")
        ax[0].plot(ctr, v, color=PAL[k % len(PAL)], lw=2.6 if emp else 1.8,
                   ls="-" if emp else "--", marker="o", ms=4, label=nm[:32])
    ax[0].axhline(0, color="#52514e", lw=.8, ls=":")
    ax[0].set_xlabel("distance (mm)"); ax[0].set_ylabel("border drop (%)")
    ax[0].set_title("(a) At the border"); ax[0].legend(fontsize=7)

    for k, (nm, v) in enumerate(inside.items()):
        emp = nm.startswith("EMPIRICAL")
        ax[1].plot(ctr, v, color=PAL[k % len(PAL)], lw=2.6 if emp else 1.8,
                   ls="-" if emp else "--", marker="o", ms=4, label=nm[:32])
    ax[1].set_xlabel("distance (mm)"); ax[1].set_ylabel("correlation (=1 at 0.3 mm)")
    ax[1].set_title("(b) Inside a region"); ax[1].legend(fontsize=7)

    nms = list(predict); xx = np.arange(len(nms)); w = .38
    for j, (lbl, key, c) in enumerate([("interior", "interior", "#1baf7a"),
                                       ("near border", "near", "#e34948")]):
        ax[2].bar(xx + (j - .5) * w, [predict[n][key] for n in nms], w * .9,
                  label=lbl, color=c, edgecolor="white", linewidth=.8)
    ax[2].set_xticks(xx)
    ax[2].set_xticklabels([n[:20] for n in nms], fontsize=6.5, rotation=30, ha="right")
    ax[2].set_ylabel("held-out $R^2$")
    ax[2].set_title(f"(c) Prediction  (border is worth {100*pz['border']:.2f}% of Var(S))")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight"); print(f"\nsaved -> {save}")
    plt.show()


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[],
                    help="a trained run directory (reads config.json + model.pth). Repeatable.")
    ap.add_argument("--manual", action="append", default=[],
                    help="comma-separated k=v, e.g. knn_method=faiss,num_modes=1000,nu=2,lengthscale=1")
    ap.add_argument("--anchors", type=int, default=3000)
    ap.add_argument("--cell", type=int, default=2, help="grid cell size in nodes (2 -> 0.2 mm)")
    ap.add_argument("--tune-lengthscale", action="store_true",
                    help="also sweep the lengthscale (otherwise use the spec's own)")
    ap.add_argument("--no-baseline", action="store_true", help="skip the Euclidean references")
    ap.add_argument("--save", default=str(CACHE / "fig_playroom.png"))
    a = ap.parse_args()

    pool = CellPool(cell=a.cell, n_anchors=a.anchors)
    specs = []
    if not a.no_baseline:
        specs += [KernelSpec.euclidean(nu=0.5, lengthscale=2.0),
                  KernelSpec.euclidean(nu=0.5, lengthscale=2.0, border_downweight=0.10)]
    for r in a.run:
        specs.append(KernelSpec.from_run(r))
    for m in a.manual:
        kw = {}
        for tok in m.split(","):
            k, v = tok.split("=", 1)
            cur = KernelSpec.__dataclass_fields__[k].type
            kw[k] = (float(v) if any(c in v for c in ".eE") and v.replace(".","").replace("-","").isdigit()
                     else int(v) if v.lstrip("-").isdigit() else v)
        specs.append(KernelSpec(**kw))
    if not specs:
        raise SystemExit("nothing to analyse -- pass --run or --manual")
    report(pool, specs, tune_lengthscale=a.tune_lengthscale, save=a.save)


if __name__ == "__main__":
    main()
