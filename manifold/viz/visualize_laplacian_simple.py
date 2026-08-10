#!/usr/bin/env python
"""Reference vs graph Laplacian vs its eigenvector reconstruction.

The question this tool answers: **how much of what the graph Laplacian
actually does to a field is captured by the first N eigenmodes?** — which is
exactly the truncation every manifold-GP kernel here lives with.

The field `f` is the reference template intensity sampled at the graph nodes
(optionally σ-smoothed first, to keep the answer about anatomy instead of
voxel speckle). Four things are then rendered on the same grid:

  f            reference intensity at the nodes (the input field)
  L·f          the exact graph Laplacian applied to f — sparse matvec, no
               eigenvectors involved. This is ground truth.
  L_N·f        the same thing rebuilt from the first N eigenpairs only:
                   L_N·f = Σ_{k<N} λ_k c_k φ_k
               (see `spectral_coeffs` for what c_k is — the projection is
               D-weighted, which matters for randomwalk normalization).
  L·f − L_N·f  the residual: everything the truncated basis misses. Where
               this lights up is where N modes are not enough — typically
               thin boundaries and high-curvature interfaces.
  f_N          the N-mode reconstruction of f itself, for reference.

Controls (right dock, all applied on the ⟳ Re-render button — nothing
recomputes until you press it):

  graph        which graph the Laplacian is built on:
                 · faiss                — plain Euclidean kNN, anatomy-blind
                 · faiss_atlas_weighted — same topology, cross-region edges
                                          inflated ×--cross-region-inflation
                                          (the "anatomically weighted" graph)
               Switching rebuilds/loads that graph's eigenbasis (cached on
               disk under --eigenvector-dir; a cold miss means a full
               eigensolve, so expect a wait the first time).
  modes N      how many eigenpairs enter L_N·f. Truncating a basis that is
               already loaded is a matvec, so this is instant.
  σ            Gaussian smoothing of the density before L is applied.
  γ / pct      colormap shaping: signed γ around zero (γ<1 exposes the
               interior of a mode) and the percentile of |signal| that
               saturates the colormap.

Left dock: the eigenvalue spectrum λ_k vs k (linear + log-log with a k^(2/3)
Weyl reference) of the *currently loaded* graph, with the truncation N shaded
— so the mode dropdown and the spectrum are read together.

The info panel reports four numbers (see `recon_metrics` for the why): res_L,
the operator residual ‖L·f − L_N·f‖/‖L·f‖; res_f, the field residual
‖f − f_N‖_D/‖f‖_D; cos, the alignment of L_N·f with L·f; and amp, the
amplitude fraction. Expect res_L ≈ 1 on an unsmoothed template at any N a
solver can reach — that is a real property of this operator (the unresolved
eigenvalues reach O(1/bandwidth²)), not a broken reconstruction; res_f and cos
are the discriminating numbers. Raising σ moves the field into the resolved
band and pulls res_L down.

Cache policy: this is a viewer, so it never solves eigenvectors behind your
back. Each graph's eigvec cache key is resolved *before* any solve; if no cache
holds >= --num-modes for that exact graph, the mode count is downgraded to the
largest one that is cached (and the modes dropdown re-caps to match), and if
nothing at all is cached the run aborts with a list of the keys that do exist —
bandwidth, nlist, inflation, root-handling, prune and mode count all enter the
key, so that list is usually enough to spot which one you mistyped. A stale
cache (right key, different edge set) is an error too, not a silent multi-GB
recompute. Pass --allow-eigensolve to opt into solving; --cache-report prints
what is loadable for every graph in the dropdown and exits.

Headless: pass --no-launch to skip napari and instead print all four metrics
as a function of N for every graph in the dropdown (a mode-sufficiency table).

Seeing the stride: the reference is added twice (--template-res both) — at its
native 25 µm and as the graph actually samples it, one value per node drawn as
a stride³ block. Both live in the same coordinate frame (the node lattice via a
napari `scale` of (stride,stride,stride), which also avoids materializing a
77 M-voxel array per layer), so toggling between them is exactly what striding
discards, and every signal layer inherits that same blocky lattice because it
is drawn nearest-neighbour. Start in 2-D (--ndisplay 2) to judge it; 3-D volume
rendering blurs the lattice away. Startup also quantifies the cost: correlation
and rel-L2 of the block-sampled reference against the true one, plus the share
of full-res tissue voxels that get no node at all. That last number is a floor
no mode count can beat — detail below the lattice was never in the operator.

A display-only node sample makes the retained spatial density explicit. For a
representative random pair, a yellow straight Euclidean segment is overlaid
with the pink geometric shortest path through kNN edges; summary statistics
over separated random pairs quantify how closely graph distance tracks
Euclidean distance. This uses coordinate-derived edge lengths, so anatomical
distance inflation still affects the Laplacian but not the sampling check.
"""
from __future__ import annotations

# --- repo path bootstrap (this file moved out of maldi/) ---
import sys as _sys
from pathlib import Path as _Path
_REPO_ = _Path(__file__).resolve().parents[2]
for _p in (str(_REPO_), str(_REPO_ / "maldi"), str(_REPO_ / "manifold"),):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib.cm as cm

from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.compute_eigenvectors import (
    LaplacianEigensolver, resolve_ncv_min, make_key as make_eig_key,
)
from manifold_gp.utils.nearest_neighbors import (
    KnnGraphCache, make_key as make_graph_key, resolve_nlist, resolve_nprobe,
)
from manifold_gp.utils.anatomical_knn import (
    labels_for_nodes_from_sub_atlas, inflate_cross_region_edges,
    labels_for_nodes_from_template_clustering, dissolve_root_labels,
    denoise_labels_majority_vote, prune_cross_region_edges,
)
from utils import (
    crop_or_stride_volume, reference_ccf_from_subvolume, coord_norm_from_reference,
)


log = logging.getLogger("visualize_laplacian_simple")

# Dropdown label -> --knn-method value. Labels are what shows in the UI.
GRAPH_LABELS = {
    "faiss": "faiss  (euclidean kNN)",
    "faiss_atlas_weighted": "atlas weighted  (anatomical prior)",
    "anatomical_atlas": "anatomical atlas  (per-region kNN)",
    "faiss_cluster_weighted": "cluster weighted  (data-driven parcels)",
}


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ---- template / nodes --------------------------------------------------
    p.add_argument("--template-name", required=True)
    p.add_argument("--reference-file", required=True)
    p.add_argument("--annotations-file", default=None,
                   help="Atlas labels (.npy). Required for the atlas-weighted "
                        "and anatomical graphs.")
    p.add_argument("--eigenvector-dir", required=True,
                   help="Cache dir holding knn/ and eigvecs/ subfolders.")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--threshold", type=int, default=5)

    # ---- graphs offered in the dropdown -----------------------------------
    p.add_argument("--graph-methods", nargs="+", default=["faiss",
                                                          "faiss_atlas_weighted"],
                   choices=list(GRAPH_LABELS),
                   help="Graphs to offer in the dropdown. Each one gets its own "
                        "cache key; the first is loaded at startup and the rest "
                        "lazily on Re-render.")
    p.add_argument("--knn-k", type=int, default=15)
    p.add_argument("--n-list", default="sqrt",
                   help="FAISS IVF nlist: an int, or 'sqrt' (default) for "
                        "round(sqrt(N)) — matches the training pipeline.")
    p.add_argument("--n-probe", dest="n_probe", default="8",
                   help="FAISS IVF nprobe. MUST be > 1 when nlist > 1, else the "
                        "graph fragments into ~nlist components.")
    p.add_argument("--cross-region-inflation", dest="cross_region_inflation",
                   type=float, default=50.0,
                   help="Cross-region squared-distance inflation for the "
                        "weighted graphs (the strength of the anatomical prior).")
    p.add_argument("--root-handling", dest="root_handling",
                   choices=["dissolve", "ignore", "cross"], default="dissolve",
                   help="How to treat the atlas label-0 'root' catch-all. "
                        "'dissolve' reassigns each root node to its nearest real "
                        "region before inflating (default); 'cross' is the legacy "
                        "inflate-everything-root-touching behaviour.")
    p.add_argument("--cluster-k", dest="cluster_k", type=int, default=64)
    p.add_argument("--cluster-spatial-weight", dest="cluster_spatial_weight",
                   type=float, default=1.0)
    p.add_argument("--cluster-seed", dest="cluster_seed", type=int, default=0)
    p.add_argument("--cluster-fit-subsample", dest="cluster_fit_subsample",
                   type=int, default=40000)
    p.add_argument("--denoise-labels", dest="denoise_labels", type=int, default=0,
                   help="Majority-vote label smoothing passes (weighted graphs). "
                        "Only affects the graph through --prune-cross-region.")
    p.add_argument("--prune-cross-region", dest="prune_cross_region", type=float,
                   default=0.0,
                   help="Fraction of cross-region edges to hard-remove. Changes "
                        "edges → distinct eigvec cache key.")

    # ---- Laplacian / spectrum ---------------------------------------------
    p.add_argument("--laplacian-norm", choices=["symmetric", "randomwalk"],
                   default="randomwalk")
    p.add_argument("--graphbandwidth", type=float, required=True)
    p.add_argument("--num-modes", type=int, default=1000,
                   help="Eigenpairs loaded per graph = the maximum N in the mode "
                        "dropdown. Memory is N_nodes × num_modes floats.")
    p.add_argument("--ncv-min", dest="ncv_min", type=int, default=-1)
    p.add_argument("--mode-ladder", type=int, nargs="+", default=None,
                   help="Explicit mode counts for the dropdown. Default: a "
                        "1/2/5-decade ladder capped at --num-modes.")
    p.add_argument("--initial-modes", type=int, default=None,
                   help="Mode count selected at startup (default: --num-modes).")
    p.add_argument("--force-recompute-graph", action="store_true")
    p.add_argument("--force-recompute-eigvecs", action="store_true")
    p.add_argument("--allow-eigensolve", action="store_true",
                   help="Permit a real eigensolve when no usable cache exists. "
                        "OFF by default: this is a viewer, and an uncached "
                        "combination costs minutes of solve plus a ~2 GB cache "
                        "write per graph (more at stride<4). By default a "
                        "cache miss instead downgrades --num-modes to the "
                        "largest cached count for that exact graph, or aborts "
                        "with a list of what IS cached.")
    p.add_argument("--cache-report", action="store_true",
                   help="Print, for every graph in --graph-methods, which "
                        "eigenvector caches exist (mode counts) and exit. "
                        "Builds the kNN graphs (cheap/cached) but never solves.")
    p.add_argument("--keep-graphs", action="store_true",
                   help="Keep every visited graph's eigenbasis resident, so "
                        "switching back is instant. Default frees the previous "
                        "one (each basis is N × num_modes floats — GBs at "
                        "stride 4).")

    # ---- field + display ---------------------------------------------------
    p.add_argument("--density-smooth-sigma", type=float, default=0.0,
                   help="Initial Gaussian σ (in strided voxels) applied to the "
                        "density before L. 0 disables.")
    p.add_argument("--contrast-pct", type=float, default=99.5,
                   help="Percentile of |signal| that saturates the diverging "
                        "colormaps. 100 = true [-amax, amax].")
    p.add_argument("--gamma", type=float, default=0.6,
                   help="Sign-preserving γ around zero; γ<1 boosts small "
                        "magnitudes so field interiors stay visible.")
    p.add_argument("--template-opacity", type=float, default=0.08)
    p.add_argument("--graph-node-display-sample", type=int, default=50_000,
                   help="Maximum graph nodes drawn as points. Display-only "
                        "subsampling; distance calculations use the full graph.")
    p.add_argument("--distance-pairs", type=int, default=16,
                   help="Random node pairs used to compare geometric graph "
                        "shortest-path distance with straight-line Euclidean "
                        "distance. 0 disables the distance diagnostic.")
    p.add_argument("--anatomical-edge-display-sample", type=int, default=4_000,
                   help="Maximum same-region and cross-region edges drawn in "
                        "the anatomical-prior overlay (per class).")
    p.add_argument("--template-res", choices=["full", "strided", "both"],
                   default="both",
                   help="Which reference layers to add. 'full' = native 25 µm; "
                        "'strided' = the reference as the graph samples it "
                        "(one value per node, drawn as stride³ blocks); 'both' "
                        "(default) adds each as its own layer in the same "
                        "coordinate frame, so toggling between them shows "
                        "exactly what striding discards. Full res costs ~77 M "
                        "voxels; the strided one is 64× smaller at stride 4.")
    p.add_argument("--template-rendering",
                   choices=["iso", "attenuated_mip", "mip", "translucent"],
                   default="iso",
                   help="3-D rendering for the reference layers (ignored in "
                        "2-D). Default 'iso': a surface at "
                        "--template-iso-threshold, so the reference reads as a "
                        "shell you can see the signal inside. The MIP modes "
                        "return the max along each ray, and a solid brain "
                        "saturates nearly every ray that crosses it — the "
                        "result is a white fog filling the layer's bounding "
                        "box, which is what makes the volume look like a "
                        "glowing cube.")
    p.add_argument("--template-attenuation", type=float, default=0.5,
                   help="Attenuation for --template-rendering attenuated_mip. "
                        "Higher = deeper samples contribute less.")
    p.add_argument("--template-iso-threshold", type=float, default=None,
                   help="Iso value for --template-rendering iso. Default: the "
                        "node --threshold, so the surface is exactly the tissue "
                        "boundary the graph was built on.")
    p.add_argument("--ndisplay", type=int, choices=[2, 3], default=3,
                   help="Start in 2-D slice view or 3-D volume view. The stride "
                        "lattice is far easier to judge in 2-D (blocks are "
                        "literal squares); 3-D volume rendering blurs it.")
    p.add_argument("--stride-report-sample", type=int, default=2_000_000,
                   help="Full-res tissue voxels sampled to quantify the stride "
                        "cost at startup. 0 skips the measurement.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--no-launch", action="store_true",
                   help="Skip napari; print the residual-vs-N table for every "
                        "graph in the dropdown and exit.")
    p.add_argument("-v", "--verbose", action="store_true")
    return vars(p.parse_args())


# =============================================================================
# Eigenvector cache lookup
#
# The eigvec key encodes the whole graph signature plus the mode count, as
# `..._modes=<int>_norm=...` (make_key sorts fields alphabetically, so `graph`
# precedes `modes` and `norm` follows it). Swapping just that field gives the
# set of sibling caches for the SAME graph at other mode counts. Eigenpairs are
# nested, so a cache with M >= N modes serves N exactly (the solver's
# allow_larger_modes does that trim); a cache with M < N can still serve M
# modes, which is the sensible fallback for a viewer.
# =============================================================================
_MODES_FIELD_RE = re.compile(r"_modes=(\d+)_norm=")


def _cache_keys(cache_dir: Path):
    """Keys in `cache_dir` that have BOTH the npz and its meta sidecar.

    A lone npz (or lone sidecar) is not loadable — `LaplacianEigensolver.load`
    returns None and the caller silently re-solves — so half-written entries
    must not count as cached.
    """
    keys = []
    for npz in sorted(Path(cache_dir).glob("*.eigpairs.npz")):
        key = npz.name[: -len(".eigpairs.npz")]
        if LaplacianEigensolver._paths(cache_dir, key)[1].exists():
            keys.append(key)
    return keys


def cached_mode_counts(cache_dir: Path, key: str) -> list:
    """Mode counts cached for the graph signature of `key`, ascending."""
    if _MODES_FIELD_RE.search(key) is None:
        return []
    neutral = _MODES_FIELD_RE.sub("_modes=*_norm=", key)
    counts = []
    for cand in _cache_keys(cache_dir):
        m = _MODES_FIELD_RE.search(cand)
        if m is not None and _MODES_FIELD_RE.sub("_modes=*_norm=", cand) == neutral:
            counts.append(int(m.group(1)))
    return sorted(set(counts))


def _keys_for_method(cache_dir: Path, method: str) -> list:
    """Cached keys whose knn method is exactly `method`.

    A plain `f"method={method}_" in key` test is wrong: method values nest
    ("faiss" is a prefix of "faiss_atlas_weighted"), so it reports every
    weighted key as a faiss key. make_key emits `field=value` joined by "_",
    and field names are lowercase alphabetic, so requiring the method value to
    be followed by the next `field=` disambiguates.
    """
    pat = re.compile(rf"_method={re.escape(method)}_(?=[a-z]+=)")
    return [k for k in _cache_keys(cache_dir) if pat.search(k)]


def _sibling_report(cache_dir: Path, method: str, wanted: str = "",
                    limit: int = 10) -> str:
    """Cached keys for this knn method, nearest-miss first — so a bw / nlist /
    inflation / prune mismatch against what is on disk is visible, not guessed."""
    hits = _keys_for_method(cache_dir, method)
    if not hits:
        return f"  (no cached eigenvectors at all for method={method})"
    if wanted:
        from difflib import SequenceMatcher
        hits.sort(key=lambda k: SequenceMatcher(None, k, wanted).ratio(),
                  reverse=True)
    shown = hits[:limit]
    tail = ("" if len(hits) <= limit
            else f"\n  … and {len(hits) - limit} more")
    head = ("  (closest first)\n" if wanted and len(hits) > 1 else "")
    return head + "\n".join(f"  {k}" for k in shown) + tail


def resolve_cached_modes(cache_dir: Path, key: str, requested: int,
                         method: str, allow_eigensolve: bool) -> int:
    """Mode count to actually ask the solver for.

    Returns `requested` when a usable cache exists (exact or larger), or when
    solving is explicitly allowed. Otherwise downgrades to the largest cached
    count for this graph, and raises if nothing is cached at all.
    """
    counts = cached_mode_counts(cache_dir, key)
    if any(c >= requested for c in counts):
        return requested
    if allow_eigensolve:
        if counts:
            log.warning(f"[{method}] no cache with >= {requested} modes "
                        f"(have {counts}); --allow-eigensolve is set, so this "
                        f"will run a full eigensolve and write a new cache.")
        else:
            log.warning(f"[{method}] nothing cached for this graph; "
                        f"--allow-eigensolve is set, so this will run a full "
                        f"eigensolve and write a new cache.")
        return requested
    if counts:
        best = max(counts)
        log.warning(f"[{method}] no eigvec cache with >= {requested} modes; "
                    f"using the largest cached count instead: {best} modes. "
                    f"(cached: {counts}; pass --allow-eigensolve to solve for "
                    f"{requested}.)")
        return best
    raise RuntimeError(
        f"No cached eigenvectors for the requested graph, and --allow-eigensolve "
        f"is not set.\n\nwanted key:\n  {key}\n\ncached keys for "
        f"method={method}:\n{_sibling_report(cache_dir, method, key)}\n\n"
        f"Either match one of those (bandwidth, nlist, inflation, "
        f"root-handling, prune and mode count all enter the key) or pass "
        f"--allow-eigensolve to compute this one (minutes + a multi-GB cache "
        f"write).")


# =============================================================================
# Common setup — template, nodes, coord normalization. Shared by all graphs.
# =============================================================================
def setup_common(args: dict) -> dict:
    device = torch.device(args["device"])
    template_full = np.load(args["reference_file"])
    annotations_full = (np.load(args["annotations_file"])
                        if args["annotations_file"] else None)

    sub_volume, sub_atlas, voxel_offset, voxel_scale_mm = crop_or_stride_volume(
        template_full, annotations_full, stride=args["stride"],
    )
    reference_ccf = reference_ccf_from_subvolume(
        sub_volume, voxel_offset, voxel_scale_mm, args["threshold"],
    )
    reference_nodes_mm = torch.tensor(np.asarray(reference_ccf, np.float32))
    # Isotropic whole-brain normalization — the same source of truth training and
    # SLEPc use, so graph/eigvec cache keys built here match the ones on disk.
    coord_mean, coord_std = coord_norm_from_reference(template_full)
    reference_nodes = ((reference_nodes_mm - coord_mean) / coord_std).to(device)

    node_voxel_idx = np.argwhere(sub_volume > args["threshold"]).astype(np.int32)
    N = node_voxel_idx.shape[0]
    assert N == reference_nodes.shape[0]

    nlist = resolve_nlist(args["n_list"], N)
    nprobe = resolve_nprobe(args["n_probe"], nlist)
    log.info(f"{N:,} graph nodes  (stride {args['stride']}, thresh "
             f"{args['threshold']}, faiss nlist={nlist} nprobe={nprobe})")

    sc = {}
    if int(args["stride_report_sample"]) > 0 and args["stride"] > 1:
        sc = stride_cost(template_full, sub_volume, args["stride"],
                         args["threshold"], int(args["stride_report_sample"]))
        if sc:
            log.info(f"stride {args['stride']} cost on the reference: "
                     f"r={sc['r']:.4f}, rel-L2={sc['rel_l2']:.3f}, "
                     f"{100 * sc['dropped']:.1f}% of full-res tissue voxels "
                     f"have no node")

    return dict(
        device=device, stride_cost=sc,
        template_full=template_full,
        sub_volume=sub_volume, sub_atlas=sub_atlas,
        node_voxel_idx=node_voxel_idx,
        reference_nodes=reference_nodes,
        N=N, nlist=nlist, nprobe=nprobe,
        eigenvector_dir=Path(args["eigenvector_dir"]),
    )


# =============================================================================
# Graph + eigenbasis for one --knn-method. Ported from maldi_kernel_explorer so
# the cache keys (and therefore the on-disk eigvecs) are byte-identical.
# =============================================================================
def build_graph(args: dict, method: str, com: dict,
                probe_only: bool = False) -> dict:
    device = com["device"]
    sub_volume, sub_atlas = com["sub_volume"], com["sub_atlas"]
    reference_nodes = com["reference_nodes"]
    N, nlist, nprobe = com["N"], com["nlist"], com["nprobe"]

    graphs = KnnGraphCache(cache_dir=com["eigenvector_dir"] / "knn", verbose=True)
    graph_key_parts = {
        "template": args["template_name"], "stride": args["stride"],
        "thresh": args["threshold"], "method": method,
        "k": args["knn_k"], "nlist": nlist, "bbox": None,
    }
    atlas_stem = (Path(args["annotations_file"]).stem
                  if args["annotations_file"] else "noatlas")
    _legacy_atlas = (atlas_stem == "level_15annot")
    force_graph = args["force_recompute_graph"]

    graph_labels = None
    labels_zero_is_region = False

    if method == "faiss":
        graph_key = make_graph_key(graph_key_parts)
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key, method="faiss", coords=reference_nodes,
            k=args["knn_k"], nlist=nlist, nprobe=nprobe, extra=graph_key_parts,
            force_recompute=force_graph, device=device,
        )
    elif method == "anatomical_atlas":
        if sub_atlas is None:
            raise ValueError("anatomical_atlas requires --annotations-file.")
        graph_key_parts["atlas"] = "annotation_coarse_d4"
        graph_key_parts["conn"] = 3
        graph_key = make_graph_key(graph_key_parts)
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key, method="anatomical_atlas", volume=sub_volume,
            threshold=args["threshold"], atlas_volume=sub_atlas, connectivity=3,
            coords=reference_nodes, k=args["knn_k"], nlist=nlist, nprobe=nprobe,
            extra=graph_key_parts, force_recompute=force_graph, device=device,
        )
        graph_labels = labels_for_nodes_from_sub_atlas(
            sub_volume, sub_atlas, args["threshold"])
    elif method in ("faiss_atlas_weighted", "faiss_cluster_weighted"):
        # Both keep the plain faiss topology and only reweight edges that cross a
        # region boundary; they differ in where the region labels come from.
        base_key_parts = dict(graph_key_parts, method="faiss")
        base_key = make_graph_key(base_key_parts)
        knn, edge_index, edge_value = graphs.train_or_load(
            key=base_key, method="faiss", coords=reference_nodes,
            k=args["knn_k"], nlist=nlist, nprobe=nprobe, extra=base_key_parts,
            force_recompute=force_graph, device=device,
        )
        inflation = float(args["cross_region_inflation"])
        if method == "faiss_atlas_weighted":
            if sub_atlas is None:
                raise ValueError("faiss_atlas_weighted requires "
                                 "--annotations-file.")
            graph_labels = labels_for_nodes_from_sub_atlas(
                sub_volume, sub_atlas, args["threshold"])
            root_mode = args["root_handling"]
            if root_mode == "dissolve":
                graph_labels = dissolve_root_labels(
                    graph_labels, reference_nodes.detach().cpu().numpy())
            treat_zero = (root_mode == "cross")
            edge_index, edge_value, info = inflate_cross_region_edges(
                edge_index, edge_value, graph_labels,
                inflation=inflation, treat_zero_as_cross=treat_zero)
            log.info(f"faiss_atlas_weighted (root={root_mode}): "
                     f"{info['n_cross']:,}/{info['n_total']:,} cross-region "
                     f"edges ×{inflation:g}")
            _base_wt = (f"atlas_x{inflation:g}" if _legacy_atlas
                        else f"{atlas_stem}_x{inflation:g}")
            # Legacy 'cross' key stays un-suffixed so old eigvec caches load.
            graph_key_parts["weighting"] = (_base_wt if root_mode == "cross"
                                            else f"{_base_wt}_root{root_mode}")
        else:
            cluster_k, sw = int(args["cluster_k"]), float(
                args["cluster_spatial_weight"])
            cseed = int(args["cluster_seed"])
            graph_labels = labels_for_nodes_from_template_clustering(
                sub_volume, args["threshold"], n_clusters=cluster_k,
                spatial_weight=sw,
                fit_subsample=int(args["cluster_fit_subsample"]), seed=cseed)
            labels_zero_is_region = True     # cluster id 0 is a real region
            edge_index, edge_value, info = inflate_cross_region_edges(
                edge_index, edge_value, graph_labels,
                inflation=inflation, treat_zero_as_cross=False)
            log.info(f"faiss_cluster_weighted (k={cluster_k}, sw={sw:g}): "
                     f"{info['n_cross']:,}/{info['n_total']:,} cross-cluster "
                     f"edges ×{inflation:g}")
            graph_key_parts["weighting"] = (
                f"tmplclust_k{cluster_k}_sw{sw:g}_s{cseed}_x{inflation:g}")
        graph_key = make_graph_key(graph_key_parts)
    else:
        raise ValueError(f"Unknown graph method {method!r}")

    # ---- optional label denoise + hard prune ------------------------------
    # Denoise alone leaves edges untouched (so cache keys are unchanged); prune
    # cuts edges, so it and the denoise that shaped it enter the key.
    n_denoise = int(args["denoise_labels"] or 0)
    if n_denoise > 0 and graph_labels is not None:
        graph_labels = denoise_labels_majority_vote(
            graph_labels, edge_index.cpu().numpy(), n_denoise)
    prune = float(args["prune_cross_region"] or 0.0)
    if prune > 0.0 and graph_labels is not None:
        edge_index, edge_value = prune_cross_region_edges(
            edge_index, edge_value, graph_labels, prune, labels_zero_is_region)
        graph_key_parts["prune"] = f"{prune:g}"
        if n_denoise > 0:
            graph_key_parts["denoise"] = n_denoise
        graph_key = make_graph_key(graph_key_parts)

    bw = float(args["graphbandwidth"])
    laplacian_op = GraphLaplacianOperator(
        edge_value, edge_index, N, torch.tensor(bw, device=device),
        args["laplacian_norm"],
    )

    # ---- eigenvectors: cache first ----------------------------------------
    # The key is fully determined by the graph signature + mode count, so we can
    # see what is loadable BEFORE committing to a solve. num_modes is downgraded
    # to the largest cached count unless --allow-eigensolve says otherwise, and
    # strict_fingerprint makes a stale cache (same key, different edge set) an
    # error rather than a silent multi-GB recompute+overwrite.
    eigvec_dir = com["eigenvector_dir"] / "eigvecs"
    allow_solve = args["allow_eigensolve"] or args["force_recompute_eigvecs"]
    probe_key = make_eig_key({"graph": graph_key, "norm": args["laplacian_norm"],
                              "bw": bw, "modes": args["num_modes"]})
    if probe_only:
        return dict(method=method, key=probe_key, n_edges=int(edge_index.shape[1]),
                    cached_modes=cached_mode_counts(eigvec_dir, probe_key))
    num_modes = resolve_cached_modes(eigvec_dir, probe_key, args["num_modes"],
                                     method, allow_solve)
    eigvec_key_parts = {"graph": graph_key, "norm": args["laplacian_norm"],
                        "bw": bw, "modes": num_modes}
    eigvec_key = make_eig_key(eigvec_key_parts)
    solver = LaplacianEigensolver(
        num_modes=num_modes, backend="cupy", tol=1e-4,
        ncv_min=resolve_ncv_min(num_modes, args["ncv_min"]),
        strict_fingerprint=not allow_solve, verbose=True,
    )
    eigval, eigvec = solver.compute_or_load(
        laplacian_op, cache_dir=eigvec_dir,
        key=eigvec_key, graphbandwidth=bw,
        laplacian_normalization=args["laplacian_norm"], extra=eigvec_key_parts,
        force_recompute=args["force_recompute_eigvecs"], device=device,
        allow_larger_modes=True,
    )
    if eigvec.shape[0] != N:
        raise RuntimeError(
            f"Node-count mismatch: {N} nodes but eigvec has {eigvec.shape[0]} "
            f"rows — the cached graph/eigvecs were built on a different node set "
            f"(stride/threshold). Delete the stale cache under "
            f"{com['eigenvector_dir']} or pass --force-recompute-eigvecs.")
    log.info(f"[{method}] {eigvec.shape[1]} eigenmodes, "
             f"λ ∈ [{float(eigval[0]):.4g}, {float(eigval[-1]):.4g}], "
             f"{edge_index.shape[1]:,} edges")

    # Degree weights define the inner product the eigenvectors are orthogonal in
    # (see spectral_coeffs). For 'symmetric' they are all 1 by construction.
    if args["laplacian_norm"] == "randomwalk":
        degree = laplacian_op.degree_mat.to(eigvec.dtype)
    else:
        degree = torch.ones(N, device=device, dtype=eigvec.dtype)

    return dict(
        method=method, key=eigvec_key,
        laplacian_op=laplacian_op, eigval=eigval, eigvec=eigvec,
        degree=degree, n_edges=int(edge_index.shape[1]),
        edge_index=edge_index.detach().cpu().numpy(),
        edge_value=edge_value.detach().cpu().numpy(),
        graph_labels=graph_labels,
    )


# =============================================================================
# Signals on the nodes
# =============================================================================
def density_at_nodes(sub_volume: np.ndarray, node_voxel_idx: np.ndarray,
                     sigma: float = 0.0) -> np.ndarray:
    """Reference intensity at each node, optionally σ-blurred in voxel space
    first. σ is in *strided* voxels: at stride 4, σ=2 ≈ 8 full-res voxels."""
    vol = sub_volume.astype(np.float32, copy=False)
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        vol = gaussian_filter(vol, sigma=float(sigma))
    return vol[node_voxel_idx[:, 0], node_voxel_idx[:, 1], node_voxel_idx[:, 2]]


def apply_laplacian(laplacian_op: GraphLaplacianOperator,
                    f: torch.Tensor) -> torch.Tensor:
    """Exact L·f via the sparse operator — no eigenvectors involved."""
    return laplacian_op._matmul(f.unsqueeze(-1)).squeeze(-1)


def spectral_coeffs(g: dict, f: torch.Tensor) -> torch.Tensor:
    """Expansion coefficients of `f` in the graph's eigenbasis.

    The cached eigenvectors are NOT ℓ2-orthonormal under 'randomwalk'
    normalization: L_rw = D^{-1/2} L_sym D^{1/2}, so its eigenvectors are
    φ_k ∝ D^{-1/2} u_k with u_k the orthonormal eigenvectors of L_sym, and the
    solver then ℓ2-normalizes each column. They are orthogonal in the
    D-weighted inner product instead, which gives

        c_k = (φ_kᵀ D f) / (φ_kᵀ D φ_k).

    Using a plain φ_kᵀf here would silently produce a wrong reconstruction
    (nonzero residual even at N = all modes). Under 'symmetric' normalization
    D = I and this reduces to the usual dot product.
    """
    Phi, d = g["eigvec"], g["degree"]
    denom = (Phi * Phi * d.unsqueeze(-1)).sum(dim=0).clamp(min=1e-30)
    return (Phi * d.unsqueeze(-1)).transpose(0, 1) @ f / denom


def reconstruct(g: dict, coeffs: torch.Tensor, n_modes: int):
    """(f_N, L_N·f) from the leading `n_modes` eigenpairs."""
    n = int(np.clip(n_modes, 1, g["eigvec"].shape[1]))
    Phi = g["eigvec"][:, :n]
    c = coeffs[:n]
    f_n = Phi @ c
    Lf_n = Phi @ (g["eigval"][:n] * c)
    return f_n, Lf_n


def recon_metrics(g: dict, f, Lf, f_n, Lf_n) -> dict:
    """How good the N-mode reconstruction is, from three angles.

    Why three: on a raw (unsmoothed) template the single number
    ‖L·f − L_N·f‖/‖L·f‖ pins at ~1.0 for any N a solver can reach, and stops
    discriminating. The reason is structural, not a bug — L_N·f = L·f_N exactly
    (each φ_k is a true eigenvector), so that residual is really ‖L(f − f_N)‖,
    and the out-of-band part of f gets amplified by the unresolved eigenvalues,
    which run to O(1/bandwidth²) while λ_N stays O(1). Hence:

      res_L  ‖L·f − L_N·f‖ / ‖L·f‖   how much of the operator's *action* the
                                     truncation misses. Near 1 whenever f has
                                     any voxel-scale content.
      res_f  ‖f − f_N‖_D / ‖f‖_D     how much of the *field* the basis misses,
                                     in the D-weighted norm the modes are
                                     orthogonal in. This is the one that moves.
      cos    ⟨L·f, L_N·f⟩ / (‖·‖‖·‖) whether the reconstruction at least points
                                     the same way — i.e. does the coarse
                                     structure of L·f survive truncation, even
                                     at a fraction of the amplitude.
      amp    ‖L_N·f‖ / ‖L·f‖         that amplitude fraction.
    """
    d = g["degree"]

    def dnorm(v):
        return float(torch.sqrt((d * v * v).sum()).clamp(min=0))

    Lf_norm = float(torch.linalg.norm(Lf))
    Lf_n_norm = float(torch.linalg.norm(Lf_n))
    dot = float((Lf * Lf_n).sum())
    return dict(
        res_L=float(torch.linalg.norm(Lf - Lf_n)) / max(Lf_norm, 1e-30),
        res_f=dnorm(f - f_n) / max(dnorm(f), 1e-30),
        cos=dot / max(Lf_norm * Lf_n_norm, 1e-30),
        amp=Lf_n_norm / max(Lf_norm, 1e-30),
        Lf_norm=Lf_norm,
    )


def mode_ladder(num_modes: int, explicit=None) -> list:
    """Mode counts for the dropdown: a 1/2/5 ladder capped at `num_modes`."""
    if explicit:
        vals = sorted({int(np.clip(v, 1, num_modes)) for v in explicit})
        return vals or [num_modes]
    vals, step = [], 1
    while step <= num_modes:
        for m in (1, 2, 5):
            v = m * step
            if 1 <= v <= num_modes:
                vals.append(v)
        step *= 10
    vals.append(num_modes)
    return sorted(set(vals))


def graph_distance_diagnostic(g: dict, node_voxel_idx: np.ndarray,
                              n_pairs: int, seed: int = 0) -> dict | None:
    """Compare geometric graph shortest paths with straight-line distance.

    Edge lengths come from node coordinates, not ``edge_value``. Weighted graph
    variants deliberately inflate some Laplacian edges; this diagnostic asks
    only whether the retained graph topology densely traces physical space.
    """
    n_pairs = int(n_pairs)
    n_nodes = int(node_voxel_idx.shape[0])
    if n_pairs <= 0 or n_nodes < 2:
        return None
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    edges = np.asarray(g["edge_index"], dtype=np.int64)
    u, v = edges[0], edges[1]
    xyz = np.asarray(node_voxel_idx, dtype=np.float64)
    edge_len = np.linalg.norm(xyz[u] - xyz[v], axis=1)
    adjacency = coo_matrix(
        (edge_len, (u, v)), shape=(n_nodes, n_nodes)).tocsr()
    adjacency = adjacency.maximum(adjacency.T)

    rng = np.random.default_rng(seed)
    n_candidates = max(8 * n_pairs, 128)
    src = rng.integers(0, n_nodes, size=n_candidates)
    dst = rng.integers(0, n_nodes, size=n_candidates)
    euclid = np.linalg.norm(xyz[src] - xyz[dst], axis=1)
    positive = euclid > 0
    if not positive.any():
        return None
    # Nearby pairs agree almost trivially. Use the more illustrative, spatially
    # separated half of the random candidates.
    cutoff = float(np.percentile(euclid[positive], 50.0))
    keep = positive & (euclid >= cutoff)
    src = src[keep][:n_pairs]
    dst = dst[keep][:n_pairs]
    euclid = euclid[keep][:n_pairs]
    if src.size == 0:
        return None

    distances, predecessors = dijkstra(
        adjacency, directed=False, indices=src, return_predecessors=True)
    graph_dist = distances[np.arange(src.size), dst]
    finite = np.isfinite(graph_dist) & (euclid > 0)
    if not finite.any():
        return None
    src, dst = src[finite], dst[finite]
    euclid, graph_dist = euclid[finite], graph_dist[finite]
    predecessors = predecessors[finite]
    ratio = graph_dist / euclid

    # Highlight a representative pair rather than an unusually good/bad one.
    show = int(np.argmin(np.abs(ratio - np.median(ratio))))
    source, target = int(src[show]), int(dst[show])
    path = [target]
    while path[-1] != source:
        previous = int(predecessors[show, path[-1]])
        if previous < 0:
            return None
        path.append(previous)
    path.reverse()

    return dict(
        source=source, target=target, path=np.asarray(path, dtype=np.int64),
        euclidean=euclid, graph=graph_dist, ratio=ratio,
        median_ratio=float(np.median(ratio)),
        p90_relative_error=float(np.percentile(np.abs(ratio - 1.0), 90.0)),
        correlation=(float(np.corrcoef(euclid, graph_dist)[0, 1])
                     if euclid.size > 1 else 1.0),
    )


def anatomical_edge_overlay(g: dict, node_voxel_idx: np.ndarray,
                            inflation: float, max_edges: int,
                            seed: int = 0) -> dict | None:
    """Sample within/cross-region edges and their atlas-induced attenuation."""
    labels = g.get("graph_labels")
    if labels is None:
        return None
    edges = np.asarray(g["edge_index"], dtype=np.int64)
    d2_after = np.asarray(g["edge_value"], dtype=np.float64)
    u, v = edges[0], edges[1]
    # Avoid drawing reciprocal duplicates when a graph stores both directions.
    unique_direction = u < v
    if unique_direction.any():
        u, v, d2_after = (
            u[unique_direction], v[unique_direction],
            d2_after[unique_direction],
        )
    cross = np.asarray(labels)[u] != np.asarray(labels)[v]
    rng = np.random.default_rng(seed)

    def sample(mask):
        idx = np.flatnonzero(mask)
        cap = max(int(max_edges), 0)
        if idx.size > cap:
            idx = rng.choice(idx, cap, replace=False)
        return idx

    all_cross_idx = np.flatnonzero(cross)
    same_idx, cross_idx = sample(~cross), sample(cross)
    if cross_idx.size == 0:
        return None
    xyz = np.asarray(node_voxel_idx)

    def segments(idx):
        return [xyz[[u[i], v[i]]].astype(np.float32) for i in idx]

    bw = float(g["laplacian_op"].graphbandwidth.squeeze())
    factor = max(float(inflation), 1.0)
    cross_after = np.exp(-d2_after[cross_idx] / (4.0 * bw * bw))
    cross_before = np.exp(
        -(d2_after[cross_idx] / factor) / (4.0 * bw * bw))
    attenuation = cross_after / np.maximum(cross_before, 1e-300)
    # Boundaries should not disappear merely because edge rendering is sampled:
    # use endpoints from every cross-region edge for the boundary point layer.
    boundary_nodes = np.unique(
        np.concatenate([u[all_cross_idx], v[all_cross_idx]])).astype(np.int64)
    return dict(
        within=segments(same_idx),
        cross=segments(cross_idx),
        boundary_nodes=xyz[boundary_nodes].astype(np.float32),
        before_affinity=cross_before,
        after_affinity=cross_after,
        attenuation=attenuation,
        n_cross_total=int(cross.sum()),
        n_within_total=int((~cross).sum()),
    )


# =============================================================================
# Rasterization — node values back onto the strided grid.
# =============================================================================
def rasterize_sub(values: np.ndarray, node_voxel_idx: np.ndarray,
                  sub_shape: tuple) -> np.ndarray:
    """Scatter a per-node signal into a strided-resolution volume.

    Layers are then placed with napari `scale=(stride, stride, stride)`, which
    puts them in full-res template voxel coordinates without allocating a
    full-res array per layer.
    """
    grid = np.zeros(sub_shape, dtype=np.float32)
    grid[node_voxel_idx[:, 0], node_voxel_idx[:, 1],
         node_voxel_idx[:, 2]] = np.asarray(values, np.float32)
    return grid


def stride_cost(template_full: np.ndarray, sub_volume: np.ndarray, stride: int,
                threshold: int, n_sample: int = 2_000_000, seed: int = 0) -> dict:
    """Quantify what striding throws away, on the reference itself.

    The graph sees full-res voxel (i,j,k) as `sub_volume[i//s, j//s, k//s]` —
    the same block placement napari reproduces via `scale`. So comparing the
    true 25 µm value against its block value over a random sample of tissue
    voxels measures the stride effect directly:

      r         correlation between true and block-sampled intensity
      rel_l2    ‖true − block‖ / ‖true‖
      dropped   fraction of full-res tissue voxels whose block falls below the
                node threshold — tissue with no node at all, i.e. structure the
                Laplacian cannot represent no matter how many modes you keep.

    Sampled rather than exhaustive: 77 M voxels × float would cost hundreds of
    MB for a number that converges at ~10^6 samples.
    """
    rng = np.random.default_rng(seed)
    z, y, x = template_full.shape
    # Oversample and filter, rather than materializing a 77 M-element mask.
    idx = np.empty((0, 3), dtype=np.int64)
    tries = 0
    while idx.shape[0] < n_sample and tries < 8:
        cand = np.stack([rng.integers(0, z, 4 * n_sample),
                         rng.integers(0, y, 4 * n_sample),
                         rng.integers(0, x, 4 * n_sample)], axis=1)
        vals = template_full[cand[:, 0], cand[:, 1], cand[:, 2]]
        idx = np.concatenate([idx, cand[vals > threshold]], axis=0)
        tries += 1
    idx = idx[:n_sample]
    if idx.shape[0] == 0:
        return {}
    true = template_full[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float64)
    b = idx // stride
    block = sub_volume[b[:, 0], b[:, 1], b[:, 2]].astype(np.float64)
    r = float(np.corrcoef(true, block)[0, 1])
    rel = float(np.linalg.norm(true - block) / max(np.linalg.norm(true), 1e-30))
    return dict(n_sampled=int(idx.shape[0]), r=r, rel_l2=rel,
                dropped=float((block <= threshold).mean()))


def compute_amax_pct(values: np.ndarray, pct: float = 99.5) -> float:
    """`pct`-th percentile of |values| over nonzero entries (outlier-robust)."""
    a = np.abs(np.asarray(values))
    nz = a[a > 0]
    if nz.size == 0:
        return 1.0
    return max(float(np.percentile(nz, pct)), 1e-12)


def categorical_node_colors(labels: np.ndarray) -> np.ndarray:
    """Stable, bright RGBA colors for integer atlas/cluster labels."""
    from matplotlib.colors import hsv_to_rgb
    labels = np.asarray(labels, dtype=np.int64)
    rgba = np.ones((labels.size, 4), dtype=np.float32)
    # Multiplication by the golden-ratio conjugate spreads consecutive atlas
    # IDs around the hue wheel instead of assigning nearly identical colors.
    hue = np.mod(labels.astype(np.float64) * 0.61803398875, 1.0)
    hsv = np.column_stack([
        hue,
        np.full(labels.size, 0.72),
        np.full(labels.size, 1.0),
    ])
    rgba[:, :3] = hsv_to_rgb(hsv).astype(np.float32)
    rgba[labels == 0, :3] = (0.72, 0.72, 0.72)
    return rgba


def _diverging_cmap(gamma: float = 1.0):
    """RdBu_r with a transparent centre and signed γ baked into the lookup, so
    zero background disappears and γ<1 exposes low-magnitude structure."""
    from napari.utils.colormaps import Colormap
    # Use an odd-sized LUT so 0.5 is an actual sample. With 256 entries, zero
    # falls between two pale, slightly opaque entries; 3-D ray accumulation of
    # that tiny opacity over the raster's zero-valued background produces the
    # bright bounding-box "glow".
    pos = np.linspace(0.0, 1.0, 257)
    raw_signed = 2.0 * pos - 1.0
    signed = raw_signed.copy()
    if gamma != 1.0:
        signed = np.sign(signed) * (np.abs(signed) ** gamma)
    rgba = cm.get_cmap("RdBu_r")(np.clip(0.5 + 0.5 * signed, 0.0, 1.0)).copy()
    # Suppress numerical/interpolation leakage near zero as well as exact zero.
    # The deadband is in normalized signal units, before gamma shaping.
    deadband = 0.04
    rgba[:, 3] = np.clip(
        (np.abs(raw_signed) - deadband) / (1.0 - deadband), 0.0, 1.0)
    rgba[pos == 0.5, 3] = 0.0
    return Colormap(colors=rgba, name=f"RdBu_r_div_g{gamma:.2g}")


def _sequential_cmap(gamma: float = 1.0):
    """magma with an alpha ramp from transparent, for the non-negative fields."""
    from napari.utils.colormaps import Colormap
    pos = np.linspace(0.0, 1.0, 256)
    warped = pos ** gamma
    rgba = cm.get_cmap("magma")(warped).copy()
    rgba[:, 3] = warped
    return Colormap(colors=rgba, name=f"magma_a_g{gamma:.2g}")


# =============================================================================
# Eigenvalue spectrum widget (matplotlib in a Qt dock)
# =============================================================================
class SpectrumWidget:
    """λ_k vs k, linear + log-log with a k^(2/3) (3-D Weyl) reference.

    Reading it: smooth power-law growth → clean Laplacian; a flat segment near
    k=0 → weakly-connected components; stairsteps → discrete graph clusters.
    The green span is the truncation N currently feeding L_N·f.
    """

    def __init__(self):
        self.widget = None
        try:
            from qtpy.QtWidgets import QWidget, QVBoxLayout
            from matplotlib.figure import Figure
            try:
                from matplotlib.backends.backend_qtagg import FigureCanvas
            except ImportError:
                from matplotlib.backends.backend_qt5agg import (
                    FigureCanvasQTAgg as FigureCanvas)
        except ImportError:
            return
        self._fig = Figure(figsize=(4.2, 5.2), tight_layout=True)
        self._ax_lin = self._fig.add_subplot(2, 1, 1)
        self._ax_log = self._fig.add_subplot(2, 1, 2)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumHeight(420)
        self.widget = QWidget()
        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self._canvas)
        self._lam = None
        self._N = 1
        self._title = ""

    def set_spectrum(self, eigval: torch.Tensor, title: str = ""):
        self._lam = eigval.detach().cpu().numpy()
        self._title = title
        self._redraw()

    def set_N(self, N: int):
        self._N = int(N)
        self._redraw()

    def _redraw(self):
        if self.widget is None or self._lam is None:
            return
        lam, ks = self._lam, np.arange(self._lam.shape[0])
        for ax in (self._ax_lin, self._ax_log):
            ax.clear()

        self._ax_lin.plot(ks, lam, marker=".", ms=2.5, lw=0.7, color="#3578a8")
        self._ax_lin.axhline(0.0, color="gray", lw=0.5, alpha=0.5)
        self._ax_lin.set_xlabel("k")
        self._ax_lin.set_ylabel(r"$\lambda_k$")
        self._ax_lin.set_title(f"λ_k  (linear) — {self._title}", fontsize=9)
        self._ax_lin.grid(True, alpha=0.3)

        nz = lam > 1e-12
        if nz.any():
            kp, lp = ks[nz], lam[nz]
            self._ax_log.loglog(kp, lp, marker=".", ms=2.5, lw=0.7,
                                color="#3578a8")
            if kp.size > 10:
                a = max(kp.size // 4, 1)
                ref = float(lp[a]) * (kp.astype(np.float64) / float(kp[a])) ** (2 / 3)
                self._ax_log.loglog(kp, ref, "--", lw=0.7, alpha=0.7,
                                    color="#cc3344",
                                    label=r"$k^{2/3}$ (3-D Weyl)")
                self._ax_log.legend(fontsize=8, loc="lower right")
        self._ax_log.set_xlabel("k")
        self._ax_log.set_ylabel(r"$\lambda_k$")
        self._ax_log.set_title("λ_k  (log-log)", fontsize=9)
        self._ax_log.grid(True, alpha=0.3, which="both")

        self._ax_lin.axvspan(0, max(self._N, 0.1), alpha=0.15, color="#3aaa54")
        self._ax_log.axvspan(1, max(self._N, 1.0), alpha=0.15, color="#3aaa54")
        self._canvas.draw_idle()


# =============================================================================
# Cache report — what can be loaded without solving anything.
# =============================================================================
def report_cache(args: dict, com: dict):
    eigvec_dir = com["eigenvector_dir"] / "eigvecs"
    print("\n" + "=" * 72)
    print(f"Eigenvector cache under {eigvec_dir}")
    print(f"requested: {args['num_modes']} modes · bw={args['graphbandwidth']:g} "
          f"· {args['laplacian_norm']} · stride {args['stride']} "
          f"· thresh {args['threshold']}")
    print("=" * 72)
    for method in args["graph_methods"]:
        p = build_graph(args, method, com, probe_only=True)
        counts = p["cached_modes"]
        if any(c >= args["num_modes"] for c in counts):
            verdict = f"OK — loads {args['num_modes']} modes from cache"
        elif counts:
            verdict = (f"DOWNGRADE — will load {max(counts)} modes "
                       f"(pass --allow-eigensolve to solve for "
                       f"{args['num_modes']})")
        else:
            verdict = "MISS — nothing cached for this graph signature"
        print(f"\n[{method}]  {p['n_edges']:,} edges")
        print(f"  key           : {p['key']}")
        print(f"  cached modes  : {counts if counts else '(none)'}")
        print(f"  verdict       : {verdict}")
        if not counts:
            print(f"  cached keys for method={method}:")
            print(_sibling_report(eigvec_dir, method, p["key"]))
    print("\n" + "=" * 72 + "\n")


# =============================================================================
# Headless report — residual vs N for every graph in the dropdown.
# =============================================================================
def report_residuals(args: dict, com: dict):
    ladder = mode_ladder(args["num_modes"], args["mode_ladder"])
    sigma = max(float(args["density_smooth_sigma"]), 0.0)
    f_np = density_at_nodes(com["sub_volume"], com["node_voxel_idx"], sigma)
    f = torch.as_tensor(f_np, device=com["device"], dtype=torch.float32)

    print("\n" + "=" * 72)
    print(f"Spectral sufficiency of L·f   (f = reference density, σ={sigma:g})")
    print(f"{com['N']:,} nodes · {args['num_modes']} modes loaded per graph")
    print("=" * 72)
    for method in args["graph_methods"]:
        g = build_graph(args, method, com)
        Lf = apply_laplacian(g["laplacian_op"], f)
        coeffs = spectral_coeffs(g, f)
        # A graph whose cache tops out below --num-modes loads fewer modes
        # (see resolve_cached_modes), so cap the ladder at what it really has.
        K = int(g["eigvec"].shape[1])
        rows = [n for n in mode_ladder(K, args["mode_ladder"])]
        print(f"\n[{method}]  {g['n_edges']:,} edges   {K} modes   "
              f"‖L·f‖={float(torch.linalg.norm(Lf)):.4g}")
        print(f"  {'N':>6}  {'res_L':>8}  {'res_f':>8}  {'cos':>8}  "
              f"{'amp':>9}  {'λ_N-1':>10}")
        for n in rows:
            f_n, Lf_n = reconstruct(g, coeffs, n)
            m = recon_metrics(g, f, Lf, f_n, Lf_n)
            print(f"  {n:>6}  {m['res_L']:>8.4f}  {m['res_f']:>8.4f}  "
                  f"{m['cos']:>8.4f}  {m['amp']:>9.2e}  "
                  f"{float(g['eigval'][n - 1]):>10.4g}")
        if not args["keep_graphs"]:
            del g
            if com["device"].type == "cuda":
                torch.cuda.empty_cache()
    print("\nres_L = ‖L·f−L_N·f‖/‖L·f‖ (operator action missed; pins near 1 on a"
          "\n        rough f — the unresolved λ run to O(1/bw²))"
          "\nres_f = ‖f−f_N‖_D/‖f‖_D (field content missed — the metric that "
          "moves)"
          "\ncos   = alignment of L_N·f with L·f (does coarse structure survive)"
          "\namp   = ‖L_N·f‖/‖L·f‖ (amplitude fraction)"
          "\nCompare graphs at fixed N: sharper interfaces push action to higher"
          "\nfrequency, so the weighted graph needs more modes for the same res.")
    print("=" * 72 + "\n")


# =============================================================================
# Viewer
# =============================================================================
def run_viewer(args: dict, com: dict):
    import napari
    from magicgui import magicgui
    from qtpy.QtWidgets import QLabel

    stride = int(args["stride"])
    scale = (stride, stride, stride)
    sub_shape = com["sub_volume"].shape
    ladder = mode_ladder(args["num_modes"], args["mode_ladder"])
    initial_modes = int(args["initial_modes"] or args["num_modes"])
    initial_modes = min(ladder, key=lambda v: abs(v - initial_modes))

    # ---- state -----------------------------------------------------------
    st = {
        "method": args["graph_methods"][0],
        "modes": initial_modes,
        "sigma": max(float(args["density_smooth_sigma"]), 0.0),
        "gamma": float(np.clip(args["gamma"], 0.1, 2.0)),
        "pct": float(np.clip(args["contrast_pct"], 50.0, 100.0)),
    }
    graphs: dict = {}          # method -> graph dict (see --keep-graphs)
    cur = {"g": None, "f": None, "coeffs": None, "Lf": None}

    def get_graph(method: str) -> dict:
        if method in graphs:
            return graphs[method]
        if not args["keep_graphs"]:
            graphs.clear()
            if com["device"].type == "cuda":
                torch.cuda.empty_cache()
        log.info(f"building/loading graph '{method}' …")
        g = build_graph(args, method, com)
        graphs[method] = g
        # A graph whose cache tops out below --num-modes loads fewer modes, so
        # the dropdown must not offer N values this basis cannot serve.
        sync_mode_choices(int(g["eigvec"].shape[1]))
        return g

    # ---- viewer + layers -------------------------------------------------
    viewer = napari.Viewer(title="Laplacian ↔ eigenvector reconstruction")
    viewer.dims.ndisplay = int(args["ndisplay"])

    # Reference at native 25 µm and/or as the graph actually samples it. Both
    # sit in the same full-resolution voxel frame (the strided one via
    # scale=stride³), so toggling between them IS the stride effect: same
    # anatomy, one at 77 M voxels, one at N nodes.
    #
    # Contrast limits are set explicitly rather than left to napari's auto
    # (0..255 here): starting at the node threshold drops the sub-threshold
    # haze — ~3% of the volume sits at 1..5 — which is exactly the tissue the
    # graph has no node for, so it has no business lighting up the backdrop.
    thr = float(args["threshold"])
    tissue = com["sub_volume"][com["sub_volume"] > args["threshold"]]
    t_clim = (thr, float(np.percentile(tissue, 99.5)) if tissue.size else 255.0)
    t_kwargs = dict(colormap="gray", contrast_limits=t_clim,
                    opacity=float(args["template_opacity"]),
                    blending="translucent",
                    rendering=args["template_rendering"])
    if args["template_rendering"] == "attenuated_mip":
        t_kwargs["attenuation"] = float(args["template_attenuation"])
    if args["template_rendering"] == "iso":
        t_kwargs["iso_threshold"] = float(
            args["template_iso_threshold"]
            if args["template_iso_threshold"] is not None else thr)

    tres = args["template_res"]
    if tres in ("full", "both"):
        viewer.add_image(com["template_full"],
                         name="template (reference, 25 µm)",
                         visible=False, **t_kwargs)
    if tres in ("strided", "both"):
        viewer.add_image(com["sub_volume"],
                         name=f"template @ stride {stride}  (what the graph sees)",
                         scale=scale,
                         # nearest: never interpolate the lattice away — each
                         # node must read as the stride³ block it really is.
                         interpolation2d="nearest", interpolation3d="nearest",
                         visible=False, **t_kwargs)

    zeros = np.zeros(sub_shape, np.float32)

    # Show that striding still leaves a densely sampled spatial graph. This is
    # a points layer, not another translucent volume, so it cannot create the
    # white compositing haze of overlapping image layers.
    rng = np.random.default_rng(0)
    n_show = min(max(int(args["graph_node_display_sample"]), 0), com["N"])
    show_idx = (rng.choice(com["N"], n_show, replace=False)
                if n_show < com["N"] else np.arange(com["N"]))
    graph_nodes = viewer.add_points(
        com["node_voxel_idx"][show_idx], name="graph nodes (display sample)",
        scale=scale, size=2.5, face_color="#20f7d4",
        opacity=1.0, blending="opaque", visible=True,
    )
    pair_nodes = viewer.add_points(
        np.empty((0, 3), dtype=np.float32), name="distance pair",
        scale=scale, size=5.0, face_color="#ffd166",
        opacity=1.0, visible=True,
    )
    empty_3d_segment = np.zeros((2, 3), dtype=np.float32)
    euclid_line = viewer.add_shapes(
        [empty_3d_segment], shape_type="line", name="Euclidean straight line",
        scale=scale, edge_color="#ffd166", edge_width=2.5,
        face_color="transparent", visible=True,
    )
    graph_path = viewer.add_shapes(
        [empty_3d_segment], shape_type="path", name="graph shortest path",
        scale=scale, edge_color="#ef476f", edge_width=2.5,
        face_color="transparent", visible=True,
    )
    atlas_boundaries = viewer.add_points(
        np.empty((0, 3), dtype=np.float32), name="atlas boundary nodes",
        scale=scale, size=5.0, face_color="#ff2bd6",
        opacity=1.0, blending="opaque", visible=False,
    )
    within_edges = viewer.add_shapes(
        [empty_3d_segment], shape_type="line",
        name="within-region edges (sample)", scale=scale,
        edge_color="#27d7c4", edge_width=0.7,
        face_color="transparent", opacity=0.35, visible=False,
    )
    cross_before = viewer.add_shapes(
        [empty_3d_segment], shape_type="line",
        name="cross-region affinity (before atlas)", scale=scale,
        edge_color="#ffb000", edge_width=1.8,
        face_color="transparent", opacity=0.9, visible=False,
    )
    cross_after = viewer.add_shapes(
        [empty_3d_segment], shape_type="line",
        name="cross-region affinity (after atlas)", scale=scale,
        edge_color="#9d4edd", edge_width=1.8,
        face_color="transparent", opacity=0.65, visible=False,
    )

    def _add(name, cmap, visible):
        # Every signal layer lives on the node lattice, so it is drawn with
        # nearest interpolation for the same reason as the strided template.
        return viewer.add_image(zeros.copy(), name=name, scale=scale,
                                colormap=cmap, contrast_limits=(-1.0, 1.0),
                                rendering="translucent", opacity=0.95,
                                blending="translucent", visible=visible,
                                interpolation2d="nearest",
                                interpolation3d="nearest")

    lyr_f = _add("f  (reference at nodes)", _sequential_cmap(st["gamma"]), False)
    lyr_f.contrast_limits = (0.0, 1.0)
    lyr_fn = _add("f_N  (N-mode reconstruction of f)",
                  _sequential_cmap(st["gamma"]), False)
    lyr_fn.contrast_limits = (0.0, 1.0)
    lyr_Lf = _add("L·f  (exact graph Laplacian)", _diverging_cmap(st["gamma"]),
                  False)
    lyr_LfN = _add("L_N·f  (N-mode reconstruction)",
                   _diverging_cmap(st["gamma"]), False)
    lyr_res = _add("residual  L·f − L_N·f", _diverging_cmap(st["gamma"]), False)

    spectrum = SpectrumWidget()
    if spectrum.widget is not None:
        viewer.window.add_dock_widget(spectrum.widget, name="λ_k spectrum",
                                      area="left")

    info = QLabel()
    info.setWordWrap(True)
    info.setStyleSheet("QLabel { font-size: 11px; }")

    # Static, so build it once: the resolution the Laplacian actually lives at.
    sc = com.get("stride_cost") or {}
    tf_size = com["template_full"].size
    stride_html = (
        "<hr><b>resolution</b><br>"
        f"reference {tf_size:,} voxels @ 25 µm → <b>{com['N']:,} nodes</b> "
        f"@ stride {stride} ({100.0 * com['N'] / tf_size:.2f}%)<br>")
    if sc:
        stride_html += (
            f"block-sampling the reference at this stride: r={sc['r']:.4f}, "
            f"rel-L2={sc['rel_l2']:.3f}<br>"
            f"{100 * sc['dropped']:.1f}% of full-res tissue voxels land on a "
            f"block with no node<br>")
    stride_html += (
        "<span style='font-size:10px'>Every signal layer is drawn "
        "nearest-neighbour, so one node reads as the "
        f"{stride}×{stride}×{stride} block it is. Toggle "
        f"'template @ stride {stride}' against the 25 µm reference to see what "
        "the graph never had. No number of modes recovers detail below this "
        "lattice — that ceiling is the node set, not the "
        "truncation.</span>")

    # ---- render ----------------------------------------------------------
    def recompute_field():
        """f, its coefficients in the current basis, and the exact L·f."""
        f_np = density_at_nodes(com["sub_volume"], com["node_voxel_idx"],
                                st["sigma"])
        f = torch.as_tensor(f_np, device=com["device"], dtype=torch.float32)
        cur["f"] = f
        cur["Lf"] = apply_laplacian(cur["g"]["laplacian_op"], f)
        cur["coeffs"] = spectral_coeffs(cur["g"], f)

    def _set_layer(layer, vals: np.ndarray, symmetric: bool):
        layer.data = rasterize_sub(vals, com["node_voxel_idx"], sub_shape)
        if symmetric:
            amax = compute_amax_pct(vals, st["pct"])
            layer.colormap = _diverging_cmap(st["gamma"])
            layer.contrast_limits = (-amax, amax)
            return amax
        vmax = compute_amax_pct(vals, st["pct"])
        layer.colormap = _sequential_cmap(st["gamma"])
        layer.contrast_limits = (0.0, vmax)
        return vmax

    def render(changed_graph: bool, changed_field: bool):
        # f and the exact L·f depend on the graph and σ, not on N — a mode
        # change only redoes the projection, which is one matvec.
        n = int(st["modes"])
        if changed_graph or changed_field:
            recompute_field()
        f, Lf, g = cur["f"], cur["Lf"], cur["g"]
        f_n, Lf_n = reconstruct(g, cur["coeffs"], n)
        res = Lf - Lf_n

        f_np, Lf_np = f.cpu().numpy(), Lf.cpu().numpy()
        if changed_graph or changed_field:
            _set_layer(lyr_f, f_np, symmetric=False)
            _set_layer(lyr_Lf, Lf_np, symmetric=True)
        _set_layer(lyr_fn, f_n.cpu().numpy(), symmetric=False)
        _set_layer(lyr_LfN, Lf_n.cpu().numpy(), symmetric=True)
        _set_layer(lyr_res, res.cpu().numpy(), symmetric=True)

        m = recon_metrics(g, f, Lf, f_n, Lf_n)
        K = int(g["eigvec"].shape[1])

        if changed_graph:
            spectrum.set_spectrum(g["eigval"], GRAPH_LABELS[g["method"]])
            gd = graph_distance_diagnostic(
                g, com["node_voxel_idx"], args["distance_pairs"], seed=0)
            cur["graph_distance"] = gd
            if gd is None:
                pair_nodes.data = np.empty((0, 3), dtype=np.float32)
                euclid_line.data = []
                graph_path.data = []
            else:
                xyz = com["node_voxel_idx"]
                endpoints = xyz[[gd["source"], gd["target"]]]
                pair_nodes.data = endpoints
                euclid_line.data = [endpoints]
                graph_path.data = [xyz[gd["path"]]]
            anatomical = anatomical_edge_overlay(
                g, com["node_voxel_idx"], args["cross_region_inflation"],
                args["anatomical_edge_display_sample"], seed=1)
            cur["anatomical_overlay"] = anatomical
            if anatomical is None:
                graph_nodes.visible = True
                graph_nodes.face_color = "#20f7d4"
                atlas_boundaries.data = np.empty((0, 3), dtype=np.float32)
                within_edges.data = []
                cross_before.data = []
                cross_after.data = []
                for layer in (
                        atlas_boundaries, within_edges, cross_before, cross_after):
                    layer.visible = False
            else:
                graph_nodes.visible = True
                graph_nodes.face_color = categorical_node_colors(
                    np.asarray(g["graph_labels"])[show_idx])
                atlas_boundaries.data = anatomical["boundary_nodes"]
                within_edges.data = anatomical["within"]
                cross_before.data = anatomical["cross"]
                cross_after.data = anatomical["cross"]
                # Region-colored nodes are the simple default story. These
                # diagnostic overlays remain available as manual toggles.
                atlas_boundaries.visible = False
                within_edges.visible = False
                cross_before.visible = False
                cross_after.visible = False
                # Keep the layer inspectable in 3-D. Attenuation is reported
                # numerically in the info panel; mapping a factor near zero
                # directly to opacity made the layer functionally invisible.
                med_att = float(np.median(anatomical["attenuation"]))
                cross_after.opacity = float(np.clip(
                    0.35 + 0.5 * np.sqrt(med_att), 0.35, 0.85))
                log.info(
                    f"[{g['method']}] anatomical overlay: "
                    f"{anatomical['n_within_total']:,} within edges, "
                    f"{anatomical['n_cross_total']:,} cross edges, "
                    f"{len(anatomical['boundary_nodes']):,} boundary nodes, "
                    f"median affinity attenuation ×{med_att:.3e}")
        spectrum.set_N(n)

        gd = cur.get("graph_distance")
        if gd is None:
            distance_html = (
                "<hr><b>spatial graph distance</b><br>"
                "<span style='font-size:10px'>disabled or no connected random "
                "pairs found</span>")
        else:
            distance_html = (
                "<hr><b>spatial graph distance ≈ Euclidean distance</b><br>"
                f"{len(gd['ratio'])} separated random pairs · "
                f"corr={gd['correlation']:.4f}<br>"
                f"median d<sub>graph</sub>/d<sub>euclid</sub> = "
                f"<b>{gd['median_ratio']:.4f}</b> · "
                f"p90 relative error={100 * gd['p90_relative_error']:.1f}%<br>"
                "<span style='font-size:10px'>Yellow: straight Euclidean "
                "segment. Pink: geometric shortest path through kNN edges. "
                "Edge lengths use physical node coordinates; anatomical "
                "inflation affects L, not this sampling diagnostic.</span>")

        anatomical = cur.get("anatomical_overlay")
        if anatomical is None:
            anatomical_html = ""
        else:
            before_med = float(np.median(anatomical["before_affinity"]))
            after_med = float(np.median(anatomical["after_affinity"]))
            attenuation_med = float(np.median(anatomical["attenuation"]))
            anatomical_html = (
                "<hr><b>anatomical soft boundary</b><br>"
                f"{anatomical['n_cross_total']:,} cross-region edges · "
                f"d² multiplied ×{args['cross_region_inflation']:g}<br>"
                f"median affinity {before_med:.2e} → "
                f"<b>{after_med:.2e}</b> "
                f"(×{attenuation_med:.2e})<br>"
                "<span style='font-size:10px'>Graph points are colored by atlas "
                "region. Optional overlays: magenta boundary nodes; "
                "Cyan: within-region edges. Orange: cross-region affinity "
                "before the atlas penalty. Purple: the same edges after the "
                "penalty; topology remains, but diffusion is weakened.</span>")

        info.setText(
            f"<b>graph</b> {g['method']}<br>"
            f"{com['N']:,} nodes · {g['n_edges']:,} edges · {K} modes loaded<br>"
            f"λ ∈ [{float(g['eigval'][0]):.4g}, {float(g['eigval'][-1]):.4g}], "
            f"λ_{n - 1} = {float(g['eigval'][n - 1]):.4g}<br>"
            "<hr>"
            f"<b>truncation N = {n}</b> &nbsp;(of {com['N']:,} possible)<br>"
            f"res_L &nbsp;‖L·f−L_N·f‖/‖L·f‖ = <b>{m['res_L']:.4f}</b><br>"
            f"res_f &nbsp;‖f−f_N‖_D/‖f‖_D &nbsp;= <b>{m['res_f']:.4f}</b><br>"
            f"cos &nbsp;&nbsp;∠(L·f, L_N·f) &nbsp;&nbsp;&nbsp;= {m['cos']:+.4f}"
            f"<br>"
            f"amp &nbsp;&nbsp;‖L_N·f‖/‖L·f‖ &nbsp;&nbsp;= {m['amp']:.3e}<br>"
            f"<span style='font-size:10px'>σ={st['sigma']:g} · "
            f"‖L·f‖={m['Lf_norm']:.4g} · L·f ∈ [{Lf_np.min():.3g}, "
            f"{Lf_np.max():.3g}] · each layer is autoscaled to its own "
            f"p{st['pct']:g}, so amp does not affect what you see.</span>"
            "<hr>"
            "<span style='font-size:10px'>"
            "<b>L·f</b> is the exact sparse Laplacian applied to the reference "
            "density — ground truth. <b>L_N·f</b> is the same field rebuilt from "
            "N eigenpairs; the <b>residual</b> layer is what those N modes miss, "
            "and it concentrates wherever f varies faster than the basis can "
            "represent (voxel-scale texture, thin boundaries).<br><br>"
            "<b>res_L pins near 1</b> on a raw template and that is not a bug: "
            "L_N·f equals L·f_N exactly, so res_L is ‖L(f−f_N)‖ — and the "
            "unresolved eigenvalues reach O(1/bandwidth²) while λ_N is still "
            "O(1), so the out-of-band remainder dominates. Raise σ to move the "
            "field into the resolved band and watch res_L fall. <b>res_f</b> and "
            "<b>cos</b> are the numbers that discriminate between N values.<br>"
            "<br>Compare the two graphs at the same N: sharper region interfaces "
            "push action to higher frequency, so the weighted graph pays a larger "
            "truncation cost — that is the price of the anatomical prior."
            "</span>"
            f"{distance_html}"
            f"{anatomical_html}"
            f"{stride_html}")
        print(f"[render] graph={g['method']} N={n} σ={st['sigma']:g}  "
              f"res_L={m['res_L']:.4f} res_f={m['res_f']:.4f} "
              f"cos={m['cos']:+.4f} amp={m['amp']:.2e}")

    # ---- controls --------------------------------------------------------
    method_choices = [GRAPH_LABELS[m] for m in args["graph_methods"]]
    label_to_method = {GRAPH_LABELS[m]: m for m in args["graph_methods"]}

    def sync_mode_choices(K: int):
        """Cap the modes dropdown at the loaded basis size K.

        A graph whose cache tops out below --num-modes loads fewer modes, and
        offering unreachable N values would silently clip inside `reconstruct`.
        Called from get_graph, which on the very first load runs before
        `controls` is bound — the NameError guard covers that, and the widget is
        built with the correct ladder anyway.
        """
        try:
            combo = controls.modes
        except NameError:
            return
        new = mode_ladder(K, args["mode_ladder"])
        if list(combo.choices) == new:
            return
        keep = st["modes"] if st["modes"] in new else max(new)
        combo.choices = new
        combo.value = keep
        st["modes"] = keep
        log.info(f"modes dropdown capped at {K} (loaded basis size): {new}")

    @magicgui(
        call_button="⟳  Re-render",
        auto_call=False,
        graph={"widget_type": "ComboBox", "choices": method_choices,
               "label": "graph"},
        modes={"widget_type": "ComboBox", "choices": ladder,
               "label": "modes N"},
        sigma={"widget_type": "FloatSpinBox", "min": 0.0, "max": 10.0,
               "step": 0.5, "label": "density σ (voxels)"},
        gamma={"widget_type": "FloatSpinBox", "min": 0.1, "max": 2.0,
               "step": 0.05, "label": "γ (colormap)"},
        pct={"widget_type": "FloatSpinBox", "min": 50.0, "max": 100.0,
             "step": 0.5, "label": "saturation pct"},
    )
    def controls(
        graph: str = GRAPH_LABELS[st["method"]],
        modes: int = initial_modes,
        sigma: float = st["sigma"],
        gamma: float = st["gamma"],
        pct: float = st["pct"],
    ):
        method = label_to_method[graph]
        changed_graph = (method != st["method"]) or cur["g"] is None
        changed_field = sigma != st["sigma"]
        st.update(method=method, modes=int(modes), sigma=float(sigma),
                  gamma=float(gamma), pct=float(pct))
        if changed_graph:
            cur["g"] = get_graph(method)
        render(changed_graph, changed_field)

    viewer.window.add_dock_widget(controls, name="render controls", area="right")
    viewer.window.add_dock_widget(info, name="reconstruction quality",
                                  area="right")

    # First render (also what populates the spectrum + info panel).
    cur["g"] = get_graph(st["method"])
    render(changed_graph=True, changed_field=True)

    tf = com["template_full"]
    print("\n" + "=" * 68)
    print("Laplacian ↔ eigenvector-reconstruction viewer ready.")
    print(f"  reference  : {tf.shape} = {tf.size:,} voxels at 25 µm")
    print(f"  nodes      : {com['N']:,}  (stride {stride}, "
          f"thresh {args['threshold']}) "
          f"= {100.0 * com['N'] / tf.size:.2f}% of the voxels")
    sc = com.get("stride_cost") or {}
    if sc:
        print(f"  stride cost: block-sampling the reference at stride {stride} "
              f"gives r={sc['r']:.4f}, rel-L2={sc['rel_l2']:.3f};")
        print(f"               {100 * sc['dropped']:.1f}% of full-res tissue "
              f"voxels land on a block with no node at all")
        print(f"               (sampled {sc['n_sampled']:,} tissue voxels)")
    print(f"  graphs     : {', '.join(args['graph_methods'])}")
    print(f"  mode ladder: {ladder}")
    print(f"  layers     : reference at 25 µm + at stride {stride}, "
          f"f, f_N, L·f, L_N·f, residual")
    print(f"               every node-lattice layer is drawn nearest-neighbour, "
          f"so a node")
    print(f"               reads as the {stride}×{stride}×{stride} block it "
          f"actually is")
    print(f"  ref render : '{args['template_rendering']}' in 3-D, "
          f"contrast {t_clim[0]:g}–{t_clim[1]:g}")
    if args["template_rendering"] in ("mip", "attenuated_mip"):
        print("               NOTE: a MIP takes the max along each ray, and a "
              "solid brain")
        print("               saturates nearly every ray — the layer then reads "
              "as a white")
        print("               fog filling its bounding box. Use "
              "--template-rendering iso.")
    print("  see stride : toggle 'template @ stride' against the 25 µm layer "
          "— same")
    print("               anatomy, one at 77 M voxels and one at "
          f"{com['N']:,} nodes.")
    print(f"               Easiest in 2-D (--ndisplay 2, now "
          f"{args['ndisplay']}-D); 3-D volume")
    print("               rendering blurs the lattice away.")
    print("  controls   : pick graph + modes N, press '⟳ Re-render'")
    print("               (switching graph loads/solves that eigenbasis; "
          "changing N only)")
    print("               (re-projects an already-loaded one, so it is instant)")
    print("=" * 68 + "\n")
    napari.run()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args["verbose"] else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    com = setup_common(args)
    if args["cache_report"]:
        report_cache(args, com)
        return
    if args["no_launch"]:
        report_residuals(args, com)
        return
    run_viewer(args, com)


if __name__ == "__main__":
    main()
