#!/usr/bin/env python3
"""Check which eigenvector caches are present vs missing.

Given lists of arguments (knn_k, threshold, num_modes, stride, nlist, and the
KNN method faiss / faiss_atlas_weighted) this builds the eigvec cache key for
every combination -- using the SAME make_key helpers the pipeline uses, so the
keys can't drift -- and reports which `.eigpairs.npz` files exist under
`<eigenvector-dir>/eigvecs/` and which are missing.

Two modes:
  * grid (default): cartesian product of the value lists you pass -> PRESENT /
    MISSING per combination.
  * --list: just inventory whatever is already in the cache dir (parsed back
    from the filenames), no grid needed.

Examples
--------
# Does the K=15, thr in {5,40,50}, 2300-mode, stride-4 set exist for plain
# faiss AND atlas-weighted x{10,50}?  (--nlist sqrt is resolved per stride/thr)
python manifold/check_eigencache.py \
    --eigenvector-dir /s3/mlibra/mlibra-data/artiom/eigenvectors \
    --knn-k 15 --threshold 5 40 50 --num-modes 2300 --stride 4 \
    --nlist sqrt --method faiss faiss_atlas_weighted --inflation 10 50

# Just show me everything that's cached:
python manifold/check_eigencache.py \
    --eigenvector-dir /s3/mlibra/mlibra-data/artiom/eigenvectors --list

Notes
-----
* nlist is keyed only when != 1 (recall ~1.0 at nlist=1, so the pipeline keeps
  those caches shared). Pass --nlist 1 or --nlist sqrt; 'sqrt' is resolved to
  round(sqrt(N)) per (stride, threshold) from --reference-file, matching the
  pipeline. Use --list to see what's actually on disk.
* inflation only applies to faiss_atlas_weighted; it's ignored for plain faiss.
"""
from __future__ import annotations

# --- repo path bootstrap (this file moved out of maldi/) ---
import sys as _sys
from pathlib import Path as _Path
_REPO_ = _Path(__file__).resolve().parents[1]
for _p in (str(_REPO_), str(_REPO_ / "maldi"),):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import argparse
import itertools
import os
import re
import sys
from pathlib import Path

# Make the repo root importable so we reuse the pipeline's key builders verbatim.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from manifold_gp.utils.nearest_neighbors import (  # noqa: E402
    make_key as make_graph_key, resolve_nlist,
)
from manifold_gp.utils.compute_eigenvectors import (  # noqa: E402
    make_key as make_eig_key, resolve_modes_key,
)


# ---------------------------------------------------------------------------
# nlist='sqrt' resolution -- N depends on (stride, threshold), resolved the same
# way slepc_eigensolve.node_count does so the keys match. Memoized per pair.
# ---------------------------------------------------------------------------
_N_CACHE: dict = {}


def node_count(reference_file, annotations_file, stride, threshold) -> int:
    import numpy as np
    from maldi.utils import (crop_or_stride_volume,
                             reference_ccf_from_subvolume)
    reference_image = np.load(reference_file)
    annotation_volume = np.load(annotations_file) if annotations_file else None
    sub_volume, _, voxel_offset, voxel_scale_mm = crop_or_stride_volume(
        reference_image, annotation_volume, stride)
    mm_coords = reference_ccf_from_subvolume(
        sub_volume, voxel_offset, voxel_scale_mm, threshold)
    return int(mm_coords.shape[0])


def resolve_nlist_spec(spec, *, stride, threshold, reference_file,
                       annotations_file) -> int:
    """'1' -> 1, 'sqrt' -> round(sqrt(N(stride,threshold))). Memoized."""
    if str(spec).strip().lower() != "sqrt":
        return resolve_nlist(spec, 0)
    if not reference_file:
        sys.exit("ERROR: --nlist sqrt needs --reference-file to compute N.")
    ck = (stride, threshold)
    if ck not in _N_CACHE:
        _N_CACHE[ck] = node_count(reference_file, annotations_file,
                                  stride, threshold)
    return resolve_nlist("sqrt", _N_CACHE[ck])


# ---------------------------------------------------------------------------
# Key construction -- mirrors slepc_eigensolve.resolve_graph_keys + the eigvec
# key in __main__, so a hit here means a hit in the real pipeline.
# ---------------------------------------------------------------------------
def eig_graph_key(*, template, stride, threshold, knn_k, method, nlist,
                  inflation):
    parts = {
        "template": template, "stride": stride, "thresh": threshold,
        "method": method, "k": knn_k, "bbox": None,
    }
    if nlist != 1:
        parts["nlist"] = nlist
    if method == "anatomical_atlas":
        parts["atlas"] = "annotation_coarse_d4"
        parts["conn"] = 3
    if method == "faiss_atlas_weighted":
        parts["weighting"] = f"atlas_x{inflation:g}"
    return make_graph_key(parts)


def eigvec_key(*, norm, bandwidth, num_modes, **graph_kwargs):
    graph = eig_graph_key(**graph_kwargs)
    return make_eig_key({
        "graph": graph, "norm": norm, "bw": bandwidth, "modes": num_modes,
    })


def cache_paths(eigvec_dir: Path, key: str):
    return (eigvec_dir / f"{key}.eigpairs.npz",
            eigvec_dir / f"{key}.eigpairs.meta.json")


# ---------------------------------------------------------------------------
# Inventory (--list): parse fields back out of existing filenames.
# ---------------------------------------------------------------------------
_FIELD_RE = {
    "stride": r"stride=(\d+)",
    "thresh": r"thresh=([\d.eE+-]+)",
    "knn_k": r"(?:^|_)k=(\d+)",
    "method": r"method=(faiss_atlas_weighted|anatomical_atlas|faiss)",
    "nlist": r"nlist=(\d+)",
    "weighting": r"weighting=atlas_x([\d.]+)",
    "modes": r"modes=(\d+)",
    "norm": r"norm=(\w+)",
    "bw": r"bw=([\d.eE+-]+)",
}


def parse_key(key: str) -> dict:
    out = {}
    for name, pat in _FIELD_RE.items():
        m = re.search(pat, key)
        out[name] = m.group(1) if m else None
    return out


def inventory(eigvec_dir: Path):
    rows = []
    for npz in sorted(eigvec_dir.glob("*.eigpairs.npz")):
        key = npz.name[: -len(".eigpairs.npz")]
        f = parse_key(key)
        f["size_mb"] = npz.stat().st_size / 1e6
        rows.append(f)
    return rows


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eigenvector-dir",
                   default=os.environ.get("EIGENVECTOR_DIR",
                                          "/s3/mlibra/mlibra-data/artiom/eigenvectors"),
                   help="Cache root (eigvecs live in <dir>/eigvecs). "
                        "Defaults to $EIGENVECTOR_DIR.")
    p.add_argument("--list", action="store_true",
                   help="Inventory the cache dir instead of checking a grid.")
    p.add_argument("--allow-larger", action="store_true",
                   help="Count a combo as satisfiable (REUSABLE) when a cache "
                        "with the same graph but MORE modes exists -- matches "
                        "the lgp experiments' allow_larger_modes reuse.")

    # Grid axes -- each accepts multiple values; the script takes the product.
    p.add_argument("--knn-k", type=int, nargs="+", default=[15])
    p.add_argument("--threshold", type=float, nargs="+", default=[5.0])
    p.add_argument("--num-modes", type=int, nargs="+", default=[2300])
    p.add_argument("--stride", type=int, nargs="+", default=[4])
    p.add_argument("--nlist", nargs="+", default=["1"], choices=["1", "sqrt"],
                   help="IVF nlist: '1' (exact flat, not keyed) or 'sqrt' "
                        "(round(sqrt(N)), keyed). 'sqrt' needs --reference-file.")
    p.add_argument("--method", nargs="+", default=["faiss"],
                   choices=["faiss", "faiss_atlas_weighted", "anatomical_atlas"])
    p.add_argument("--inflation", type=float, nargs="+", default=[10.0],
                   help="cross_region_inflation, faiss_atlas_weighted only.")

    # Fixed (single-value) axes -- rarely swept, but overridable.
    p.add_argument("--norm", default="randomwalk",
                   choices=["randomwalk", "symmetric"])
    p.add_argument("--bandwidth", type=float, default=1.0)
    p.add_argument("--template", default="reference")

    # Only needed to resolve --nlist sqrt (N = nodes for stride+threshold).
    p.add_argument("--reference-file",
                   default=os.environ.get("REFERENCE_FILE",
                                          "/s3/mlibra/mlibra-data/reference_image.npy"))
    p.add_argument("--annotations-file",
                   default=os.environ.get("ANNOTATIONS_FILE",
                                          "/s3/mlibra/mlibra-data/level_15annot.npy"))
    args = p.parse_args()

    eigvec_dir = Path(args.eigenvector_dir) / "eigvecs"
    if not eigvec_dir.is_dir():
        sys.exit(f"ERROR: no eigvecs dir at {eigvec_dir}")

    # ---- inventory mode -------------------------------------------------
    if args.list:
        rows = inventory(eigvec_dir)
        if not rows:
            print(f"(no eigvec caches found in {eigvec_dir})")
            return
        hdr = ("method", "stride", "thresh", "knn_k", "nlist", "weighting",
               "modes", "norm", "bw", "size_mb")
        print(f"{len(rows)} eigvec cache(s) in {eigvec_dir}:\n")
        widths = {h: max(len(h), *(len(str(r.get(h) or "-")) for r in rows))
                  for h in hdr}
        line = "  ".join(f"{h:<{widths[h]}}" for h in hdr)
        print(line)
        print("  ".join("-" * widths[h] for h in hdr))
        for r in sorted(rows, key=lambda r: (r.get("method") or "",
                                             r.get("stride") or "",
                                             r.get("modes") or "")):
            vals = []
            for h in hdr:
                v = r.get(h)
                v = "-" if v is None else (f"{v:.1f}" if h == "size_mb" else str(v))
                vals.append(f"{v:<{widths[h]}}")
            print("  ".join(vals))
        return

    # ---- grid mode ------------------------------------------------------
    # Build the combination list. inflation only multiplies the weighted method;
    # it's a no-op (and would create bogus duplicates) for the others.
    combos = []
    seen = set()
    for method, k, thr, modes, stride, nlist_spec, infl in itertools.product(
            args.method, args.knn_k, args.threshold, args.num_modes,
            args.stride, args.nlist, args.inflation):
        if method != "faiss_atlas_weighted":
            infl = None  # not part of the key; collapse the inflation axis
        nlist = resolve_nlist_spec(
            nlist_spec, stride=stride, threshold=thr,
            reference_file=args.reference_file,
            annotations_file=args.annotations_file)
        sig = (method, k, thr, modes, stride, nlist, infl)
        if sig in seen:
            continue
        seen.add(sig)
        combos.append(dict(method=method, knn_k=k, threshold=thr,
                           num_modes=modes, stride=stride, nlist=nlist,
                           nlist_spec=nlist_spec, inflation=infl))

    present, reusable, missing = [], [], []
    for c in combos:
        key = eigvec_key(
            template=args.template, norm=args.norm, bandwidth=args.bandwidth,
            num_modes=c["num_modes"], stride=c["stride"], threshold=c["threshold"],
            knn_k=c["knn_k"], method=c["method"], nlist=c["nlist"],
            inflation=(c["inflation"] if c["inflation"] is not None else 0.0))
        npz, meta = cache_paths(eigvec_dir, key)
        if npz.exists() and meta.exists():
            present.append((c, key, None))
        elif args.allow_larger:
            # Same graph/norm/bw, more modes? resolve_modes_key returns that
            # sibling's key (else the original), exactly as the experiments reuse.
            alt = resolve_modes_key(eigvec_dir, key, c["num_modes"])
            if alt != key:
                reusable.append((c, key, alt))
            else:
                missing.append((c, key, None))
        else:
            missing.append((c, key, None))

    def label(c):
        m = c["method"]
        if m == "faiss_atlas_weighted":
            m = f"faiss_w x{c['inflation']:g}"
        bits = [f"method={m}", f"k={c['knn_k']}", f"thr={c['threshold']:g}",
                f"modes={c['num_modes']}", f"stride={c['stride']}"]
        if c["nlist"] != 1:
            bits.append(f"nlist={c['nlist_spec']}({c['nlist']})")
        return "  ".join(bits)

    print(f"Eigvec cache dir: {eigvec_dir}")
    print(f"Checked {len(combos)} combination(s) "
          f"(norm={args.norm}, bw={args.bandwidth:g}, template={args.template})\n")

    def alt_modes(alt_key):
        m = re.search(r"_modes=(\d+)_norm=", alt_key)
        return m.group(1) if m else "?"

    print(f"PRESENT ({len(present)}):")
    for c, key, _ in present:
        print(f"  [x] {label(c)}")
    if not present:
        print("  (none)")

    if args.allow_larger:
        print(f"\nREUSABLE via larger-modes cache ({len(reusable)}):")
        for c, key, alt in reusable:
            print(f"  [~] {label(c)}  <- modes={alt_modes(alt)} cache")
        if not reusable:
            print("  (none)")

    print(f"\nMISSING ({len(missing)}):")
    for c, key, _ in missing:
        print(f"  [ ] {label(c)}")
    if not missing:
        print("  (none)")

    # Non-zero exit only if something is truly unobtainable. With --allow-larger,
    # a REUSABLE combo counts as obtainable (the experiments will reuse it).
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
