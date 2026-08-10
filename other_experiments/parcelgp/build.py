"""Build a ``ParcelField`` from the reference image and cache it.

    python -m other_experiments.parcelgp.build --reference-file .../reference_image.npy \
        --out .../parcels/full_k64.npz --n-parcels 64

Nothing here reads lipids or an atlas. ``--features`` selects the representation
and exists mainly so the validation script can build matched controls:

  ``full``    the multi-scale + Hessian-shape + depth stack (the proposal);
  ``simple``  single-scale intensity/gradient/local-std -- reproduces the
              representation the previous template clustering used, as the
              like-for-like baseline;
  ``spatial`` no appearance features at all -- pure geometry, the control that
              says how much of any measured gain is just "regions are compact".
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from .border import border_distance
from .features import FEATURE_VERSION, FeatureSpec, template_features
from .field import ParcelField, publish_file
from .parcellate import parcellate
from .volume import (LR_AXIS, coord_norm_from_reference, load_reference,
                     midline_mm, node_coords_mm, node_voxels, standardize,
                     stride_volume, tissue_mask)

log = logging.getLogger("parcelgp.build")

FEATURE_PRESETS = {
    "full": FeatureSpec(),
    "simple": FeatureSpec(smooth_scales_mm=(0.1,), gradient_scales_mm=(0.1,),
                          std_sizes_mm=(0.3,), hessian_scales_mm=(),
                          include_depth=False),
    "spatial": FeatureSpec(smooth_scales_mm=(), gradient_scales_mm=(),
                           std_sizes_mm=(), hessian_scales_mm=(),
                           include_depth=False),
}


def add_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--reference-file", required=True,
                   help="reference_image.npy (256-level intensity template).")
    p.add_argument("--out", required=True, help="Destination .npz for the parcel field.")
    p.add_argument("--stride", type=int, default=4,
                   help="Template subsampling stride. 4 => 0.1 mm node spacing.")
    p.add_argument("--threshold", type=float, default=5,
                   help="Tissue threshold on the reference intensity.")

    p.add_argument("--features", choices=sorted(FEATURE_PRESETS), default="full",
                   help="Appearance representation to cluster (see module docstring).")
    p.add_argument("--n-parcels", type=int, default=64)
    p.add_argument("--spatial-weight", type=float, default=1.0,
                   help="Weight on the z-scored coordinate block. 0 = pure "
                        "appearance clustering; larger = more compact parcels.")
    p.add_argument("--no-symmetric", dest="symmetric", action="store_false",
                   help="Do NOT fold the left-right axis before clustering "
                        "(default is to fold, giving bilaterally symmetric parcels).")
    p.add_argument("--no-contiguity", dest="contiguity", action="store_false",
                   help="Skip the connected-component repair (leaves speckle).")
    p.add_argument("--min-component-frac", type=float, default=0.15,
                   help="A parcel's connected component survives if it holds at "
                        "least this fraction of the parcel.")
    p.add_argument("--min-component-voxels", type=int, default=20)
    p.add_argument("--n-membership", type=int, default=4,
                   help="Top-J parcels kept in each node's soft membership vector.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Membership softmax temperature, in units of the median "
                        "distance to the assigned centroid.")
    p.add_argument("--fit-subsample", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--normalize-blocks", dest="normalize_blocks",
                   action="store_true",
                   help="Weight each descriptor BLOCK equally in the k-means "
                        "distance instead of each feature. Off by default = "
                        "current behaviour, where a block's influence is just "
                        "how many numbers it emits (hessian 37.5%%, depth 6.3%%). "
                        "On makes it 20%% each. Neither is known to be better — "
                        "build both and compare with parcelgp.validate.")
    p.add_argument("--include-background-border", action="store_true",
                   help="Count the tissue surface as a parcel border too.")
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if --out already holds a matching field.")
    p.set_defaults(symmetric=True, contiguity=True)
    return p


#: Every argument that changes what the parcellation IS. A cached field may only
#: be reused when all of these match. Kept as an explicit list rather than "all
#: args" so genuinely cosmetic flags (--force) don't force pointless rebuilds.
BUILD_KEYS = (
    "reference_file", "stride", "threshold", "features", "n_parcels",
    "spatial_weight", "symmetric", "contiguity", "min_component_frac",
    "min_component_voxels", "n_membership", "temperature", "fit_subsample",
    "seed", "include_background_border", "normalize_blocks",
    "feature_version",
)


def check_cached(out_path, args) -> tuple[bool, list[str]]:
    """Can the field at ``out_path`` be reused for this build?

    Returns ``(reusable, differences)``. Filenames are a hopeless cache key --
    they only encode the parameters someone remembered to put in them, so a
    changed --stride silently reuses the wrong field. The recorded build
    parameters are the authority instead.
    """
    meta_path = Path(str(out_path) + ".meta.json")
    if not Path(out_path).exists():
        return False, ["no cached field"]
    if not meta_path.exists():
        return False, ["cached field has no .meta.json — cannot verify it"]
    meta = json.loads(meta_path.read_text())
    diffs = []
    for k in BUILD_KEYS:
        want = FEATURE_VERSION if k == "feature_version" else args.get(k)
        got = meta.get(k, "<absent>")
        if k == "n_parcels":            # meta records the requested count separately
            got = meta.get("n_parcels_requested", got)
        if isinstance(want, float) or isinstance(got, float):
            same = (got != "<absent>" and abs(float(want) - float(got)) < 1e-9)
        else:
            same = (str(want) == str(got))
        if not same:
            diffs.append(f"{k}: cached={got!r} requested={want!r}")
    return (not diffs), diffs


def build(args: dict) -> ParcelField:
    t0 = time.time()
    ref = load_reference(args["reference_file"])
    coord_mean, coord_std = coord_norm_from_reference(ref)
    mid_mm = midline_mm(ref)
    midline_std = float((mid_mm - coord_mean[LR_AXIS]) / coord_std)

    sub, voxel_scale_mm = stride_volume(ref, int(args["stride"]))
    mask = tissue_mask(sub, args["threshold"])
    node_vox = node_voxels(mask)
    coords_std = standardize(node_coords_mm(node_vox, voxel_scale_mm),
                             coord_mean, coord_std)
    log.info("reference %s -> strided %s, %d tissue nodes (%.3f mm spacing), "
             "midline at %.3f (standardized)",
             ref.shape, sub.shape, node_vox.shape[0], voxel_scale_mm, midline_std)

    spec = FEATURE_PRESETS[args["features"]]
    if args["features"] == "spatial":
        feats = np.zeros((node_vox.shape[0], 0), dtype=np.float32)
        names: list[str] = []
    else:
        feats, names = template_features(sub, mask, node_vox, voxel_scale_mm, spec)

    if feats.shape[1] == 0 and args["spatial_weight"] <= 0:
        raise ValueError("--features spatial requires --spatial-weight > 0 "
                         "(there would be nothing to cluster on)")

    parc, pstats = parcellate(
        feats=feats, coords_std=coords_std, node_vox=node_vox, shape=sub.shape,
        n_parcels=int(args["n_parcels"]),
        spatial_weight=float(args["spatial_weight"]),
        symmetric=bool(args["symmetric"]), midline_std=midline_std,
        contiguity=bool(args["contiguity"]),
        min_component_frac=float(args["min_component_frac"]),
        min_component_voxels=int(args["min_component_voxels"]),
        n_membership=int(args["n_membership"]),
        temperature=float(args["temperature"]),
        fit_subsample=int(args["fit_subsample"]), seed=int(args["seed"]),
        feature_names=names, normalize_blocks=bool(args["normalize_blocks"]),
    )

    bf = border_distance(
        parc.labels, node_vox, sub.shape, voxel_scale_mm,
        n_parcels=parc.n_parcels,
        include_background=bool(args["include_background_border"]),
    )

    meta = {
        # Every BUILD_KEYS entry must be recorded here, or check_cached cannot
        # tell a stale field from a matching one.
        "reference_file": str(args["reference_file"]),
        "stride": int(args["stride"]), "threshold": float(args["threshold"]),
        "features": args["features"], "feature_names": names,
        "feature_version": FEATURE_VERSION,
        "contiguity": bool(args["contiguity"]),
        "min_component_frac": float(args["min_component_frac"]),
        "min_component_voxels": int(args["min_component_voxels"]),
        "n_membership": int(args["n_membership"]),
        "temperature": float(args["temperature"]),
        "fit_subsample": int(args["fit_subsample"]),
        "include_background_border": bool(args["include_background_border"]),
        "normalize_blocks": bool(args["normalize_blocks"]),
        "voxel_scale_mm": float(voxel_scale_mm),
        "coord_mean": [float(v) for v in coord_mean],
        "coord_std": float(coord_std), "midline_std": midline_std,
        "build_seconds": None, **pstats, **bf.stats,
    }
    field = ParcelField(
        node_coords_std=coords_std, labels=parc.labels,
        mem_idx=parc.mem_idx, mem_val=parc.mem_val,
        d_border_z=bf.d_border_z, d_border_rel=bf.d_border_rel,
        nearest_other=bf.nearest_other, n_parcels=parc.n_parcels,
        node_vox=node_vox, volume_shape=sub.shape, meta=meta,
    )
    field.meta["build_seconds"] = round(time.time() - t0, 1)
    return field


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = add_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    args = vars(p.parse_args(argv))

    reusable, diffs = check_cached(args["out"], args)
    if reusable and not args["force"]:
        log.info("reusing cached field %s (build parameters match)", args["out"])
        return
    if diffs and diffs[0] != "no cached field" and not args["force"]:
        raise SystemExit(
            f"\n{args['out']} exists but was built with DIFFERENT parameters:\n  "
            + "\n  ".join(diffs)
            + "\n\nReusing it would silently train against the wrong parcellation. "
              "Either point --out at a new path (recommended — keep both) or pass "
              "--force to overwrite.\n")

    field = build(args)
    out = field.save(args["out"])
    # Same atomic dance for the sidecar: check_cached reads it, and a torn
    # write there would make a good field look unverifiable.
    meta_path = Path(str(out) + ".meta.json")
    with tempfile.TemporaryDirectory(prefix="parcelgp-meta-") as td:
        local = Path(td) / "meta.json"
        local.write_text(json.dumps(field.meta, indent=2))
        publish_file(local, meta_path)
    print(json.dumps(field.meta, indent=2))


if __name__ == "__main__":
    main()
