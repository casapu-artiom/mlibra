"""Does the reference-only parcellation actually separate the lipids?

This is the gate. The parcellation is built with no access to lipid data, so the
honest question is whether the lipid data agrees with it -- and there is prior
evidence that a *bad* partition (the coarse Allen atlas) fails all three of these
checks: cross-boundary lipid change only 1.22x the within-region change, residual
variance essentially flat with distance-to-boundary, and 18% variance explained.
Those numbers are the bar to clear.

Three checks, all on measured voxels only, all comparable across parcellations:

  **1. boundary_contrast** -- take pairs of *adjacent measured voxels* inside one
  section and compare the mean absolute lipid difference for pairs that cross a
  parcel border against pairs that do not. A partition that captures real lipid
  structure makes chemistry change faster across its borders. Both pair types are
  one pixel apart, so distance is controlled by construction. Reported with a
  bootstrap over sections (pairs within a section are not independent).

  **2. border_trend** -- restricted to *within-parcel* pairs, bin by how deep in
  the parcel the pair sits and track the mean absolute difference. This is the
  check that decides whether a distance-to-border feature is worth building at
  all: it pays off only if lipids vary faster near a border and settle down in a
  parcel's interior, i.e. a decreasing curve. A flat curve means the border
  distance carries nothing, whatever the boundary contrast says.

  **3. parcel_ev** -- variance explained by the parcel-mean predictor, averaged
  over lipids. Held out by SECTION by default, not by voxel: the point of the
  model is to fill the gaps between measured sections, so a parcel that only
  looks good when it can see voxels from the same section is not evidence.

Pass several ``--field`` arguments to score parcellations side by side; add
``--atlas-file`` to include the Allen annotation volume as the reference row.
Build a ``--features spatial`` field at matched K for the "compactness alone"
control -- any gain a parcellation shows over that is what the template's
appearance is actually contributing.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .field import ParcelField
from .volume import (coord_norm_from_reference, load_reference, standardize,
                     stride_volume)

log = logging.getLogger("parcelgp.validate")

#: Registered CCF coordinates -- (AP, DV, LR) -- used for the parcel lookup only.
COORD_COLS = ("xccf", "yccf", "zccf")
#: Native acquisition pixel indices within a section, and the section identifier.
PIXEL_COLS = ("x", "y")
SECTION_COL = "SampleSection"
SAMPLE_COL = "Sample"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_filters(slices_file, which="train"):
    """DNF filter list for pyarrow, from a splits json ('train'|'test'|'ignore')."""
    if slices_file is None:
        return None
    spec = json.loads(Path(slices_file).read_text())[which]
    return [[tuple(clause)] for clause in spec]


def _and_sections(filters, sections):
    """AND a ``SampleSection in {...}`` clause into every DNF group."""
    clause = (SECTION_COL, "in", set(sections))
    if not filters:
        return [[clause]]
    return [list(group) + [clause] for group in filters]


def load_maldi(maldi_file, lipid_names, filters, coord_mean, coord_std,
               log_transform=False, max_sections=0, max_voxels=0, seed=0):
    """Load measured voxels, keeping WHOLE sections.

    Subsampling has to be by section, never by voxel: the adjacency check below
    needs intact pixel neighbourhoods, and dropping a random 90% of voxels would
    silently destroy them.

    Returns ``(coords_std (M,3), Y (M,L) z-scored, section_id (M,),
    pix (M,2) int32, sample_id (M,))``.
    """
    meta = pd.read_parquet(
        maldi_file, columns=[SECTION_COL, SAMPLE_COL], filters=filters)
    counts = meta[SECTION_COL].value_counts()
    rng = np.random.default_rng(seed)
    sections = rng.permutation(counts.index.to_numpy())
    if max_sections and max_sections < sections.size:
        sections = sections[:max_sections]
    if max_voxels:
        keep_n = np.cumsum(counts.loc[sections].to_numpy()) <= max_voxels
        if keep_n.any():
            sections = sections[keep_n]
    log.info("keeping %d/%d sections (%d voxels)", sections.size, counts.size,
             int(counts.loc[sections].sum()))
    del meta

    cols = list(COORD_COLS) + list(PIXEL_COLS) + [SECTION_COL, SAMPLE_COL]
    sec_filters = _and_sections(filters, sections)
    df = pd.read_parquet(maldi_file, columns=cols, filters=sec_filters)
    Y = pd.read_parquet(maldi_file, columns=list(lipid_names),
                        filters=sec_filters).values.astype(np.float32)

    xyz = df[list(COORD_COLS)].values.astype(np.float32)
    pix = df[list(PIXEL_COLS)].values.astype(np.int64)
    keep = np.isfinite(Y).all(1) & np.isfinite(xyz).all(1)
    xyz, Y, pix, df = xyz[keep], Y[keep], pix[keep], df[keep]

    if log_transform:
        Y = np.log1p(np.clip(Y, 0, None))
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)

    _, section_id = np.unique(df[SECTION_COL].to_numpy(), return_inverse=True)
    _, sample_id = np.unique(df[SAMPLE_COL].to_numpy(), return_inverse=True)
    coords_std = standardize(xyz, coord_mean, coord_std)
    log.info("MALDI: %d voxels x %d lipids across %d sections / %d samples",
             xyz.shape[0], Y.shape[1], section_id.max() + 1, sample_id.max() + 1)
    return (coords_std, Y, section_id.astype(np.int32), pix.astype(np.int32),
            sample_id.astype(np.int32))


def pool_to_nodes(coords_std, Y, field, min_count=3):
    """Average the measured voxels onto the template nodes.

    Checks 1 and 2 must NOT be run on raw voxels. Two reasons, both fatal:

      * the registration lookup lands a voxel on its nearest node with a median
        error of ~47 um, which is *larger* than the ~25 um acquisition pitch — so
        for exactly the adjacent pairs that straddle a border, which side each
        voxel falls on is close to a coin flip, and the cross/within split is
        contaminated where it matters most;
      * a single voxel carries ~19% pure measurement noise (measured: adjacent-pair
        |dY| is 0.435x a random pair's), which dilutes every ratio toward 1.

    Pooling onto the 0.1 mm node grid fixes both: a node's parcel label is exact
    by construction, and averaging many voxels (across samples and sections)
    suppresses the noise. This is the same remedy as the 0.2 mm grid-cell pooling
    used in the earlier distance analysis.

    Returns ``(node_ids (n,), Ybar (n, L) re-z-scored, counts (n,))`` for nodes
    with at least ``min_count`` measured voxels.
    """
    idx = field.node_index(coords_std)
    N, L = field.node_coords.shape[0], Y.shape[1]
    counts = np.bincount(idx, minlength=N)
    sums = np.zeros((N, L), dtype=np.float64)
    np.add.at(sums, idx, Y.astype(np.float64))
    keep = np.flatnonzero(counts >= int(min_count))
    Ybar = (sums[keep] / counts[keep, None]).astype(np.float32)
    Ybar = (Ybar - Ybar.mean(0)) / (Ybar.std(0) + 1e-8)
    log.info("pooled %d voxels -> %d nodes with >=%d measurements "
             "(median %d voxels/node)", Y.shape[0], keep.size, min_count,
             int(np.median(counts[keep])))
    return keep, Ybar, counts[keep]


def node_adjacent_pairs(node_vox, node_ids, include_ap=False, n_blocks=20):
    """Pairs of measured template nodes that are neighbours on the strided grid.

    ``include_ap=False`` (default) uses only the two in-plane axes. A step along
    the anterior-posterior axis lands on a node filled by a *different* coronal
    section, so those pairs carry between-section batch effects on top of any
    spatial difference; the in-plane axes stay within one section.

    Pairs are tagged with an anterior-posterior block id, which is the resampling
    unit for the block bootstrap (neighbouring pairs are spatially correlated, so
    resampling pairs would badly understate the interval).
    """
    BIG = np.int64(1 << 12)
    vox = node_vox[node_ids].astype(np.int64)
    keys = (vox[:, 0] * BIG + vox[:, 1]) * BIG + vox[:, 2]
    order = np.argsort(keys)
    skeys = keys[order]
    axes = ((0, 0, 1), (0, 1, 0)) + (((1, 0, 0),) if include_ap else ())

    pi, pj = [], []
    for d in axes:
        q = ((vox[:, 0] + d[0]) * BIG + vox[:, 1] + d[1]) * BIG + vox[:, 2] + d[2]
        pos = np.clip(np.searchsorted(skeys, q), 0, skeys.size - 1)
        ok = skeys[pos] == q
        if not ok.any():
            continue
        i = np.flatnonzero(ok)
        pi.append(i)
        pj.append(order[pos[i]])
    if not pi:
        raise RuntimeError("no adjacent node pairs — is the parcel field's stride "
                           "the same as the one used to build it?")
    pi, pj = np.concatenate(pi), np.concatenate(pj)
    ap = vox[pi, 0]
    edges = np.quantile(ap, np.linspace(0, 1, n_blocks + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    blocks = np.clip(np.digitize(ap, edges[1:-1]), 0, n_blocks - 1).astype(np.int32)
    log.info("node pairs: %d (%s axes) across %d AP blocks", pi.size,
             "3" if include_ap else "2 in-plane", np.unique(blocks).size)
    return pi, pj, blocks


def adjacent_pairs(pix, section_id, max_pairs=0, seed=0):
    """Pairs of measured voxels that are 4-neighbours on the acquisition grid.

    Adjacency is taken in the NATIVE section pixel grid, not in registered CCF
    space: registration warps the sections, so a fixed CCF radius would pick up a
    different number of neighbours in different parts of the brain and bias the
    comparison. On the native grid, "adjacent" means exactly one pixel apart
    everywhere. Returns ``(pair_i, pair_j, pair_section)``.
    """
    BIG = np.int64(1 << 20)
    pi, pj, ps = [], [], []
    for s in np.unique(section_id):
        m = np.flatnonzero(section_id == s)
        if m.size < 4:
            continue
        x, y = pix[m, 0].astype(np.int64), pix[m, 1].astype(np.int64)
        keys = x * BIG + y
        order = np.argsort(keys)
        skeys = keys[order]
        for dx, dy in ((1, 0), (0, 1)):
            q = (x + dx) * BIG + (y + dy)
            pos = np.clip(np.searchsorted(skeys, q), 0, skeys.size - 1)
            ok = skeys[pos] == q
            if not ok.any():
                continue
            i = np.flatnonzero(ok)
            pi.append(m[i])
            pj.append(m[order[pos[i]]])
            ps.append(np.full(i.size, s, dtype=np.int32))
    if not pi:
        raise RuntimeError("no adjacent pairs found — check PIXEL_COLS/SECTION_COL")
    pi, pj, ps = np.concatenate(pi), np.concatenate(pj), np.concatenate(ps)
    if 0 < max_pairs < pi.size:
        sel = np.random.default_rng(seed).choice(pi.size, max_pairs, replace=False)
        sel.sort()
        pi, pj, ps = pi[sel], pj[sel], ps[sel]
    log.info("adjacent pairs: %d across %d sections", pi.size, np.unique(ps).size)
    return pi, pj, ps


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def _grouped_abs_diff(Y, pi, pj, group, n_groups, lipid_chunk=16):
    """Sum of |Y[i]-Y[j]| per (group, lipid) plus per-group pair counts.

    Accumulated in lipid chunks so the (pairs x lipids) difference matrix is never
    materialized in full.
    """
    L = Y.shape[1]
    sums = np.zeros((n_groups, L), dtype=np.float64)
    counts = np.bincount(group, minlength=n_groups).astype(np.float64)
    for s in range(0, L, lipid_chunk):
        d = np.abs(Y[pi, s:s + lipid_chunk] - Y[pj, s:s + lipid_chunk])
        for c in range(d.shape[1]):
            sums[:, s + c] = np.bincount(group, weights=d[:, c].astype(np.float64),
                                         minlength=n_groups)
        del d
    return sums, counts


def boundary_contrast(Y, pi, pj, ps, labels, n_boot=200, seed=0):
    """Check 1: mean |dlipid| across a parcel border / within a parcel.

    ``ps`` is the resampling block of each pair (an anterior-posterior band for
    node pairs, a section for pixel pairs). The bootstrap resamples BLOCKS, not
    pairs: adjacent pairs overlap in their endpoints and are spatially
    correlated, so a pair-level bootstrap would report an interval several times
    too narrow.
    """
    cross = (labels[pi] != labels[pj])
    sections = np.unique(ps)
    sec_of = np.searchsorted(sections, ps)
    S = sections.size

    # group = section * 2 + is_cross
    g = sec_of * 2 + cross.astype(np.int64)
    sums, counts = _grouped_abs_diff(Y, pi, pj, g, 2 * S)
    sum_within, sum_cross = sums[0::2], sums[1::2]           # (S, L)
    n_within, n_cross = counts[0::2], counts[1::2]           # (S,)

    def _ratio(sel):
        w = sum_within[sel].sum(0) / max(n_within[sel].sum(), 1)
        c = sum_cross[sel].sum(0) / max(n_cross[sel].sum(), 1)
        return c / np.maximum(w, 1e-12)                      # (L,)

    point = _ratio(np.arange(S))
    rng = np.random.default_rng(seed)
    boot = np.stack([_ratio(rng.integers(0, S, S)) for _ in range(n_boot)])
    pooled = boot.mean(1)                                    # mean over lipids per draw

    return {
        "cross_pair_frac": float(cross.mean()),
        "contrast_mean": float(point.mean()),
        "contrast_median": float(np.median(point)),
        "contrast_ci95": [float(np.percentile(pooled, 2.5)),
                          float(np.percentile(pooled, 97.5))],
        "lipids_above_1": float((point > 1.0).mean()),
        "per_lipid": point.astype(np.float32),
    }


def border_trend(Y, pi, pj, labels, d_rel, n_bins=10):
    """Check 2: within-parcel |dlipid| as a function of depth into the parcel.

    Values are normalized by each lipid's own overall within-parcel mean, so the
    returned curve is a relative one and lipids with large dynamic range do not
    dominate. A decreasing curve is the signal; flat means the border distance is
    uninformative.
    """
    within = np.flatnonzero(labels[pi] == labels[pj])
    if within.size < n_bins * 50:
        raise RuntimeError("too few within-parcel pairs for the border trend")
    wi, wj = pi[within], pj[within]
    depth = np.minimum(d_rel[wi], d_rel[wj])

    edges = np.quantile(depth, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.clip(np.digitize(depth, edges[1:-1]), 0, n_bins - 1)

    sums, counts = _grouped_abs_diff(Y, wi, wj, b, n_bins)
    per_bin = sums / np.maximum(counts, 1)[:, None]           # (bins, L)
    overall = sums.sum(0) / max(counts.sum(), 1)              # (L,)
    curve = per_bin / np.maximum(overall, 1e-12)[None, :]     # relative

    mean_curve = curve.mean(1)
    # Spearman rho between bin index and the relative curve, per lipid.
    rank_bins = np.arange(n_bins, dtype=np.float64)
    rb = (rank_bins - rank_bins.mean()) / rank_bins.std()
    cz = (curve - curve.mean(0)) / (curve.std(0) + 1e-12)
    rho = (rb[:, None] * cz).mean(0)

    return {
        "n_within_pairs": int(within.size),
        "bin_depth_edges": [float(v) for v in np.quantile(
            depth, np.linspace(0, 1, n_bins + 1))],
        "relative_curve": [float(v) for v in mean_curve],
        "near_over_far": float(mean_curve[0] / max(mean_curve[-1], 1e-12)),
        "trend_rho_mean": float(rho.mean()),
        "lipids_decreasing": float((rho < 0).mean()),
    }


def parcel_ev(Y, labels, group_id, n_parcels, split="section", n_folds=5, seed=0):
    """Check 3: variance explained by the parcel mean, held out.

    ``group_id`` is the unit that gets held out -- whole sections (the
    imputation-relevant split, since the model exists to fill the gaps between
    measured sections) or whole samples. ``split='voxel'`` ignores it and does the
    much easier K-fold over voxels; it is reported only as an upper bound.
    Parcels unseen in a fold's training data fall back to the training grand mean.
    """
    M, L = Y.shape
    rng = np.random.default_rng(seed)
    if split == "voxel":
        fold = rng.integers(0, n_folds, M)
    else:
        groups = np.unique(group_id)
        assign = {g: i % n_folds for i, g in enumerate(rng.permutation(groups))}
        fold = np.array([assign[g] for g in group_id])

    pred = np.zeros_like(Y)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        if not tr.any() or not te.any():
            continue
        cnt = np.bincount(labels[tr], minlength=n_parcels).astype(np.float64)
        means = np.zeros((n_parcels, L), dtype=np.float64)
        np.add.at(means, labels[tr], Y[tr].astype(np.float64))
        grand = Y[tr].mean(0)
        seen = cnt > 0
        means[seen] /= cnt[seen, None]
        means[~seen] = grand
        pred[te] = means[labels[te]].astype(np.float32)

    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return {
        "split": split, "n_folds": int(n_folds),
        "ev_mean": float(r2.mean()), "ev_median": float(np.median(r2)),
        "ev_p10": float(np.percentile(r2, 10)),
        "ev_p90": float(np.percentile(r2, 90)),
        "per_lipid": r2.astype(np.float32),
    }


# --------------------------------------------------------------------------- #
# label sources
# --------------------------------------------------------------------------- #
def atlas_labels_at_nodes(node_vox, atlas_file, stride):
    """Allen annotation ids for the template nodes (the reference row).

    Sampled on the same strided grid the parcel fields use, so the atlas row is
    scored through exactly the same machinery as everything else.
    """
    annot, _ = stride_volume(np.load(atlas_file), stride)
    z, y, x = node_vox.T
    _, labels = np.unique(annot[z, y, x], return_inverse=True)
    return labels.astype(np.int32), int(labels.max() + 1)


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--field", action="append", default=[], metavar="[NAME=]PATH",
                   help="Parcel-field .npz to score. Repeatable.")
    p.add_argument("--maldi-file", required=True)
    p.add_argument("--available-lipids-file", required=True)
    p.add_argument("--reference-file", required=True)
    p.add_argument("--slices-dataset-file", default=None,
                   help="Splits json; only its 'train' sections are scored.")
    p.add_argument("--atlas-file", default=None,
                   help="Allen annotation .npy to include as a reference row.")
    p.add_argument("--stride", type=int, default=4, help="(--atlas-file only)")
    p.add_argument("--threshold", type=float, default=5, help="(--atlas-file only)")
    p.add_argument("--log-transform", action="store_true")
    p.add_argument("--max-sections", type=int, default=0,
                   help="Keep at most N whole sections (0 = all). Subsampling is "
                        "always by section, never by voxel — the adjacency check "
                        "needs intact pixel neighbourhoods.")
    p.add_argument("--max-voxels", type=int, default=0,
                   help="Additional cap, applied by dropping whole sections.")
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--max-lipids", type=int, default=0,
                   help="Score only the first N lipids (0 = all).")
    p.add_argument("--ev-split", choices=["section", "sample", "voxel"],
                   default="section")
    p.add_argument("--pair-space", choices=["node", "pixel"], default="node",
                   help="'node' (default) pools measurements onto the 0.1 mm "
                        "template grid before differencing — required for a clean "
                        "cross/within split, see pool_to_nodes. 'pixel' uses raw "
                        "adjacent acquisition voxels (noisy, label-contaminated); "
                        "kept for comparison only.")
    p.add_argument("--min-count", type=int, default=3,
                   help="(--pair-space node) minimum measured voxels per node.")
    p.add_argument("--include-ap-pairs", action="store_true",
                   help="(--pair-space node) also pair nodes across the "
                        "anterior-posterior axis, i.e. across sections.")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Write the full report as json.")
    args = p.parse_args(argv)

    names = list(np.load(args.available_lipids_file, allow_pickle=True))
    if args.max_lipids:
        names = names[:args.max_lipids]

    ref = load_reference(args.reference_file)
    coord_mean, coord_std = coord_norm_from_reference(ref)
    coords_std, Y, section_id, pix, sample_id = load_maldi(
        args.maldi_file, names, load_filters(args.slices_dataset_file, "train"),
        coord_mean, coord_std, args.log_transform, args.max_sections,
        args.max_voxels, args.seed)
    ev_group = {"section": section_id, "sample": sample_id,
                "voxel": section_id}[args.ev_split]

    fields = []
    for spec in args.field:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(name).stem, name
        fields.append((name, ParcelField.load(path)))
    if not fields:
        p.error("nothing to score: pass at least one --field "
                "(--atlas-file alone has no node grid to score on)")
    ref_field = fields[0][1]
    for name, f in fields[1:]:
        if f.node_coords.shape != ref_field.node_coords.shape:
            p.error(f"field {name!r} has a different node set than "
                    f"{fields[0][0]!r} — rebuild them with the same "
                    f"--stride/--threshold so they are comparable")

    # Checks 1 & 2 run on nodes; check 3 runs on the raw voxels (a mean-level
    # statistic, unaffected by per-voxel noise, and its section holdout is the
    # question we actually care about).
    if args.pair_space == "node":
        node_ids, Ybar, _ = pool_to_nodes(coords_std, Y, ref_field, args.min_count)
        pi, pj, ps = node_adjacent_pairs(ref_field.node_vox, node_ids,
                                         include_ap=args.include_ap_pairs)
        at_nodes = True
    else:
        node_ids, Ybar = None, Y
        pi, pj, ps = adjacent_pairs(pix, section_id, max_pairs=args.max_pairs,
                                    seed=args.seed)
        at_nodes = False

    sources = []
    for name, f in fields:
        pair_labels = f.labels[node_ids] if at_nodes else f.sample(coords_std).label
        pair_drel = (f.d_border_rel[node_ids] if at_nodes
                     else f.sample(coords_std).d_border_rel)
        sources.append((name, pair_labels, pair_drel, f.n_parcels,
                        f.sample(coords_std).label))
    if args.atlas_file:
        lab_nodes, K = atlas_labels_at_nodes(ref_field.node_vox, args.atlas_file,
                                             args.stride)
        vox_lab = lab_nodes[ref_field.node_index(coords_std)]
        sources.append(("atlas", lab_nodes[node_ids] if at_nodes else vox_lab,
                        None, K, vox_lab))

    report = {}
    for name, pair_labels, pair_drel, K, vox_labels in sources:
        log.info("scoring %r (K=%d) ...", name, K)
        report[name] = {
            "n_parcels": int(K),
            "pair_space": args.pair_space,
            "boundary_contrast": boundary_contrast(
                Ybar, pi, pj, ps, pair_labels, n_boot=args.n_boot, seed=args.seed),
            "parcel_ev": parcel_ev(Y, vox_labels, ev_group, K,
                                   split=args.ev_split, seed=args.seed),
            "border_trend": (
                border_trend(Ybar, pi, pj, pair_labels, pair_drel,
                             n_bins=args.n_bins) if pair_drel is not None else None),
        }

    hdr = f"{'parcellation':<24}{'K':>6}{'contrast':>12}{'ci95':>18}{'EV':>8}{'near/far':>11}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, e in report.items():
        bc, ev, bt = e["boundary_contrast"], e["parcel_ev"], e["border_trend"]
        ci = f"[{bc['contrast_ci95'][0]:.2f},{bc['contrast_ci95'][1]:.2f}]"
        nf = f"{bt['near_over_far']:.3f}" if bt else "-"
        print(f"{name:<24}{e['n_parcels']:>6}{bc['contrast_mean']:>12.3f}"
              f"{ci:>18}{ev['ev_mean']:>8.3f}{nf:>11}")
    print("\nbar to clear: contrast > 1.5 (Allen atlas = 1.22), EV > 0.25 "
          "(atlas 0.18, spatial k-means 0.25, lipid k-means 0.79 = ceiling), "
          "near/far > 1.1 for the border feature to be worth building.\n")

    if args.out:
        def _clean(o):
            if isinstance(o, dict):
                return {k: _clean(v) for k, v in o.items() if k != "per_lipid"}
            return o
        Path(args.out).write_text(json.dumps(_clean(report), indent=2))
        log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
