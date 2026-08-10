"""How good could a reference-only parcellation possibly be?

Beating the Allen atlas is a low bar -- the atlas explains ~14% of the lipid
variance, so "1.8x the atlas" can still be nowhere near useful. The question that
decides whether to keep working on the parcellation is not "did we beat the
atlas" but "how much of the reachable signal are we already capturing".

Three reference points, all scored with the SAME held-out-by-section R^2 that
``validate.parcel_ev`` uses, so the numbers are directly comparable to the
parcellation table:

  **lipid_oracle** -- k-means on the LIPIDS themselves at matched K. This is the
  best any K-region partition could do; it cannot be built in production (it needs
  the answer), it exists to put a number on the top of the scale. To keep it from
  being circular it is fit on one random half of the lipid panel and scored on the
  other half, and every method in the comparison is scored on that same held-out
  half.

  **ridge / knn on (position + template appearance)** -- not parcellations at all,
  but regressions from exactly the information a reference-only parcellation is
  allowed to use: where you are, and what the template looks like around you. Any
  parcellation is a piecewise-constant function of those inputs, so a flexible
  regression on them upper-bounds the whole family. If the parcellation is already
  close to this, no amount of cleverness in the clustering will help and the lever
  has to be a richer input, not a better partition.

  **knn on position alone** -- the "just interpolate from the nearest measured
  section" control. Anything that does not beat this is not earning its keep.

Usage::

    python -m other_experiments.parcelgp.ceiling --field parcels/full_k128.npz \\
        --maldi-file ... --available-lipids-file ... --reference-file ... \\
        --slices-dataset-file maldi/data/splits/fold_2.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsRegressor

from .features import FeatureSpec, template_features
from .field import ParcelField
from .validate import (load_filters, load_maldi, parcel_ev, pool_to_nodes)
from .volume import (coord_norm_from_reference, load_reference, node_voxels,
                     stride_volume, tissue_mask)

log = logging.getLogger("parcelgp.ceiling")


def section_folds(group_id, n_folds=5, seed=0):
    groups = np.unique(group_id)
    assign = {g: i % n_folds for i, g in enumerate(
        np.random.default_rng(seed).permutation(groups))}
    return np.array([assign[g] for g in group_id])


def regression_ceiling(X, Y, fold, n_folds=5, method="ridge", k=32, alpha=1.0,
                       max_train=150_000, max_test=60_000, seed=0):
    """Held-out R^2 per lipid for a regression from ``X`` to every lipid.

    Three estimators, chosen so each is affordable in the regime it is used:

      ``ridge``  linear bound, multi-output, essentially free.
      ``knn``    nonparametric, multi-output -- but only usable on the 3-D
                 position input. A kd-tree degenerates to brute force by ~19
                 dimensions, so this must NOT be pointed at the full feature set.
      ``gb``     histogram gradient boosting, the nonparametric bound for the full
                 feature set. Single-output, so it is fit per lipid and the caller
                 restricts how many lipids it runs on.

    Train and test rows are subsampled per fold: R^2 over tens of thousands of
    held-out voxels is already tight to well under the differences we care about,
    and the fit/query is what costs.
    """
    rng = np.random.default_rng(seed)
    num = np.zeros(Y.shape[1], dtype=np.float64)
    den = np.zeros(Y.shape[1], dtype=np.float64)
    for f in range(n_folds):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        if tr.size == 0 or te.size == 0:
            continue
        if max_train and tr.size > max_train:
            tr = rng.choice(tr, max_train, replace=False)
        if max_test and te.size > max_test:
            te = rng.choice(te, max_test, replace=False)
        if method == "ridge":
            Xt = np.hstack([X[tr], np.ones((tr.size, 1), np.float32)]).astype(np.float64)
            W = np.linalg.solve(Xt.T @ Xt + alpha * np.eye(Xt.shape[1]),
                                Xt.T @ Y[tr].astype(np.float64))
            pred = np.hstack([X[te], np.ones((te.size, 1), np.float32)]) @ W
        elif method == "knn":
            m = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
            m.fit(X[tr], Y[tr])
            pred = m.predict(X[te])
        elif method == "gb":
            from sklearn.ensemble import HistGradientBoostingRegressor
            pred = np.empty((te.size, Y.shape[1]), dtype=np.float64)
            for c in range(Y.shape[1]):
                g = HistGradientBoostingRegressor(
                    max_iter=200, learning_rate=0.1, max_depth=None,
                    early_stopping=False, random_state=seed)
                g.fit(X[tr], Y[tr, c])
                pred[:, c] = g.predict(X[te])
        else:
            raise ValueError(method)
        # Accumulate against the FOLD's own mean so folds of different sizes
        # combine correctly into one panel-level R^2.
        num += ((Y[te] - pred) ** 2).sum(0)
        den += ((Y[te] - Y[tr].mean(0)) ** 2).sum(0)
        log.info("  %s fold %d/%d (%d train, %d test)",
                 method, f + 1, n_folds, tr.size, te.size)
    return 1.0 - num / np.maximum(den, 1e-12)


def lipid_oracle_labels(Ybar_fit, n_parcels, seed=0):
    """k-means on the lipids themselves -- the best-possible K-region partition."""
    km = MiniBatchKMeans(n_clusters=int(n_parcels), random_state=int(seed),
                         n_init=10, batch_size=4096, max_iter=300)
    return km.fit_predict(Ybar_fit).astype(np.int32)


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--field", required=True,
                   help="Parcel field to place on the scale (also supplies the "
                        "node grid and K for the oracle).")
    p.add_argument("--maldi-file", required=True)
    p.add_argument("--available-lipids-file", required=True)
    p.add_argument("--reference-file", required=True)
    p.add_argument("--slices-dataset-file", default=None)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--threshold", type=float, default=5)
    p.add_argument("--log-transform", action="store_true")
    p.add_argument("--max-sections", type=int, default=0)
    p.add_argument("--max-lipids", type=int, default=0)
    p.add_argument("--min-count", type=int, default=3)
    p.add_argument("--knn-k", type=int, default=32)
    p.add_argument("--gb-lipids", type=int, default=20,
                   help="How many of the scored lipids the (per-lipid) gradient "
                        "boosted bound is run on.")
    p.add_argument("--max-train", type=int, default=150_000)
    p.add_argument("--max-test", type=int, default=60_000)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    names = list(np.load(args.available_lipids_file, allow_pickle=True))
    if args.max_lipids:
        names = names[:args.max_lipids]

    ref = load_reference(args.reference_file)
    coord_mean, coord_std = coord_norm_from_reference(ref)
    coords_std, Y, section_id, _, _ = load_maldi(
        args.maldi_file, names, load_filters(args.slices_dataset_file, "train"),
        coord_mean, coord_std, args.log_transform, args.max_sections, 0, args.seed)

    field = ParcelField.load(args.field)
    K = field.n_parcels

    # Split the panel: the oracle is fit on half A, everything is scored on half B.
    perm = np.random.default_rng(args.seed).permutation(Y.shape[1])
    half_a, half_b = perm[:Y.shape[1] // 2], perm[Y.shape[1] // 2:]
    Yb = Y[:, half_b]
    log.info("lipid panel split: %d fit / %d scored", half_a.size, half_b.size)

    # Template appearance at every measured voxel (via its nearest node).
    sub, voxel_scale_mm = stride_volume(ref, args.stride)
    mask = tissue_mask(sub, args.threshold)
    node_vox = node_voxels(mask)
    feats, fnames = template_features(sub, mask, node_vox, voxel_scale_mm,
                                      FeatureSpec())
    vidx = field.node_index(coords_std)
    Xpos = coords_std
    Xfull = np.hstack([coords_std, feats[vidx]]).astype(np.float32)

    fold = section_folds(section_id, args.n_folds, args.seed)
    rows = {}

    log.info("scoring the parcellation under test ...")
    rows[Path(args.field).stem] = parcel_ev(
        Yb, field.labels[vidx], section_id, K, split="section",
        n_folds=args.n_folds, seed=args.seed)["ev_mean"]

    log.info("scoring the lipid oracle (K=%d) ...", K)
    node_ids, Ybar, _ = pool_to_nodes(coords_std, Y[:, half_a], field, args.min_count)
    oracle_nodes = np.zeros(field.node_coords.shape[0], dtype=np.int32)
    oracle_nodes[node_ids] = lipid_oracle_labels(Ybar, K, args.seed)
    rows[f"lipid_oracle_k{K}"] = parcel_ev(
        Yb, oracle_nodes[vidx], section_id, K, split="section",
        n_folds=args.n_folds, seed=args.seed)["ev_mean"]

    # The gradient-boosted bound is per-lipid, so it runs on a subset of the
    # scored half; everything else runs on all of it.
    gb_cols = np.arange(min(args.gb_lipids, Yb.shape[1]))
    for label, X, method, cols in (
        ("knn position only", Xpos, "knn", None),
        ("ridge position+template", Xfull, "ridge", None),
        (f"gbtree position+template ({gb_cols.size}lip)", Xfull, "gb", gb_cols),
        (f"gbtree position only ({gb_cols.size}lip)", Xpos, "gb", gb_cols),
    ):
        log.info("scoring %r ...", label)
        Yt = Yb if cols is None else Yb[:, cols]
        r2 = regression_ceiling(X, Yt, fold, args.n_folds, method,
                                k=args.knn_k, max_train=args.max_train,
                                max_test=args.max_test, seed=args.seed)
        rows[label] = float(r2.mean())

    print(f"\n{'method':<32}{'held-out R2':>13}")
    print("-" * 45)
    for k, v in sorted(rows.items(), key=lambda kv: kv[1]):
        print(f"{k:<32}{v:>13.3f}")
    print("\nscored on a held-out half of the lipid panel, sections held out 5-fold.\n")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
