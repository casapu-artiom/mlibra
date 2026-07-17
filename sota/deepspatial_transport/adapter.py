"""MALDI -> DeepSpatial AnnData adapter (transport / slice-interpolation mode).

Turns the MALDI parquet into the per-section ``AnnData`` list the official
``deepspatial`` package consumes, mapping the MALDI concepts onto DeepSpatial's:

    DeepSpatial (single-cell ST)     MALDI
    ----------------------------     ------------------------------------------
    cell                             measured voxel
    slice / section                  physical coronal section (`Section` col)
    2D in-plane coords  (obsm)       (yccf, zccf)   -- the two non-section axes
    depth / z position  (obs)        xccf           -- the SECTIONING axis (AP)
    gene expression     (.X)         lipid intensities
    cell type           (obs)        Allen atlas region (level_15annot, ~70)

Two MALDI-specific adaptations (documented, necessary):
  * Sectioning axis is ``xccf`` (index 0), NOT z -- per-section mean xccf marches
    monotonically while y/z stay ~constant. So depth = xccf, in-plane = (y,z).
  * MALDI sections carry 40-75k voxels; the official UOT builds a full N0xN1
    coupling, which is infeasible at that size. Training sections are therefore
    randomly SUBSAMPLED to ``max_cells`` voxels before the coupling (the paper's
    ST slices are far sparser). Reconstruction can use denser sections.
"""
import logging

import numpy as np
import pandas as pd
import anndata as ad

# CCF metadata columns pulled alongside the lipid matrix.
_META_COLS = ["Sample", "Section", "xccf", "yccf", "zccf",
              "x_index", "y_index", "z_index"]


def _region_labels(annot, x_index, y_index, z_index):
    """Atlas region id per voxel. The volume/annot is (528,320,456) =
    (x_index, y_index, z_index) -- forced by the index bounds (x_index reaches
    520 > 456, so it must index the 528 axis; z_index maxes at 434 -> the 456
    axis). So annot[x_index, y_index, z_index]."""
    xi = np.clip(x_index.astype(int), 0, annot.shape[0] - 1)
    yi = np.clip(y_index.astype(int), 0, annot.shape[1] - 1)
    zi = np.clip(z_index.astype(int), 0, annot.shape[2] - 1)
    return annot[xi, yi, zi]


def load_sections(maldi_file, lipid_cols, filters, annot_path,
                  max_cells=None, seed=0):
    """Read the parquet (with a pyarrow ``filters`` predicate) and return
    ``{sample: [AnnData per section, ordered by Section]}``.

    Each AnnData:
      .X                      lipid matrix (n_voxels, n_lipids), float32
      .var_names              lipid names
      .obsm['spatial']        (yccf, zccf)                       in-plane coords
      .obs['z_coord']         section mean xccf (scalar per slice, AP depth)
      .obs['cell_class']      atlas region (str)
      .obs[xccf/yccf/zccf/x_index/y_index/z_index]   kept for rasterization
    """
    rng = np.random.default_rng(seed)
    cols = [str(c) for c in lipid_cols] + _META_COLS
    logging.info(f"  adapter: reading parquet ({len(lipid_cols)} lipids)...")
    df = pd.read_parquet(maldi_file, columns=cols, filters=filters)
    logging.info(f"  adapter: {len(df):,} voxels across "
                 f"{df['Sample'].nunique()} sample(s)")
    annot = np.load(annot_path)

    lipid_str = [str(c) for c in lipid_cols]
    out = {}
    for sample, sdf in df.groupby("Sample", sort=False):
        sections = []
        for sec, gdf in sorted(sdf.groupby("Section"), key=lambda kv: kv[0]):
            # max_cells falsy/<=0 => no cap (use all voxels of the section).
            if max_cells and max_cells > 0 and len(gdf) > max_cells:
                gdf = gdf.iloc[rng.choice(len(gdf), max_cells, replace=False)]
            X = gdf[lipid_str].to_numpy(dtype=np.float32)
            X[X < 0] = 0.0                          # match harness negative-impute
            xccf = gdf["xccf"].to_numpy(np.float32)
            yccf = gdf["yccf"].to_numpy(np.float32)
            zccf = gdf["zccf"].to_numpy(np.float32)
            region = _region_labels(annot, gdf["x_index"].to_numpy(),
                                    gdf["y_index"].to_numpy(),
                                    gdf["z_index"].to_numpy())
            a = ad.AnnData(X=X)
            a.var_names = lipid_str
            a.obsm["spatial"] = np.stack([yccf, zccf], axis=1)     # (y, z) in-plane
            a.obs["z_coord"] = float(xccf.mean())                  # AP depth
            a.obs["cell_class"] = pd.Categorical(region.astype(str))
            for name, arr in [("xccf", xccf), ("yccf", yccf), ("zccf", zccf),
                              ("x_index", gdf["x_index"].to_numpy()),
                              ("y_index", gdf["y_index"].to_numpy()),
                              ("z_index", gdf["z_index"].to_numpy())]:
                a.obs[name] = arr
            a.obs["Section"] = float(sec)
            sections.append(a)
        out[sample] = sections
        logging.info(f"    {sample}: {len(sections)} sections, "
                     f"{sum(s.n_obs for s in sections):,} voxels")
    return out


def fit_ccf_to_index(maldi_file, filters):
    """Fit a per-axis linear map ccf(mm) -> voxel index from measured data, so
    synthesized cells (with interpolated ccf coords) can be placed on the CCF
    grid. Returns a dict of (slope, intercept) per axis."""
    df = pd.read_parquet(
        maldi_file,
        columns=["xccf", "yccf", "zccf", "x_index", "y_index", "z_index"],
        filters=filters,
    )
    m = {}
    for c, i in [("xccf", "x_index"), ("yccf", "y_index"), ("zccf", "z_index")]:
        sl, inter = np.polyfit(df[c].to_numpy(np.float64), df[i].to_numpy(np.float64), 1)
        m[c] = (float(sl), float(inter))
    return m


def syn_flat_indices(adata_syn, ccf2idx, template_shape):
    """Flat CCF-grid voxel index for each synthesized cell (obsm['spatial']=
    (yccf,zccf), obs['z_coord']=xccf). Volume dims (528,320,456) =
    (x_index, y_index, z_index). Returns int64 flat indices for
    np.ravel over (dim0,dim1,dim2). Used by the memory-frugal per-segment path
    so we never materialize the whole brain's cells at once."""
    xccf = np.asarray(adata_syn.obs["z_coord"], dtype=np.float64)
    yccf = adata_syn.obsm["spatial"][:, 0].astype(np.float64)
    zccf = adata_syn.obsm["spatial"][:, 1].astype(np.float64)

    def to_idx(vals, axis, hi):
        sl, inter = ccf2idx[axis]
        return np.clip(np.round(sl * vals + inter).astype(np.int64), 0, hi - 1)

    d0 = to_idx(xccf, "xccf", template_shape[0])
    d1 = to_idx(yccf, "yccf", template_shape[1])
    d2 = to_idx(zccf, "zccf", template_shape[2])
    return (d0 * template_shape[1] + d1) * template_shape[2] + d2


def flat_to_indices(flat, template_shape):
    """Inverse of ``syn_flat_indices``: flat -> (M,3) (dim0,dim1,dim2) columns,
    the order ``_write_per_lipid_volumes`` writes."""
    d0 = flat // (template_shape[1] * template_shape[2])
    rem = flat % (template_shape[1] * template_shape[2])
    d1 = rem // template_shape[2]
    d2 = rem % template_shape[2]
    return np.stack([d0, d1, d2], axis=1).astype(np.int64)


def rasterize_to_volume(adata_syn, ccf2idx, template_shape):
    """Map a synthesized AnnData (obsm['spatial']=(yccf,zccf),
    obs['z_coord']=xccf, .X=lipids) onto the CCF voxel grid.

    Returns (predictions (M, n_lipids), indices (M, 3)) where indices columns are
    (x_index, y_index, z_index) = the template dims (528,320,456), the order
    _write_per_lipid_volumes writes (vol[indices[:,0], [:,1], [:,2]]).
    Duplicate voxels are averaged.
    """
    xccf = np.asarray(adata_syn.obs["z_coord"], dtype=np.float64)
    yccf = adata_syn.obsm["spatial"][:, 0].astype(np.float64)
    zccf = adata_syn.obsm["spatial"][:, 1].astype(np.float64)

    def to_idx(vals, axis, hi):
        sl, inter = ccf2idx[axis]
        return np.clip(np.round(sl * vals + inter).astype(np.int64), 0, hi - 1)

    # Volume dims (528,320,456) = (x_index, y_index, z_index): x_index reaches
    # 520 > 456 so it MUST index the 528 axis; z_index maxes at 434 -> 456 axis.
    d0 = to_idx(xccf, "xccf", template_shape[0])   # x_index -> dim0
    d1 = to_idx(yccf, "yccf", template_shape[1])   # y_index -> dim1
    d2 = to_idx(zccf, "zccf", template_shape[2])   # z_index -> dim2

    X = adata_syn.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = X.astype(np.float32)

    # Average duplicate voxels (multiple synthesized cells in one CCF voxel).
    flat = (d0.astype(np.int64) * template_shape[1] + d1) * template_shape[2] + d2
    uniq, inv = np.unique(flat, return_inverse=True)
    K = X.shape[1]
    acc = np.zeros((len(uniq), K), dtype=np.float64)
    cnt = np.zeros(len(uniq), dtype=np.float64)
    np.add.at(acc, inv, X)
    np.add.at(cnt, inv, 1.0)
    preds = (acc / cnt[:, None]).astype(np.float32)

    uz = uniq // (template_shape[1] * template_shape[2])
    rem = uniq % (template_shape[1] * template_shape[2])
    uy = rem // template_shape[2]
    ux = rem % template_shape[2]
    indices = np.stack([uz, uy, ux], axis=1).astype(np.int64)
    return preds, indices
