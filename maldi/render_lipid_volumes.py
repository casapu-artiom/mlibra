"""Headless multi-panel rendering for lipid volumes.

Pure matplotlib + scipy — no napari, no Qt, no OpenGL. Generates one
composite PNG per lipid showing 4 rows × 10 columns:
    Row 0: coronal slices (X-Z plane, varying Y)
    Row 1: axial   slices (X-Y plane, varying Z)
    Row 2: sagittal slices (Y-Z plane, varying X)
    Row 3: 3D MIP rotation frames (computed via scipy.ndimage.rotate)

If a `template_volume` (anatomy reference) is provided it's drawn faintly
behind each 2D slice for context.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")           # never touch a display
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from tqdm import tqdm


# Slice positions as fractions of axis length. 10 cuts from 5% to 95%.
SLICE_FRACTIONS = (0.05, 0.15, 0.25, 0.35, 0.45,
                   0.55, 0.65, 0.75, 0.85, 0.95)
SLICE_LABELS = ("5%", "15%", "25%", "35%", "45%",
                "55%", "65%", "75%", "85%", "95%")


# -----------------------------------------------------------------------------
# Volume normalization
# -----------------------------------------------------------------------------

def _normalize_volume(volume: np.ndarray,
                       robust: bool = False) -> tuple[np.ndarray, float, float]:
    """Map to [0, 1]. Default uses nanmin/nanmax (matches the original
    napari look). robust=True uses 2nd/98th percentiles when outliers
    crush contrast. NaNs preserved either way."""
    if robust:
        finite = volume[np.isfinite(volume)]
        if finite.size == 0:
            return np.zeros_like(volume), 0.0, 0.0
        vmin = float(np.percentile(finite, 2))
        vmax = float(np.percentile(finite, 98))
    else:
        vmin = float(np.nanmin(volume))
        vmax = float(np.nanmax(volume))
    if vmax > vmin:
        out = (volume - vmin) / (vmax - vmin)
        if robust:
            out = np.clip(out, 0.0, 1.0)
    else:
        out = np.zeros_like(volume)
    return out, vmin, vmax


# -----------------------------------------------------------------------------
# 2D slice rendering (pure matplotlib)
# -----------------------------------------------------------------------------

def _show_slice(ax, slice_2d: np.ndarray,
                anatomy_2d: np.ndarray | None,
                title: str, cmap: str = "inferno"):
    """White axes face, faint anatomy outline behind, lipid signal on top."""
    ax.set_facecolor("white")
    if anatomy_2d is not None:
        ax.imshow(anatomy_2d, cmap="gray_r", alpha=0.2, interpolation="nearest")
    ax.imshow(slice_2d, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=6, color="black")
    ax.axis("off")


def _render_2d_panels(axs, volume_norm: np.ndarray,
                       anatomy: np.ndarray | None, cmap: str = "inferno"):
    """Rows 0-2 of axs (a (4, 10) array) get the three slice planes."""
    Z, Y, X = volume_norm.shape
    z_pos = [int(Z * f) for f in SLICE_FRACTIONS]
    y_pos = [int(Y * f) for f in SLICE_FRACTIONS]
    x_pos = [int(X * f) for f in SLICE_FRACTIONS]

    # Row 0: coronal (X-Z, varying Y)
    for i, y in enumerate(y_pos):
        sl = volume_norm[:, y, :]
        an = anatomy[:, y, :] if anatomy is not None else None
        _show_slice(axs[0, i], sl, an,
                    f"Coronal Y={y} ({SLICE_LABELS[i]})", cmap)

    # Row 1: axial (X-Y, varying Z)
    for i, z in enumerate(z_pos):
        sl = volume_norm[z, :, :]
        an = anatomy[z, :, :] if anatomy is not None else None
        _show_slice(axs[1, i], sl, an,
                    f"Axial Z={z} ({SLICE_LABELS[i]})", cmap)

    # Row 2: sagittal (Y-Z, varying X)
    for i, x in enumerate(x_pos):
        sl = volume_norm[:, :, x]
        an = anatomy[:, :, x] if anatomy is not None else None
        _show_slice(axs[2, i], sl, an,
                    f"Sagittal X={x} ({SLICE_LABELS[i]})", cmap)


# -----------------------------------------------------------------------------
# 3D rotation — pure scipy + matplotlib, no GL
# -----------------------------------------------------------------------------

def _rotation_mips(volume_norm: np.ndarray,
                    n_frames: int = 10,
                    downsample: int = 2,
                    elevation_deg: float = 25.0) -> list[np.ndarray]:
    """Rotate volume around Y at evenly-spaced azimuths with a fixed
    elevation tilt; MIP along the rotated depth axis at each step.

    Tracks a finite-mask volume in parallel so the MIP output has NaN
    outside the brain — matches the 2D slice backgrounds exactly.
    """
    vol = np.where(np.isfinite(volume_norm), volume_norm, 0.0).astype(np.float32)
    mask = np.isfinite(volume_norm).astype(np.float32)

    if downsample > 1:
        vol  = vol [::downsample, ::downsample, ::downsample]
        mask = mask[::downsample, ::downsample, ::downsample]

    # Elevation tilt applied once. Rotation around X (axes=(0,1) = Z-Y plane)
    # so the camera looks down at the brain from above by `elevation_deg`.
    if abs(elevation_deg) > 1e-3:
        vol  = rotate(vol,  elevation_deg, axes=(0, 1),
                      reshape=False, order=1, mode="constant", cval=0.0)
        mask = rotate(mask, elevation_deg, axes=(0, 1),
                      reshape=False, order=1, mode="constant", cval=0.0)

    frames = []
    for i in tqdm(range(n_frames), desc="3D rotation MIPs", leave=False):
        azimuth = i * (360.0 / n_frames)
        rotated      = rotate(vol,  azimuth, axes=(0, 2),
                              reshape=False, order=1,
                              mode="constant", cval=0.0)
        rotated_mask = rotate(mask, azimuth, axes=(0, 2),
                              reshape=False, order=1,
                              mode="constant", cval=0.0)
        mip      = rotated.max(axis=2)
        mip_mask = rotated_mask.max(axis=2)
        # NaN-out background so it renders as axes facecolor (true black)
        mip = np.where(mip_mask > 0.05, mip, np.nan)
        frames.append(mip)
    return frames


def _render_3d_row(axs_row, volume_norm: np.ndarray,
                    n_frames: int = 10, downsample: int = 2,
                    elevation_deg: float = 25.0):
    """Row 3: rotation MIPs."""
    frames = _rotation_mips(volume_norm, n_frames,
                             downsample=downsample,
                             elevation_deg=elevation_deg)
    for i, frame in enumerate(frames):
        axs_row[i].set_facecolor("white")
        axs_row[i].imshow(frame, cmap="inferno", vmin=0, vmax=1,
                          interpolation="nearest")
        angle = i * (360 // n_frames)
        axs_row[i].set_title(f"3D Rotation - {angle}°",
                              fontsize=6, color="black")
        axs_row[i].axis("off")


def render_lipid_volume(
    volume: np.ndarray,
    lipid_name: str,
    out_path: Path,
    anatomy: np.ndarray | None = None,
    n_rotation_frames: int = 10,
    cmap: str = "inferno",
    robust_normalize: bool = False,
    downsample_3d: int = 2,
    elevation_deg: float = 25.0,
    dpi: int = 200,
):
    """Render one lipid volume to a composite PNG."""
    volume_norm, vmin, vmax = _normalize_volume(volume, robust=robust_normalize)

    if anatomy is not None and anatomy.shape == volume.shape:
        anatomy_norm, _, _ = _normalize_volume(anatomy, robust=False)
    else:
        anatomy_norm = None

    fig, axs = plt.subplots(
        nrows=4, ncols=n_rotation_frames,
        figsize=(3 * n_rotation_frames, 11),
        facecolor="white",
        constrained_layout=True,
    )
    fig.suptitle(
        f"Lipid {lipid_name}   (range {vmin:.4g} – {vmax:.4g})",
        fontsize=14, color="black",
    )

    _render_2d_panels(axs, volume_norm, anatomy_norm, cmap=cmap)
    _render_3d_row(axs[3], volume_norm,
                    n_frames=n_rotation_frames,
                    downsample=downsample_3d,
                    elevation_deg=elevation_deg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Reconstruction diagnostics: value distribution + true-vs-pred scatter
# -----------------------------------------------------------------------------

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, NaN-safe against degenerate (constant) inputs."""
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def render_lipid_diagnostics(
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    lipid_name: str,
    out_path: Path,
    dpi: int = 150,
):
    """One PNG per lipid: the value distribution and the true-vs-predicted
    scatter, each shown in BOTH the current (linear) range and the
    log-transformed range.

    ``true_vals`` / ``pred_vals`` are paired 1-D arrays in the model's native
    (raw, de-standardized) intensity units — typically the held-out TEST points
    where both a measurement and a prediction exist. The log panels drop
    non-positive values (which have no log) and use log-scaled axes.

    Layout (2×2)::

        [ distribution — linear ]   [ distribution — log ]
        [ scatter      — linear ]   [ scatter      — log ]
    """
    true_vals = np.asarray(true_vals, dtype=np.float64).ravel()
    pred_vals = np.asarray(pred_vals, dtype=np.float64).ravel()
    finite = np.isfinite(true_vals) & np.isfinite(pred_vals)
    true_vals, pred_vals = true_vals[finite], pred_vals[finite]
    if true_vals.size == 0:
        logging.warning(f"diagnostics: {lipid_name} has no finite paired "
                        f"values; skipping.")
        return

    pear = _corr(true_vals, pred_vals)

    fig, axs = plt.subplots(2, 2, figsize=(12, 11), facecolor="white")
    fig.suptitle(f"Lipid {lipid_name} — reconstruction diagnostics "
                 f"(n={true_vals.size:,})", fontsize=14)

    # ---- Row 0: value distributions (true vs pred) ----
    lo = float(min(true_vals.min(), pred_vals.min()))
    hi = float(max(true_vals.max(), pred_vals.max()))
    bins = np.linspace(lo, hi, 60) if hi > lo else 60
    ax = axs[0, 0]
    ax.hist(true_vals, bins=bins, alpha=0.5, color="tab:blue", label="true")
    ax.hist(pred_vals, bins=bins, alpha=0.5, color="tab:orange", label="pred")
    ax.set_title("Value distribution (linear)")
    ax.set_xlabel("value"); ax.set_ylabel("count"); ax.legend()

    ax = axs[0, 1]
    tpos, ppos = true_vals[true_vals > 0], pred_vals[pred_vals > 0]
    if tpos.size and ppos.size:
        loln = np.log10(min(tpos.min(), ppos.min()))
        hiln = np.log10(max(tpos.max(), ppos.max()))
        logbins = np.logspace(loln, hiln, 60) if hiln > loln else 60
        ax.hist(tpos, bins=logbins, alpha=0.5, color="tab:blue", label="true")
        ax.hist(ppos, bins=logbins, alpha=0.5, color="tab:orange", label="pred")
        ax.set_xscale("log"); ax.legend()
    else:
        ax.text(0.5, 0.5, "no positive values", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("Value distribution (log)")
    ax.set_xlabel("value"); ax.set_ylabel("count")

    # ---- Row 1: true vs pred scatter (density via hexbin) ----
    ax = axs[1, 0]
    if hi > lo:
        ax.hexbin(true_vals, pred_vals, gridsize=50, cmap="inferno",
                  bins="log", mincnt=1)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(f"True vs Pred (linear)   r={pear:.3f}")
    ax.set_xlabel("true"); ax.set_ylabel("pred")

    ax = axs[1, 1]
    m = (true_vals > 0) & (pred_vals > 0)
    if m.sum() >= 2:
        t, p = true_vals[m], pred_vals[m]
        r_log = _corr(np.log(t), np.log(p))
        lo2 = float(min(t.min(), p.min())); hi2 = float(max(t.max(), p.max()))
        ax.hexbin(t, p, gridsize=50, cmap="inferno", bins="log", mincnt=1,
                  xscale="log", yscale="log")
        ax.plot([lo2, hi2], [lo2, hi2], "k--", lw=1)
        ax.set_xlim(lo2, hi2); ax.set_ylim(lo2, hi2)
        ax.set_title(f"True vs Pred (log)   r_log={r_log:.3f}")
    else:
        ax.text(0.5, 0.5, "no positive pairs", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("True vs Pred (log)")
    ax.set_xlabel("true"); ax.set_ylabel("pred")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _pick_section(coords: np.ndarray):
    """Guess which axis the MALDI sections run along and pick the densest one.

    MALDI is measured section-by-section: along the sectioning axis the
    coordinates collapse onto a few discrete planes, while the two in-plane axes
    are quasi-continuous. So the section axis is the one whose values occupy the
    fewest bins (here 50 bins across each axis' range). Returns
    ``(section_axis, in_plane_axes, plane_mask)`` where ``plane_mask`` selects the
    single densest section.
    """
    occupancy, binned = [], []
    for a in range(3):
        col = coords[:, a]
        lo, hi = float(col.min()), float(col.max())
        res = max((hi - lo) / 50.0, 1e-9)
        b = np.round((col - lo) / res).astype(np.int64)
        binned.append(b)
        occupancy.append(np.unique(b).size)
    saxis = int(np.argmin(occupancy))
    vals, counts = np.unique(binned[saxis], return_counts=True)
    plane_bin = vals[int(np.argmax(counts))]
    plane_mask = binned[saxis] == plane_bin
    in_plane = [a for a in range(3) if a != saxis]
    return saxis, in_plane, plane_mask


def render_error_slice(
    coords: np.ndarray,
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    lipid_name: str,
    out_path: Path,
    axis_labels: Sequence[str] = ("x", "y", "z"),
    dpi: int = 150,
):
    """One MALDI section shown as a reconstruction-error heatmap.

    Scatters the held-out TEST measurements of a single densest section, colored
    by the SIGNED error ``pred - true`` (raw intensity units) on a diverging
    colormap centered at 0 — blue = under-prediction, red = over-prediction. The
    companion middle/right panels show the true and predicted intensity on the
    same section for context (shared sequential scale).

    ``coords`` is ``(N, 3)`` paired row-for-row with ``true_vals`` / ``pred_vals``
    — either physical mm (``*ccf``) or integer voxel indices; the units only set
    the axis ticks. The section axis is inferred from the coordinate geometry.
    """
    coords = np.asarray(coords, dtype=np.float64)
    true_vals = np.asarray(true_vals, dtype=np.float64).ravel()
    pred_vals = np.asarray(pred_vals, dtype=np.float64).ravel()
    finite = (np.isfinite(true_vals) & np.isfinite(pred_vals)
              & np.all(np.isfinite(coords), axis=1))
    coords, true_vals, pred_vals = coords[finite], true_vals[finite], pred_vals[finite]
    if coords.shape[0] == 0:
        logging.warning(f"error-slice: {lipid_name} has no finite paired "
                        f"values; skipping.")
        return

    saxis, in_plane, plane_mask = _pick_section(coords)
    ax0, ax1 = in_plane
    x = coords[plane_mask, ax0]
    y = coords[plane_mask, ax1]
    true_s = true_vals[plane_mask]
    pred_s = pred_vals[plane_mask]
    err = pred_s - true_s
    if err.size == 0:
        logging.warning(f"error-slice: {lipid_name} densest section is empty; "
                        f"skipping.")
        return

    rmse = float(np.sqrt(np.mean(err ** 2)))
    emax = float(np.percentile(np.abs(err), 99)) or float(np.abs(err).max()) or 1.0
    vlo = float(min(true_s.min(), pred_s.min()))
    vhi = float(max(true_s.max(), pred_s.max()))
    lbl = str(axis_labels[saxis]) if saxis < len(axis_labels) else f"axis{saxis}"
    plane_pos = float(np.median(coords[plane_mask, saxis]))

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")
    fig.suptitle(f"Lipid {lipid_name} — error heatmap on densest MALDI section "
                 f"({lbl}≈{plane_pos:.3g}, n={err.size:,}, RMSE={rmse:.3g})",
                 fontsize=13)
    marker = max(4, min(40, int(30000 / max(err.size, 1))))

    def _panel(ax, c, cmap, vmin, vmax, title):
        sc = ax.scatter(x, y, c=c, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=marker, marker="s", linewidths=0)
        ax.set_aspect("equal"); ax.set_title(title)
        ax.set_xlabel(str(axis_labels[ax0]) if ax0 < len(axis_labels) else f"axis{ax0}")
        ax.set_ylabel(str(axis_labels[ax1]) if ax1 < len(axis_labels) else f"axis{ax1}")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    _panel(axs[0], true_s, "inferno", vlo, vhi, "True")
    _panel(axs[1], pred_s, "inferno", vlo, vhi, "Predicted")
    _panel(axs[2], err, "RdBu_r", -emax, emax, "Error (pred − true)")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_selected_lipids(
    template_volume,
    volume_dir: Path,
    output_dir: Path,
    selected_lipids_names: Sequence[str],
    lipid_indices: Iterable[int] | None = None,
    lipid_names: Iterable[str] | None = None,
    suffix: str = "",
    n_rotation_frames: int = 10,
):
    """High-level entry: render a chosen subset of lipid volumes.

    Resolution rules (matches _resolve_lipid_filter):
      - lipid_indices wins if both given
      - else lipid_names
      - else everything in selected_lipids_names

    Output: one PNG per lipid in `output_dir`. Skips lipids whose
    `{lipid_name}_volume{suffix}.npy` file is missing.
    """
    all_names = [str(n) for n in selected_lipids_names]

    if lipid_indices is not None:
        idx = [int(i) for i in lipid_indices]
        names = [all_names[i] for i in idx]
    elif lipid_names is not None:
        names = [str(n) for n in lipid_names]
    else:
        names = list(all_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    anatomy = template_volume

    for lipid_name in tqdm(names, desc="rendering lipid volumes"):
        vol_path = volume_dir / f"{lipid_name}_volume{suffix}.npy"
        out_path = output_dir / f"{lipid_name}_multi_panel.png"
        if not vol_path.exists():
            logging.warning(f"Skipping {lipid_name}: {vol_path} does not exist.")
            continue
        if out_path.exists():
            logging.info(f"Skipping {lipid_name}: render already exists.")
            continue
        try:
            volume = np.load(vol_path)
            render_lipid_volume(
                volume=volume, lipid_name=lipid_name, out_path=out_path,
                anatomy=anatomy, n_rotation_frames=n_rotation_frames,
            )
        except Exception as e:
            logging.error(f"Failed to render {lipid_name}: {e}")
            continue