#!/usr/bin/env python
# encoding: utf-8
"""
toy_box_cylinder.py
===================

Toy geometry: a parallelepiped (box) with a cylinder inside.

Two sampling modes test the core theory:

  "surface" -- points only on the surfaces (6 box faces + cylinder lateral
               surface + 2 caps).  The gap between the cylinder outer wall and
               the facing box wall is the "fold": the two surfaces are
               Euclidean-close but geodesically far (to traverse between them
               you must go around the top or bottom of the structure along
               the surface).  A geodesic signal assigned antiphase to the two
               surfaces creates Euclidean-near / value-opposite pairs that fool
               a Euclidean GP.

  "volume"  -- points fill the box interior uniformly.  The same signal
               extended smoothly in 3D has no such fold.  Euclidean GP works.

Signal kinds:

  "geodesic" -- sin(n_az * θ) on the cylinder surface,
                sin(n_az * θ + π) on the box walls (antiphase).
                For volume mode the phase shifts smoothly from 0 to π as
                r goes from the cylinder radius to the box wall -- smooth 3D.

  "ambient"  -- smooth 3D function sin(2π x / λ); Euclidean GP wins everywhere.

Surface labels for inflate_cross_region_edges:
  LABEL_CYL = 0  (cylinder lateral surface + caps)
  LABEL_BOX = 1  (all six box faces)

Returns a dict with:
  X          (N, 3)  3-D coordinates
  intrinsic  (N, k)  intrinsic / oracle coordinates
                     surface: (unrolled_arc, z)
                     volume:  (x, y, z)  (== X, oracle is Euclidean)
  signal     (N,)    target values
  labels     (N,)    region labels (int64)
  theta      (N,)    azimuthal angle around z-axis
  r_xy       (N,)    radial distance from z-axis
  params     dict    generation parameters
"""
from __future__ import annotations
import numpy as np

LABEL_CYL = 0
LABEL_BOX = 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_box_cylinder(
    n: int = 60_000,
    box_half: float = 4.0,          # cube half-side: box spans [-L, L]^3
    cyl_radius: float = 3.5,        # outer cylinder radius; gap = box_half - cyl_radius
    cyl_half_height: float = 3.0,   # cylinder half-height; spans [-Hc, Hc] along z
    mode: str = "surface",          # "surface" | "volume"
    signal_kind: str = "geodesic",  # "geodesic" | "ambient"
    n_azimuthal_cycles: int = 2,    # full oscillation cycles around the cylinder
    wavelength_gaps: float = 3.0,   # ambient-signal wavelength in units of radial gap
    seed: int = 0,
) -> dict:
    """Generate box+cylinder toy data.

    See module docstring for geometry and signal design.
    """
    rng = np.random.default_rng(seed)
    L, R, Hc = box_half, cyl_radius, cyl_half_height
    gap = L - R   # radial gap between cylinder outer wall and box wall

    if mode == "surface":
        X, labels = _sample_surface(rng, n, L, R, Hc)
    elif mode == "volume":
        X = rng.uniform(-L, L, (n, 3)).astype(np.float64)
        labels = np.full(n, LABEL_BOX, dtype=np.int64)
    else:
        raise ValueError(f"unknown mode {mode!r}; choose 'surface' or 'volume'")

    theta = np.arctan2(X[:, 1], X[:, 0])
    r_xy  = np.hypot(X[:, 0], X[:, 1])
    z     = X[:, 2]

    # ---- signal ------------------------------------------------------------
    if signal_kind == "geodesic":
        if mode == "surface":
            # Cylinder: phase 0.  Box walls: phase π.
            # Adjacent 3D-near points across the gap have opposite values.
            phase = np.where(labels == LABEL_CYL, 0.0, np.pi)
        else:  # volume
            # Phase increases smoothly from 0 (at cylinder) to π (at box wall).
            # No discontinuity → Euclidean GP can handle this.
            t = np.clip((r_xy - R) / gap, 0.0, 1.0)
            phase = np.pi * t
        signal = np.sin(n_azimuthal_cycles * theta + phase)
    elif signal_kind == "ambient":
        lam = wavelength_gaps * gap
        signal = (np.sin(2 * np.pi * X[:, 0] / lam) +
                  np.cos(2 * np.pi * X[:, 2] / lam)) / 2.0
    else:
        raise ValueError(
            f"unknown signal_kind {signal_kind!r}; choose 'geodesic' or 'ambient'")

    # ---- intrinsic coordinates for oracle GP --------------------------------
    if mode == "surface":
        # Each point mapped to (unrolled_arc, z).
        # Cylinder: arc = θ * R  (unrolled cylinder circumference).
        # Box walls: arc = θ * L (box "azimuthal arc" at wall radius).
        # The scale difference between the two separates them in intrinsic space.
        arc = np.where(labels == LABEL_CYL, theta * R, theta * L)
        intr = np.c_[arc, z]
    else:
        intr = X.copy()   # volume oracle = Euclidean

    return dict(
        X=X.astype(np.float64),
        intrinsic=intr.astype(np.float64),
        signal=signal.astype(np.float64),
        labels=labels.astype(np.int64),
        theta=theta,
        r_xy=r_xy,
        params=dict(
            n=n, box_half=L, cyl_radius=R, cyl_half_height=Hc,
            mode=mode, signal_kind=signal_kind,
            n_azimuthal_cycles=n_azimuthal_cycles,
            wavelength_gaps=wavelength_gaps, seed=seed,
        ),
    )


# ---------------------------------------------------------------------------
# Internal sampling helpers
# ---------------------------------------------------------------------------

def _sample_surface(rng: np.random.Generator, n: int,
                    L: float, R: float, Hc: float):
    """Distribute n points proportionally to surface area across all surfaces."""
    area_box     = 6.0 * (2 * L) ** 2
    area_cyl_lat = 2 * np.pi * R * (2 * Hc)
    area_cyl_cap = 2 * np.pi * R ** 2
    total        = area_box + area_cyl_lat + area_cyl_cap

    n_box = max(1, round(n * area_box / total))
    n_lat = max(1, round(n * area_cyl_lat / total))
    n_cap = max(2, n - n_box - n_lat)   # remainder → caps (always even)
    if n_cap % 2:
        n_cap += 1  # keep top/bottom equal

    X_box = _sample_box_faces(rng, L, n_box)
    lbl_box = np.full(n_box, LABEL_BOX, dtype=np.int64)

    theta_lat = rng.uniform(-np.pi, np.pi, n_lat)
    z_lat     = rng.uniform(-Hc, Hc, n_lat)
    X_lat     = np.c_[R * np.cos(theta_lat), R * np.sin(theta_lat), z_lat]
    lbl_lat   = np.full(n_lat, LABEL_CYL, dtype=np.int64)

    X_cap, lbl_cap = _sample_cylinder_caps(rng, R, Hc, n_cap // 2)

    X      = np.vstack([X_box, X_lat, X_cap]).astype(np.float64)
    labels = np.concatenate([lbl_box, lbl_lat, lbl_cap])
    return X, labels


def _sample_box_faces(rng: np.random.Generator, L: float, n: int) -> np.ndarray:
    """Uniform surface sample on the 6 faces of the cube [-L,L]^3."""
    # Face encoding: (fixed_axis, sign)
    faces = [(2, -1), (2, +1), (0, -1), (0, +1), (1, -1), (1, +1)]
    face_idx = rng.integers(0, 6, n)
    u = rng.uniform(-L, L, n)
    v = rng.uniform(-L, L, n)
    X = np.empty((n, 3))
    for fi, (ax, sign) in enumerate(faces):
        m = face_idx == fi
        if not m.any():
            continue
        other = [j for j in range(3) if j != ax]
        X[m, other[0]] = u[m]
        X[m, other[1]] = v[m]
        X[m, ax]       = sign * L
    return X


def _sample_cylinder_caps(rng: np.random.Generator,
                          R: float, Hc: float, n_each: int):
    """Uniform sample on top (+Hc) and bottom (-Hc) disks of the cylinder."""
    def _disk(z_val):
        r = np.sqrt(rng.uniform(0.0, R ** 2, n_each))
        t = rng.uniform(-np.pi, np.pi, n_each)
        return np.c_[r * np.cos(t), r * np.sin(t), np.full(n_each, z_val)]

    X   = np.vstack([_disk(+Hc), _disk(-Hc)])
    lbl = np.full(2 * n_each, LABEL_CYL, dtype=np.int64)
    return X, lbl


# ---------------------------------------------------------------------------
# Diagnostics (analogous to fold_report in toy_manifold.py)
# ---------------------------------------------------------------------------

def fold_report(data: dict, n_src: int = 200, k: int = 10, seed: int = 0) -> dict:
    """Characterise the fold quality of the box+cylinder geometry.

    Reports two quantities that determine whether the Euclidean GP is confused:

    1. Intra-surface signal smoothness: how much does the signal vary within
       the k nearest same-surface neighbours?  Small → signal is smooth per surface.

    2. Cross-surface signal jump: for each source, find the nearest point on the
       OTHER surface and report the signal difference.  Large + small Euclidean
       distance → Euclidean GP is confused; manifold GP should win.
    """
    from scipy.spatial import cKDTree
    rng_d = np.random.default_rng(seed)
    X, labels, signal = data["X"], data["labels"], data["signal"]
    kdt  = cKDTree(X)
    srcs = rng_d.choice(len(X), size=min(n_src, len(X)), replace=False)
    _, nn = kdt.query(X[srcs], k=k + 1)

    intra_jump, cross_dist, cross_sig_jump = [], [], []
    for a, nbrs in zip(srcs, nn):
        same = nbrs[nbrs != a][:k]
        same = same[labels[same] == labels[a]]
        if len(same) > 0:
            intra_jump.append(np.abs(signal[same] - signal[a]).max())

        # Find nearest point on the OTHER surface (only meaningful in surface mode)
        other_mask = labels != labels[a]
        if not other_mask.any():
            continue
        other_idx = np.where(other_mask)[0]
        dists = np.linalg.norm(X[other_idx] - X[a], axis=1)
        closest = other_idx[dists.argmin()]
        cross_dist.append(float(dists.min()))
        cross_sig_jump.append(float(abs(signal[closest] - signal[a])))

    result = dict(
        median_intra_surface_signal_jump = float(np.median(intra_jump)) if intra_jump else float("nan"),
        n_src=n_src, k=k,
    )
    if cross_dist:
        result.update(
            median_cross_surface_dist        = float(np.median(cross_dist)),
            min_cross_surface_dist           = float(np.min(cross_dist)),
            median_cross_surface_signal_jump = float(np.median(cross_sig_jump)),
            p90_cross_surface_signal_jump    = float(np.percentile(cross_sig_jump, 90)),
        )
    return result


if __name__ == "__main__":
    print("=== surface mode ===")
    d = make_box_cylinder(n=20_000, mode="surface", signal_kind="geodesic")
    r = fold_report(d, n_src=100)   # n_src small: brute-force cross-surface search
    print(f"  intra-surface signal smoothness (median jump): "
          f"{r['median_intra_surface_signal_jump']:.3f}")
    print(f"  cross-surface: min dist={r['min_cross_surface_dist']:.3f}  "
          f"median dist={r['median_cross_surface_dist']:.3f}")
    print(f"  cross-surface signal jump: "
          f"median={r['median_cross_surface_signal_jump']:.3f}  "
          f"p90={r['p90_cross_surface_signal_jump']:.3f}")
    print("  --> Euclidean-close pairs with antiphase signal = fold.")

    print("\n=== volume mode ===")
    dv = make_box_cylinder(n=20_000, mode="volume", signal_kind="geodesic")
    rv = fold_report(dv, n_src=100)
    print(f"  intra-surface signal smoothness (median jump): "
          f"{rv['median_intra_surface_signal_jump']:.3f}")
    if "median_cross_surface_dist" in rv:
        print(f"  cross-surface signal jump: median={rv['median_cross_surface_signal_jump']:.3f}")
    else:
        print("  no cross-surface structure (all points have same label in volume mode)")
