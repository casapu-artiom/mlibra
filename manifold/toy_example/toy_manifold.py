#!/usr/bin/env python
# encoding: utf-8
"""
toy_manifold.py
===============

Generate a 2D manifold embedded in 3D that is FOLDED so that geodesically-distant
parts come close in Euclidean space -- the regime where a manifold kernel should
beat a Euclidean one.

Geometry: a "tight swiss roll". A flat 2D sheet (intrinsic coords (s, h), s =
arc-length along the roll, h = height) is rolled up with a small radial GAP per
turn. The surface is developable, so intrinsic Euclidean distance == true
geodesic distance. Adjacent layers of the roll share an angle but differ in s by
~one turn of arc length: they are Euclidean-near but geodesic-far.

The signal is a smooth function of the geodesic coordinate s (a gradient along
the roll), so two Euclidean-close points on different layers carry very different
values -- exactly what fools a Euclidean kernel and what a manifold kernel can
resolve.

Returns a dict with:
  X         (N,3)  3D coordinates
  intrinsic (N,2)  (s, h) -- intrinsic coords; euclidean here == geodesic
  s         (N,)   geodesic arc-length coordinate along the roll
  signal    (N,)   value function f(s)  (smooth along the manifold)
  t,r       (N,)   roll angle parameter and radius (for plotting)
  total_arclength, params

Helpers: geodesic_from(intrinsic, i), euclidean_from(X, i),
manifold_kernel_cosine(...) for visualization.

Pure numpy/scipy/sklearn.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import cumulative_trapezoid

def make_folded_manifold(n=6000, n_turns=3.5, gap=0.25, r0=1.0, height=6.0,
                         cycles_per_turn=1.5, thickness=0.0, seed=0,
                         signal_kind="geodesic", wavelength_gaps=3.0):
    """Tight swiss roll. Smaller `gap` => tighter fold => geodesic >> euclidean
    between adjacent layers.
 
    `cycles_per_turn` sets the signal frequency PER TURN (not per unit arc
    length). A half-integer value (0.5, 1.5, 2.5, ...) makes radially-adjacent
    layers exactly ANTIPHASE -- i.e. a point and its euclidean-near / geodesic-far
    partner one turn away carry OPPOSITE values -- because sin(c*(t+2pi)) flips
    sign whenever c has fractional part 0.5. This holds regardless of how the
    turn arc length grows with radius, which a signal tied to raw arc length does
    NOT (there, ~1 cycle/turn makes adjacent layers accidentally in-phase)."""
    rng = np.random.default_rng(seed)
    Tmax = n_turns * 2 * np.pi
 
    # arc-length lookup s(t) on a fine grid (developable surface)
    tg = np.linspace(0.0, Tmax, 8000)
    rg = r0 + gap * tg / (2 * np.pi)
    drdt = gap / (2 * np.pi)
    sg = cumulative_trapezoid(np.sqrt(rg ** 2 + drdt ** 2), tg, initial=0.0)
 
    t = rng.uniform(0.0, Tmax, n)
    h = rng.uniform(0.0, height, n)
    r = r0 + gap * t / (2 * np.pi)
    X = np.c_[r * np.cos(t), h, r * np.sin(t)].astype(np.float64)
    if thickness > 0:
        X += rng.normal(0.0, thickness, X.shape)
 
    s = np.interp(t, tg, sg)                       # geodesic coordinate
    intrinsic = np.c_[s, h]                         # intrinsic == geodesic metric
    total_s = float(sg[-1])

    # ---- signal: what the target field is a function of --------------------
    # 'geodesic'   : f(arclength t)         -> smooth ALONG the sheet  => manifold wins
    # 'ambient_*'  : f(3D ambient position) -> smooth in space         => euclidean wins
    # 'radial'     : f(distance from roll axis)                         => euclidean wins
    # Pair an ambient/radial signal with the DENSE regime (thickness >= gap) for
    # the clearest euclidean-favourable case. Note the coordinate layout:
    # X[:,1] is HEIGHT; the spiral plane is (X[:,0], X[:,2]); the radius from the
    # roll axis is hypot(X[:,0], X[:,2]).
    if signal_kind == "euclidean":          # backward-compat alias
        signal_kind = "ambient_grid"
    px, pz = X[:, 0], X[:, 2]
    lam = wavelength_gaps * gap             # wavelength a few gaps wide: smooth across a
                                            # layer but oscillating across the structure
    if signal_kind == "geodesic":
        # turn-locked: smooth along the geodesic, antiphase across adjacent layers
        signal = np.sin(cycles_per_turn * t)
    elif signal_kind == "ambient_x":
        signal = np.sin(2 * np.pi * px / lam)
    elif signal_kind == "ambient_grid":
        signal = np.sin(2 * np.pi * px / lam) + np.cos(2 * np.pi * pz / lam)
    elif signal_kind == "radial":
        signal = np.sin(2 * np.pi * np.hypot(px, pz) / lam)
    else:
        raise ValueError(
            f"unknown signal_kind {signal_kind!r}; choose "
            f"geodesic | ambient_x | ambient_grid | radial (or 'euclidean' alias)")
 
    return dict(
        X=X, intrinsic=intrinsic, s=s, signal=signal, t=t, r=r,
        total_arclength=total_s,
        params=dict(n=n, n_turns=n_turns, gap=gap, r0=r0, height=height,
                    cycles_per_turn=cycles_per_turn, thickness=thickness, seed=seed,
                    signal_kind=signal_kind, wavelength_gaps=wavelength_gaps),
    )


def geodesic_from(intrinsic, i):
    """True geodesic distance from point i (euclidean in intrinsic coords)."""
    return np.linalg.norm(intrinsic - intrinsic[i], axis=1)


def euclidean_from(X, i):
    return np.linalg.norm(X - X[i], axis=1)


def manifold_kernel_cosine(intrinsic, i, lengthscale):
    """A ground-truth 'manifold kernel' from point i: an RBF in the TRUE geodesic
    metric (intrinsic euclidean). This is what a perfect manifold kernel encodes;
    use it to visualize the kernel-from-source on the manifold."""
    g = geodesic_from(intrinsic, i)
    return np.exp(-0.5 * (g / lengthscale) ** 2)


def euclidean_kernel_from(X, i, lengthscale):
    e = euclidean_from(X, i)
    return np.exp(-0.5 * (e / lengthscale) ** 2)


def fold_report(data, n_src=200, k=10, seed=0):
    """Quantify the fold: for random sources, how geodesically far are the
    EUCLIDEAN-nearest neighbors? A big gap = a real fold (geodesic != euclidean)."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    X, intr = data["X"], data["intrinsic"]
    kdt = cKDTree(X)
    srcs = rng.choice(len(X), size=min(n_src, len(X)), replace=False)
    _, nn = kdt.query(X[srcs], k=k + 1)            # euclidean nearest neighbors
    geo_to_euc_nn, sig_jump = [], []
    for a, nbrs in zip(srcs, nn):
        nbrs = nbrs[nbrs != a][:k]
        g = np.linalg.norm(intr[nbrs] - intr[a], axis=1)
        geo_to_euc_nn.append(g.max())
        sig_jump.append(np.abs(data["signal"][nbrs] - data["signal"][a]).max())
    return dict(
        median_geodist_of_euclidean_nn=float(np.median(geo_to_euc_nn)),
        p90_geodist_of_euclidean_nn=float(np.percentile(geo_to_euc_nn, 90)),
        median_signal_jump_to_euclidean_nn=float(np.median(sig_jump)),
        local_euclidean_nn_spacing=float(np.median(
            [np.linalg.norm(X[nn[j][1]] - X[srcs[j]]) for j in range(len(srcs))])),
    )


def preview_png(data, path="toy_manifold_preview.png", src=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    X, intr, sig = data["X"], data["intrinsic"], data["signal"]
    fig = plt.figure(figsize=(14, 4.5))
    ax1 = fig.add_subplot(131, projection="3d")
    p = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=sig, cmap="coolwarm", s=4)
    ax1.set_title("manifold in 3D, colored by signal f(s)")
    fig.colorbar(p, ax=ax1, shrink=0.6)
    ax2 = fig.add_subplot(132)
    ax2.scatter(intr[:, 0], intr[:, 1], c=sig, cmap="coolwarm", s=4)
    ax2.set_title("unrolled (intrinsic s,h) -- the true geometry")
    ax2.set_xlabel("s (geodesic)"); ax2.set_ylabel("h")
    ax3 = fig.add_subplot(133, projection="3d")
    if src is None:
        src = int(np.argmin(np.linalg.norm(X - X.mean(0), axis=1)))
    g = geodesic_from(intr, src); e = euclidean_from(X, src)
    # points euclidean-close but geodesic-far (the fold) in red
    close_far = (e < np.percentile(e, 8)) & (g > np.percentile(g, 60))
    ax3.scatter(X[:, 0], X[:, 1], X[:, 2], c="lightgray", s=3)
    ax3.scatter(*X[src], c="black", s=80, marker="*", label="source")
    ax3.scatter(X[close_far, 0], X[close_far, 1], X[close_far, 2],
                c="red", s=8, label="euclid-near, geodesic-far")
    ax3.set_title("the fold: red = close in 3D, far on manifold")
    ax3.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


if __name__ == "__main__":
    data = make_folded_manifold()
    print("total arclength:", round(data["total_arclength"], 2),
          "| N:", len(data["X"]))
    rep = fold_report(data)
    for k, v in rep.items():
        print(f"  {k}: {v:.3f}")
    png = preview_png(data)
    print("preview:", png)