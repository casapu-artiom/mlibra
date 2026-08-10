# parcelgp

Reference-only parcellation geometry, and whether it is worth feeding to a GP.

Self-contained: numpy / scipy / scikit-learn / pandas only. Shares no code with
`manifold_gp` — no graph Laplacian, no eigensolve, no Riemann kernel, no faiss.
The coordinate conventions in `volume.py` are deliberately identical to the ones
the MALDI parquet is read with, so the two frames line up (verified: median
nearest-node error 0.047 mm against a 0.1 mm node grid).

## Pipeline

```
reference_image.npy
  -> features.template_features    multi-scale appearance + Hessian shape + depth
  -> parcellate.parcellate         bilaterally symmetric, contiguous parcels
                                   + sparse soft memberships
  -> border.border_distance        distance from each node to its parcel border
  -> field.ParcelField             cached .npz, queryable at any coordinate
```

```bash
python -m other_experiments.parcelgp.build --reference-file .../reference_image.npy \
    --out parcels/full_k128.npz --n-parcels 128 --features full --spatial-weight 3.0

python -m other_experiments.parcelgp.validate \
    --field full=parcels/full_k128.npz --field spatial=parcels/spatial_k128.npz \
    --atlas-file .../level_15annot.npy \
    --maldi-file .../maindata_minimal.parquet \
    --available-lipids-file .../maindata_minimal_available_lipids.npy \
    --reference-file .../reference_image.npy \
    --slices-dataset-file maldi/data/splits/fold_2.json
```

A build is ~10 s for 531k nodes at stride 4; a full validation over 173 lipids and
74 sections is ~4 min.

## Looking at it

Two viewers. Both build/reuse the field with the same knobs `run_parcel_per_lipid.sh` uses,
so all three scripts always refer to the same parcellation.

```bash
./other_experiments/parcelgp/view_parcels.sh                                   # napari
MONTAGE=/tmp/parcels.png ./other_experiments/parcelgp/view_parcels.sh          # headless PNG grid
STATS=1 ./other_experiments/parcelgp/view_parcels.sh                           # per-parcel table

./other_experiments/parcelgp/parcel_vs_euclidean_explorer.sh
RUN=<run dir> RUN_LIPID="PA 36:1" ./other_experiments/parcelgp/parcel_vs_euclidean_explorer.sh
DUMP=/tmp/k.png SECTION=9 ./other_experiments/parcelgp/parcel_vs_euclidean_explorer.sh
```

**`view_parcels`** — the reference volume in grey with the parcel ids on top,
either as a napari `Labels` layer (hover to read the parcel under the cursor;
extra layers for the border shell, the top-1 membership and `d_border_rel`) or as
a PNG grid of sections along any axis. This is the check `validate.py` cannot
make: whether the parcels are contiguous blobs that follow anatomy.

**`parcel_vs_euclidean_explorer`** — one sample's MALDI voxels, snapped to the
field's nodes and laid out flat per section. Click a voxel and the section is
drawn twice: left `k_base(test, ·)`, right the same kernel times the parcel
factor (or the factor alone, in [0, 1], or the signed difference). Parcel borders
are drawn on both panels, and the measured lipid is a toggleable layer, so the
question "does the right panel stop at a border, and does the lipid actually
change there?" is answered by looking. The text panel carries the numbers:
effective neighbourhood size for both kernels, mean covariance within vs across a
border, and each kernel's k-weighted SD of the lipid (how chemically mixed the
tissue it smooths over is).

`B` comes from one of two places. By default `B = strength · I`, so the embedding
is the membership vector itself and the picture is the partition's own geometry —
nothing fitted to lipids, and `strength=0` is an exact no-op. With `RUN=` /
`RUN_LIPID=` it is instead the `B`, ARD lengthscale and outputscale that lipid's
run actually learned, read out of `checkpoints/batch_*.pt`, so both panels
reproduce the deployed model. Point `RUN` at the *baseline* arm and the
checkpoint has no `B` at all; the tool says so and both panels match.

## The three checks

`validate.py` is the gate, and it is deliberately hard to fool.

1. **boundary_contrast** — mean |Δlipid| for adjacent measured points that cross a
   parcel border, over the same quantity for pairs that don't. Distance is
   controlled by construction (both are one step apart).
2. **border_trend** (`near/far`) — within-parcel pairs only, binned by depth into
   the parcel. Decides whether a distance-to-border feature carries anything: it
   does only if lipids vary faster near a border, i.e. `near/far > 1`.
3. **parcel_ev** — variance explained by the parcel mean, **held out by section**,
   because the model exists to fill gaps between measured sections.

Two methodology points that changed the answers, both the hard way:

* Checks 1–2 **must** be run on the pooled 0.1 mm node grid, not on raw
  acquisition voxels. The registration lookup has a median error of 47 µm against
  a ~25 µm pixel pitch, so at pixel scale the cross/within split is a coin flip
  exactly at the borders that matter; a single voxel also carries ~19% pure
  measurement noise. Run at pixel scale, every parcellation scores contrast ≈ 1.0,
  including the Allen atlas. Pooled onto nodes, the atlas returns 1.26 —
  reproducing the 1.22 measured previously by an independent route.
* Bootstrap over **spatial blocks**, not pairs. Adjacent pairs overlap in their
  endpoints; a pair-level bootstrap gives intervals several times too narrow.

## Results (fold_2 train sections, 173 lipids, 4.97M voxels → 267k pooled nodes)

| parcellation      |   K | contrast | ci95        |    EV | near/far |
|-------------------|-----|----------|-------------|-------|----------|
| atlas (level_15)  |  70 |  1.268   | [1.20,1.34] | 0.141 |    –     |
| simple_k32        |  32 |  1.215   | [1.18,1.24] | 0.196 |  1.145   |
| simple_k64        |  64 |  1.206   | [1.16,1.23] | 0.229 |  1.006   |
| full_k64          |  64 |  1.120   | [1.09,1.14] | 0.229 |  1.005   |
| spatial_k128      | 128 |  1.007   | [1.00,1.02] | 0.230 |  0.911   |
| simple_k128       | 128 |  1.118   | [1.08,1.15] | 0.247 |  0.912   |
| **full_k128**     | 128 |  1.126   | [1.08,1.16] | **0.259** | 1.034 |
| full_k256         | 256 |  1.097   | [1.06,1.13] | 0.284 |  0.958   |

`simple` = single-scale intensity/gradient/local-std (reproduces the previous
template clustering). `full` = the multi-scale + Hessian + depth stack.
`spatial` = geometry only, the "compactness alone" control.

**Cluster belonging works.** At matched K = 128, EV is 0.259 (full) vs 0.141 for
the Allen atlas at K = 70 — roughly 1.8×. The control matters: spatial-only
already reaches 0.230, so *appearance* contributes about +0.03 EV, and the rest is
"more, smaller, compact regions". EV keeps climbing with K (0.229 → 0.259 → 0.284
for K = 64 → 128 → 256), as it must, so only matched-K comparisons mean anything.

**Distance-to-border does not.** `near/far` scatters between 0.91 and 1.15 with no
pattern across K, spatial weight or feature set. Lipids do not vary faster near a
template-derived parcel border, so a non-stationary lengthscale or a border-depth
input column has nothing to key on. Not implemented, by design.

**Borders are the weak part everywhere.** No template parcellation beats the
atlas's boundary contrast (1.268), and none comes near the 1.5 bar. Note the
tension between checks: the configurations with the best contrast (`simple`, low
K) have the worst EV, and vice versa. The signal is in *which parcel you are in*,
not in *where its edge is* — consistent with the manifold-side finding that the
usable structure lives in the lowest ~10–50 Laplacian modes, i.e. region identity.

## Running the model

`kernels.py` multiplies any existing `ScaleKernel` by a learned parcel-similarity
factor `exp(-‖z(x)-z(x')‖²/2)`, `z(x) = m(x)ᵀB`. The partition is reference-only;
only `B` is learned, per lipid. `B → 0` reverts exactly to the base kernel, so the
ablation is honest.

```bash
./local_run/run_parcel_per_lipid.sh                    # baseline + parcel, back to back
LIMIT=4 EPOCHS=1 ./local_run/run_parcel_per_lipid.sh   # smoke test
MODE=parcel N_PARCELS=256 ./local_run/run_parcel_per_lipid.sh
```

It builds/caches the parcel field on first use and runs **both arms by default** —
a parcel run alone tells you nothing, the only question is whether it beats the
identical model without the factor. Both arms share every hyperparameter, seed and
split; the sole difference is `--parcel-field`.

Watch `parcel/offdiag_mean` (logged to W&B): the mean covariance multiplier between
different parcels. It starts near 1 and falls only if the model actually learned to
stop smoothing at borders. If it stays at 1, the factor was learned away.

The same flags exist directly on `maldi/lgp_experiment_per_lipid.py`
(`--parcel-field`, `--parcel-rank`, `--parcel-init-scale`, `--parcel-shared-B`) if
you'd rather drive it from your own script.

**Gotcha:** the per-lipid runner defaults to `--nu 2.0`, which is only valid for
the Riemann kernel; `MaternKernel` accepts 0.5/1.5/2.5 only and the euclidean
family crashes at the default. `run_parcel_per_lipid.sh` passes 2.5.

## What this implies for the model

Use memberships, not borders. `ParcelField.dense_memberships(coords)` gives the
partition-of-unity a coregionalization factor `⟨m(x), Σ m(x')⟩` (Σ learnable, PSD,
low-rank) would consume, and the same vector can widen `LinearMean` into a
per-parcel latent offset. `d_border_rel` is still computed and stored — it is
cheap and the diagnostic is worth keeping — but nothing should be built on it
without a parcellation that first clears check 2.
