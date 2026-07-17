# DeepSpatial — faithful transport mode (MALDI)

Runs the **official** [`deepspatial`](https://github.com/yyh030806/DeepSpatial)
package (v1.0.0, PyPI) on MALDI as the paper actually intends: **within-specimen
slice interpolation**. This is the sole DeepSpatial implementation — an earlier
harness-plugged stand-in that reframed it as noise→data conditional regression
(to fit the per-voxel eval) was removed in favour of the real method here.
Launch via `MODEL=deepspatial ./sota/run_sota.sh` (which delegates here) or
`./sota/deepspatial_transport/run_deepspatial_transport.sh`. The method:

1. **UOT** establishes cross-section cell (voxel) correspondences,
2. a **GiT** transformer learns a flow-matching velocity field over the
   multi-modal state (in-plane position + lipids + region),
3. a **probability-flow ODE** synthesizes the tissue **between** measured
   sections along the sectioning axis.

## MALDI ↔ DeepSpatial mapping

| DeepSpatial (single-cell ST) | MALDI |
|------------------------------|-------|
| cell | measured voxel |
| slice / section | physical coronal `Section` |
| in-plane XY (`obsm['spatial']`) | `(yccf, zccf)` — the two non-section axes |
| depth / z (`obs['z_coord']`) | **`xccf`** — the sectioning (AP) axis |
| gene expression (`.X`) | lipid intensities |
| cell type (`obs['cell_class']`) | Allen atlas region (`level_15annot`, ~65–70) |

Key data fact: the **sectioning axis is `xccf` (coords column 0)**, not `zccf` —
per-section mean `xccf` marches monotonically while y/z stay ~constant. (Fixing
this same assumption in NTF/Spa3D's section logic was done alongside this work.)

## Experimental setup (keeps the fold split)

- **Train** on the TRAIN-fold mice (each carries its own section stack). Multiple
  mice are handled by building **within-mouse** adjacent-section UOT pairs with a
  **shared global** atlas-region label set — no pair ever crosses a mouse
  boundary ([`run_deepspatial_transport.py`](run_deepspatial_transport.py)
  `build_trajectories`).
- **Reconstruct** the full brain volume of each held-out TEST-fold mouse by
  ODE-integrating between its adjacent sections, rasterize the synthesized cells
  onto the CCF grid, and **render** per-lipid volumes with the same renderer the
  manifold / run_sota runs use (`render_lipid_volumes.render_selected_lipids`).
- **Metrics**: leave-one-section-out interpolation on the test mice — drop an
  interior section, reconstruct its gap, match synthesized cells to the held
  section by nearest in-plane position, score per-lipid corr/RMSE → `metrics.csv`.

## MALDI-specific adaptations (documented, necessary)

- **Subsampling for UOT.** MALDI sections carry 40–75k voxels; the official UOT
  builds a full N₀×N₁ coupling, infeasible at that size (the paper's ST slices
  are far sparser). Sections are randomly subsampled to `--ds-max-cells` (train)
  / `--ds-max-cells-recon` (reconstruction) before the coupling.
- **`--ds-thickness` is in `xccf` (mm) units** — the inter-plane spacing;
  `target_cells ≈ n_sec · gap / thickness`. MALDI gaps are ~0.15–2 mm, so the
  default `0.02` fills the volume (vs. the paper's µm-scale default).

## Dependencies (heavy — isolated)

`deepspatial==1.0.0` pulls in Lightning, POT, torchdiffeq, timm,
scanpy/anndata, pyvista. These are **deliberately kept out** of the main
`requirements.txt` / `Dockerfile` (the pure-torch container for NTF / Spa3D / the
manifold GP). Use [`../../Dockerfile.deepspatial`](../../Dockerfile.deepspatial)
+ [`requirements-deepspatial.txt`](requirements-deepspatial.txt) for this
experiment only.

## Usage

```sh
# local (inside the deepspatial env / image); all I/O env vars default to local
N_EPOCHS=100 DS_HIDDEN_SIZE=256 DS_DEPTH=6 \
  ./sota/deepspatial_transport/run_deepspatial_transport.sh

# denser / coarser volume: smaller/larger thickness
DS_THICKNESS=0.01 DS_MAX_CELLS_RECON=8000 ./sota/deepspatial_transport/run_deepspatial_transport.sh
```

Overridable I/O env vars (local defaults): `DATA_PATH`, `OUTPUT_DIR`,
`MALDI_FILE`, `REFERENCE_FILE`, `ANNOTATION_FILE`, `SLICES_DATASET_FILE`,
`AVAILABLE_LIPIDS_FILE`, `RECONSTRUCTION_LIPIDS_FILE`, `TEMPLATE_NAME`,
`EXP_PREFIX`, `RECONSTRUCT`, plus the `DS_*` knobs and `WANDB`.

Cluster: [`../../submit/run_deepspatial_transport.sh`](../../submit/run_deepspatial_transport.sh)
(runai, uses `Dockerfile.deepspatial`).

## Outputs

`<output-dir>/<exp-name>/`:
- `checkpoints/` — Lightning checkpoints + `config.json`,
- `metrics.csv` — per-lipid leave-one-section-out corr/RMSE,
- `volume/<lipid>_volume_<mouse>.npy` — rasterized reconstructed volumes,
- `renders/<mouse>/<lipid>_multi_panel.png` — imaging renders per test mouse.

## Caveats

- The current fold split is **cross-mouse**; DeepSpatial transports **within** a
  mouse, so the test-mouse reconstruction is a genuine within-specimen
  interpolation using a model trained on other mice's dynamics. LOSO metrics need
  a test mouse with ≥3 sections.
- Reconstruction density scales with `n_sec / thickness`; tune `--ds-thickness`
  and `--ds-max-cells-recon` for coverage vs. compute.
