# SOTA 3D-reconstruction models for MALDI

Implementations of recent spatial-omics **3D reconstruction** methods, adapted to
the MALDI lipid dataset. All are launched from the single `MODEL=… ./local_run/run_sota.sh`
entry point (see below). `ntf`/`spa3d` plug into the *same* harness the manifold-GP
runs use (`baselines/experiment_baselines.py`) — whole-brain renders, `metrics.csv`,
diagnostics comparable to `run_manifold` / `run_baseline`.

| `--model` | Paper | File | Core mechanism |
|-----------|-------|------|----------------|
| `ntf` | Neural Transcriptomic Field (bioRxiv 2026.05.28.726140) | [`ntf_model.py`](ntf_model.py) | Multiresolution hash-grid INR + heteroscedastic loss + smoothness + PSF averaging + low-freq bias net + per-slice variance |
| `spa3d` | Spatial-pattern-enhanced GCN, *Brief. Bioinform.* bbag060 | [`spa3d_model.py`](spa3d_model.py) | Spatial Pattern Enhancement (Hilbert / ALFT) + z-aware 3D GCN |
| `deepspatial` | DeepSpatial (bioRxiv 2026.04.28.721395) | [`deepspatial_transport/`](deepspatial_transport/) | **Faithful** — official GiT flow-matching + UOT + probability-flow ODE, within-specimen slice interpolation |
| `gplfr` | GP Latent Factor Regression (arXiv:2606.06576) | [`gplfr/`](gplfr/) + [`gplfr_experiment.py`](gplfr_experiment.py) | Latent GP (`BASE_GP`=euclidean/riemann/spectral) + analytically-marginalized linear decoder, on the MaldiExperiment harness |

`ntf` and `spa3d` are **pure PyTorch** (hash grid, GCN via `torch_geometric`
which the `gcn` baseline already needs) — no `tinycudann` / CUDA compile. They
plug into the regression harness. The other two run their own drivers (still from
`run_sota.sh`): `deepspatial` runs the official `deepspatial==1.0.0` package
(Lightning / POT / torchdiffeq / anndata) as within-specimen slice interpolation
(see [`deepspatial_transport/`](deepspatial_transport/)); `gplfr` runs
[`gplfr_experiment.py`](gplfr_experiment.py) on the `MaldiExperiment` harness (the
riemann/spectral bases need the eigenpair pipeline).

## How the papers map onto MALDI

All three papers reconstruct a continuous 3D molecular field from sparse 2D
sections. For `ntf` / `spa3d` the MALDI task is posed as a conditional
regression `predict(coords) -> lipids` over the dense voxel grid (no discrete
cells/cell-types). Adaptation notes per model:

- **NTF** is a near-direct fit: it is already a coordinate→expression field.
  "Section identity" = coronal slice, binned from the (standardized) **`xccf`**
  sectioning axis. Faithful to the official `models.py` (GYQ-form/NTF), it includes:
  a **PSF** Monte-Carlo average over `--ntf-psf-samples` coordinate
  perturbations (models the capture volume) applied at train time only, so
  reconstruction queries the **deblurred** field; a **bias network** on the
  low-frequency hash levels + a shared slice embedding (`--ntf-levels-bias`,
  additive here vs. the paper's multiplicative log-bias because MALDI targets
  are z-scored); and a heteroscedastic variance from a latent-z `sigma_net`
  **plus a per-slice `log_var_slice`** (`--ntf-features-z`). The bias is
  **dropped at reconstruction** (clean field). Zero-inflation BCE head is off by
  default (MALDI intensities are continuous, not zero-inflated).
- **Spa3D** contributes (1) per-section Spatial Pattern Enhancement — the
  analytic-signal envelope (Hilbert) or anti-leakage Fourier transform (ALFT)
  denoise, applied on a rasterized `(x,y)` grid per section — and (2) a GCN over
  a graph whose metric scales the true inter-slice distance along the **`xccf`**
  sectioning axis (`--spa3d-z-weight`). SPE smooths the training target only.
- **DeepSpatial** is cell-resolution/generative and is **not** shoehorned into
  the regression harness. It runs the official `deepspatial==1.0.0` package as
  the paper intends — **within-specimen slice interpolation**: UOT correspondence
  between real coronal sections, a GiT transformer flow-matching velocity field
  over the multi-modal state (in-plane position + lipids + atlas region), and
  probability-flow ODE synthesis of the tissue *between* measured sections.
  Trained on the train-fold mice, it reconstructs each held-out test mouse's full
  brain volume (per-lipid volumes + renders) plus leave-one-section-out metrics.
  Launch with `MODEL=deepspatial ./local_run/run_sota.sh` (delegates) — see
  [`deepspatial_transport/`](deepspatial_transport/).

## Runner scripts

All runners live in `sota/` and mirror `run_baseline.sh`. Every input/output
path is an env var **defaulting to the local paths**; the submit script only
overrides those env vars (see below). They default to `RECONSTRUCT=whole_brain`,
which produces the composite renders **and** the per-lipid true-vs-pred
scatterplots + value-distribution diagnostics — parity with `run_manifold`.

- [`local_run/run_sota.sh`](run_sota.sh) — the main runner; `MODEL` selects the method.
- [`local_run/run_ntf.sh`](run_ntf.sh), [`local_run/run_spa3d.sh`](run_spa3d.sh),
  [`local_run/run_deepspatial.sh`](run_deepspatial.sh) — thin per-method wrappers
  (just pin `MODEL`).

```sh
# per-method wrappers
N_EPOCHS=30 ./local_run/run_ntf.sh
SPA3D_SPE=alft SPA3D_Z_WEIGHT=0.5 BATCH_SIZE=4096 ./local_run/run_spa3d.sh
N_EPOCHS=60 ./local_run/run_deepspatial.sh

# or the MODEL-switch runner, with W&B logging on
MODEL=ntf N_EPOCHS=30 BATCH_SIZE=16384 WANDB=1 ./local_run/run_sota.sh
```

Overridable I/O env vars (local defaults): `DATA_PATH`, `OUTPUT_DIR`,
`MALDI_FILE`, `REFERENCE_FILE`, `ANNOTATION_FILE`, `SLICES_DATASET_FILE`,
`AVAILABLE_LIPIDS_FILE`, `RECONSTRUCTION_LIPIDS_FILE`, `TEMPLATE_NAME`,
`SRC_PATH`, `EXP_PREFIX`, `RECONSTRUCT`.

## Logging (W&B + progress)

The `ntf` / `spa3d` harness models log to **Weights & Biases** when enabled
(`WANDB=1`, or `--wandb` on the driver; `WANDB_PROJECT` sets the project, default
`sota_maldi`). The driver opens the run and each model logs per-epoch metrics —
`<model>/train_*`, `<model>/val_mse`, `<model>/test_mse`, plus NTF's `train_nll`
/ `tv_reg` / `bias_reg` / `logvar_mean` — and a throttled per-step loss. With no
run active, logging is a safe no-op. `WANDB_MODE=offline` records locally without
a login. (The faithful DeepSpatial uses the official package's own logging.)

Progress bars are **nested**: an inner per-batch `tqdm` (rich postfix — NTF shows
`loss/nll/tv/bias`, Spa3D the running MSE) under a per-epoch `logging.info`
summary line.

## Early stopping

`ntf` and `spa3d` use **best-checkpoint early stopping**: a validation split is
carved from **train** (`VAL_FRAC`, default 0.05 — the held-out test mice are
never used for selection), and the best-val weights are restored before
reconstruction (never ship the last epoch, which for a high-capacity field on
the cross-mouse folds is usually the worst). `EARLY_STOP_PATIENCE` (default 5)
also terminates early after that many epochs without val improvement; set
`EARLY_STOP_MONITOR=test` to select on the held-out-mouse metric instead.

## Cluster submission

[`submit/run_sota_batch.sh`](../submit/run_sota_batch.sh) submits one run:ai job
per `(model, fold)`, mirroring `run_submit_baselines.sh`. Because the runner
already specifies all I/O with local defaults, the batch script simply overrides
the input/output env vars (`-e DATA_PATH=… -e OUTPUT_DIR=… …`) to the S3-mounted
dirs.

```sh
./submit/run_sota_batch.sh                     # all 3 models, fold-3
MODELS="ntf" FOLDS="fold-1 fold-2" ./submit/run_sota_batch.sh
DRY_RUN=1 ./submit/run_sota_batch.sh           # print the runai commands only
```

## Direct invocation / outputs

```sh
python sota/run_sota.py --mode lgp --model ntf \
    --dataset-path ... --maldi-file ... --available-lipids-file ... \
    --reference-file ... --output-dir ... --template-name reference \
    --slices-dataset-file maldi/data/splits/fold_2.json \
    --exp-name FOLD-2-SOTA-NTF --epochs 30 --batch-size 16384 \
    --reconstruct whole_brain
```

Outputs land in `<output-dir>/<exp-name>/` with the standard layout:
`metrics.csv` (per-lipid held-out R²/corr/rmse/mae), `train/` + `test/`
predictions & coordinates, `model.pth`, `config.json`, and — with
`--reconstruct whole_brain` — `volume/` (per-lipid volumes) and `renders/`
(composite renders + per-lipid scatter/distribution diagnostics).

Set `RECONSTRUCT=none` (or `--reconstruct none`) to skip the expensive
whole-brain voxel pass (fit + metrics only). `--skip-training` re-runs
reconstruction from a saved `model.pth`.
