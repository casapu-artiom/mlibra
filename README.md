# mLibra — Latent Gaussian Process Models for Spatial Lipidomics

> **mLibra** is a research codebase for modelling the spatial distribution of lipids in the mouse brain using Latent Gaussian Process (LGP) models applied to MALDI-MSI (Matrix-Assisted Laser Desorption/Ionisation Mass Spectrometry Imaging) data. The repository ships both a reusable modelling library (`l3di`) and a complete experiment framework (`maldi/`) for cross-validation, whole-brain reconstruction, and visualisation.

---

## Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Repository Structure](#3-repository-structure)
4. [The `l3di` Library](#4-the-l3di-library)
   - 4.1 [Core GP Model — `lgp.py`](#41-core-gp-model--lgppy)
   - 4.2 [Mixture-of-Experts Decoder — `lgpmoe.py`](#42-mixture-of-experts-decoder--lgpmopy)
   - 4.3 [Multimodal Extension — `lgpmultimodal.py`](#43-multimodal-extension--lgpmultimodalpy)
   - 4.4 [Voxel Diffusion — `lgp_vdiffusion.py`](#44-voxel-diffusion--lgp_vdiffusionpy)
   - 4.5 [Simple 1-D DDPM — `simple_ddpm1d.py`](#45-simple-1-d-ddpm--simple_ddpm1dpy)
5. [The `maldi/` Experiment Module](#5-the-maldi-experiment-module)
   - 5.1 [Data Format](#51-data-format)
   - 5.2 [Cross-Validation Splits](#52-cross-validation-splits)
   - 5.3 [Configuration — `config.py`](#53-configuration--configpy)
   - 5.4 [Experiment Orchestrator — `experiment.py`](#54-experiment-orchestrator--experimentpy)
   - 5.5 [Entry Points](#55-entry-points)
   - 5.6 [Utilities — `utils.py`](#56-utilities--utilspy)
   - 5.7 [Whole-Brain Volumetric Reconstruction — `voxel_diffusion.py`](#57-whole-brain-volumetric-reconstruction--voxel_diffusionpy)
   - 5.8 [Visualisation — `visualisations.py`](#58-visualisation--visualisationspy)
   - 5.9 [CV Result Collection — `collect-cv-lgp.py`](#59-cv-result-collection--collect-cv-lgppy)
   - 5.10 [Output Directory Layout](#510-output-directory-layout)
6. [The `multimodal/` Extension](#6-the-multimodal-extension)
7. [The `manifold/` Extension — Riemann Manifold GP](#7-the-manifold-extension--riemann-manifold-gp)
   - 7.1 [What It Changes](#71-what-it-changes)
   - 7.2 [Module Layout](#72-module-layout)
   - 7.3 [Installation](#73-installation)
   - 7.4 [Running Locally](#74-running-locally)
   - 7.5 [Cluster Submission](#75-cluster-submission)
   - 7.6 [Geometry Caches](#76-geometry-caches)
   - 7.7 [Output Layout](#77-output-layout)
   - 7.8 [Tooling](#78-tooling)
8. [Shell Scripts & Use Cases](#8-shell-scripts--use-cases)
   - 8.1 [`deploy.sh` — Code Sync](#81-deploysh--code-sync)
   - 8.2 [`run_all.sh` — Single LGP Fold Run](#82-run_allsh--single-lgp-fold-run)
   - 8.3 [`run_all_moe.sh` — Single LGPMOE Fold Run](#83-run_all_moesh--single-lgpmoe-fold-run)
   - 8.4 [`run_final.sh` — Full-Data Final Model](#84-run_finalsh--full-data-final-model)
   - 8.5 [`do_cv.sh` — Local CV Grid](#85-do_cvsh--local-cv-grid)
   - 8.6 [`submit_cv.sh` — Cluster CV Grid](#86-submit_cvsh--cluster-cv-grid)
9. [Cross-Validation Protocol](#9-cross-validation-protocol)
10. [Experiment Naming Convention](#10-experiment-naming-convention)
11. [Installation](#11-installation)
12. [Running an Experiment](#12-running-an-experiment)
13. [Testing](#13-testing)
14. [Infrastructure & Deployment](#14-infrastructure--deployment)
15. [Dependencies](#15-dependencies)
16. [The EUCLID Baseline](#16-the-euclid-baseline)

---

## 1. Background & Motivation

MALDI-MSI allows the simultaneous measurement of hundreds of lipid species at thousands of spatial locations across a tissue section. Each pixel carries a high-dimensional lipid spectrum, and each section is one 2-D slice of a 3-D mouse brain registered to the Allen Common Coordinate Framework (CCF). The scientific goal is to learn a continuous spatial model of the lipidome — one that:

- generalises across samples and brain regions,
- accounts for the high dimensionality of the lipid measurements,
- can reconstruct unseen sections (held-out brain slices from different biological replicates),
- and can generate whole-brain volumetric maps of any lipid species.

The core modelling idea is a **Latent Gaussian Process (LGP)**: a sparse, variational GP is placed over a low-dimensional latent space indexed by 3-D brain coordinates, and an MLP decoder maps the latent representation back to the full lipid spectrum. This is analogous to a spatial latent variable model where the GP provides smooth, spatially-aware priors over the latent codes rather than a fixed isotropic prior.

---

## 2. High-Level Architecture

```
3-D CCF coordinates (x, y, z)
         │
         ▼
┌────────────────────────────────┐
│  Sparse Variational GP         │  ← IndependentMultitaskGPModel
│  (d latent dims, M inducing)   │    Kernel: RBF / Matern / Symmetric
└────────────┬───────────────────┘
             │  latent mean z ∈ R^d
             ▼
┌────────────────────────────────┐
│  MLP Decoder                   │  ← shared, MoE, or per-modality
│  (d → hidden → p lipids)       │
└────────────┬───────────────────┘
             │  x̂ ∈ R^p
             ▼
    Reconstruction / NLL loss + β · KL(q(u) ‖ p(u))
```

For the **diffusion extension**, a trained LGP is used as a VAE-like encoder; its latent mean and decoded output condition a 1-D DDPM UNet that refines predictions at inference time.

---

## 3. Repository Structure

```
mlibra/
├── l3di/                        # Core modelling library (installable package)
│   ├── __init__.py
│   ├── lgp.py                   # LGP, GPVAE, IndependentMultitaskGPModel, Custom3DKernel
│   ├── lgpmoe.py                # Mixture-of-Experts decoder variant
│   ├── lgpmultimodal.py         # Multimodal (MALDI + gene expression) LGP
│   ├── lgp_vdiffusion.py        # LGP-conditioned Voxel Diffusion (DDPM)
│   ├── simple_ddpm1d.py         # Standalone 1-D DDPM (development/reference)
│   └── example.py               # Minimal usage example
│
├── maldi/                       # MALDI-MSI experiment application
│   ├── config.py                # MaldiConfig dataclass, argument parsing helpers
│   ├── experiment.py            # MaldiExperiment: train / predict / save loop
│   ├── lgp_experiment.py        # CLI entry point for LGP model
│   ├── lgpmoe_experiment.py     # CLI entry point for LGPMOE model
│   ├── utils.py                 # Inducing point generation (KMeans + symmetry)
│   ├── visualisations.py        # Napari-based 3-D brain volume viewer
│   ├── voxel_diffusion.py       # Whole-brain reconstruction with diffusion (notebook)
│   ├── collect-cv-lgp.py        # Aggregates CV metrics across folds (notebook)
│   ├── collect-cv.py            # Earlier CV collection script
│   ├── example.py               # Minimal maldi usage example
│   ├── reference.py             # Reference image / CCF utilities
│   ├── requirements.txt         # maldi-specific pip requirements
│   ├── data/
│   │   └── splits/              # Cross-validation fold definitions (JSON)
│   │       ├── fold_1.json … fold_8.json
│   │       └── all.json         # All samples as training (for final model)
│   ├── mouse_connectivity/      # Allen CCF reference data (25 µm NRRD + manifest)
│   ├── output_images/           # Example visualisation outputs
│   ├── run_all.sh               # Worker script: single LGP fold
│   ├── run_all_moe.sh           # Worker script: single LGPMOE fold
│   ├── run_final.sh             # Final model on all data
│   ├── do_cv.sh                 # Run full CV grid locally
│   ├── submit_cv.sh             # Submit full CV grid to RunAI cluster
│   └── deploy.sh                # Rsync source to remote compute node
│
├── multimodal/                  # Extension: joint MALDI + gene expression
│   ├── config.py                # MultiModalConfig
│   ├── experiment.py            # MultiModalExperiment
│   ├── multimodallgp_experiment.py  # CLI entry point
│   └── utils.py
│
├── manifold/                    # Extension: Riemann manifold (graph-Laplacian) GP
│   ├── lgp_manifold.py          # ManifoldLGP / SpectralManifoldLGP + their latent GPs
│   ├── lgp_manifold_experiment.py           # CLI entry point (inducing points)
│   ├── spectral_lgp_manifold_experiment.py  # CLI entry point (weight space)
│   ├── manifold_gp/             # Vendored IMGP package (kernels, operators, utils)
│   ├── slepc/, docker/          # MPI eigensolver; training/notebook images
│   ├── benchmarks/, visualizations/, notebooks/, toy_example/
│   └── README.md                # Full documentation for this module
│
├── baselines/                   # Non-GP baselines, same I/O contract as the GP runs
│   ├── experiment_baselines.py  # mean / linear / xgboost / mlp / mlp_eigen / gcn / euclid
│   └── euclid_kernel.py         # Runs EUCLID's anatomical_interpolation unmodified
│
├── sota/                        # Published 3-D reconstruction comparisons (NTF, Spa3D, ...)
├── reports/                     # CV aggregation (lgp_report.py) + figures + notebooks
│
├── euclid/                      # Clone of github.com/lamanno-epfl/EUCLID (untracked)
│                                #   supplies the source that euclid_kernel.py executes,
│                                #   plus its 100 um reference/annotation volumes
│
├── local_run/                   # In-container worker scripts (env -> CLI -> python)
├── submit/                      # RunAI submission sweeps + pod helpers
│
├── tests/                       # Pytest test suite
│   ├── conftest.py              # Shared fixtures (device, mock data)
│   ├── test_infrastructure.py   # Fixture sanity checks
│   ├── test_lgp_gpmodel.py      # GP model initialisation and forward pass
│   └── test_lgp_kernel.py       # Custom3DKernel properties
│
├── Dockerfile                   # Jupyter-based container for Renku/RunAI
├── pyproject.toml               # l3di package build config (setuptools)
├── setup.cfg
├── requirements.txt             # Top-level pip requirements
├── CONVENTIONS.md               # Coding standards enforced in this project
├── MAP.md                       # Auto-generated file tree
├── generate_map.py              # Script to regenerate MAP.md
├── progress.txt                 # Task tracking log
├── PRD.json                     # Product requirements document
└── REFLECT.md                   # Retrospective notes
```

---

## 4. The `l3di` Library

`l3di` (Latent 3-D Inference) is a standalone Python package that contains all model definitions. It is designed to be modality-agnostic; the `maldi/` folder is simply one application that imports from it.

Install locally in editable mode:

```bash
pip install -e .
```

### 4.1 Core GP Model — `lgp.py`

This file contains three classes.

#### `Custom3DKernel`

A GPyTorch-compatible kernel designed for 3-D brain coordinates. It wraps a Matern kernel with **coronal-plane symmetry**: before computing covariances the z-coordinate (medial-lateral axis) is reflected to its absolute value, encoding the assumption that the left and right hemispheres of the mouse brain are symmetric. The lengthscale can be lower-bounded (`minimal_length_scale`) to prevent the GP from fitting at sub-voxel resolution.

```python
kernel = Custom3DKernel(nu=2.5, minimal_length_scale=0.1)
```

#### `IndependentMultitaskGPModel`

A sparse variational approximate GP (inheriting `gpytorch.models.ApproximateGP`) with `d` independent tasks. Each task has its own variational distribution (`CholeskyVariationalDistribution`) and shares inducing point locations (which are learned). Three kernel types are supported:

| `kernel_type` | Description |
|---|---|
| `"rbf"` | Squared-exponential (RBF) with ARD per spatial dimension |
| `"matern"` | Matern with configurable smoothness `nu` (0.5, 1.5, 2.5) |
| `"symmetric"` | `Custom3DKernel` — Matern with coronal-plane symmetry |

The inducing points are initialised from a KMeans clustering of brain CCF coordinates (see `utils.py`) and are then jointly optimised with the model parameters.

The forward pass returns a `MultitaskMultivariateNormal` distribution; the mean has shape `(batch, d)`.

#### `LGP` (Latent Gaussian Process)

The main model. A GP defines a prior over the `d`-dimensional latent space; an MLP decoder maps latent codes to the `p`-dimensional lipid measurement.

**Architecture:**
- Input: 3-D CCF coordinate `(x, y, z)`
- GP posterior → latent mean `z ∈ R^d`
- MLP decoder: `z → [hidden layers] → x̂ ∈ R^p`
- Observation noise `log_var_n ∈ R^p` (learnable, per-lipid)

**Loss function:**
```
L = NLL(x, x̂, log_var_n) + β · KL(q(u) ‖ p(u))
```
where `KL(q(u) ‖ p(u))` is the KL divergence from the variational GP posterior to the GP prior, computed analytically via `variational_strategy.kl_divergence()`.

**Training interface:**
```python
lgp.train_model(exp_path, dataloader, optimizer, epochs, current_epoch)
```
Each batch provides `(lipid_values, 3d_coordinates)`. Metrics (loss, reconstruction loss, KL, MSE) are logged to **Weights & Biases** (`wandb`).

**Prediction interface:**
```python
x_reconstructed, gp_posterior = lgp.predict(coords)
```

#### `GPVAE` (Gaussian Process Variational Autoencoder)

An alternative formulation where an MLP encoder maps lipid values to a latent mean and variance, while the GP acts as a structured prior over the latent space. The KL divergence is computed between the MLP posterior `q(z|x)` and the GP prior `p(z|coords)`. This class exists for experimentation; the primary model used in production experiments is `LGP`.

---

### 4.2 Mixture-of-Experts Decoder — `lgpmoe.py`

**`LGPMOE`** replaces the single MLP decoder of `LGP` with a **soft Mixture of Experts**. This is biologically motivated: different lipid classes are expected to have distinct spatial patterns and may benefit from specialised decoder branches.

**Expert assignment matrix `A` (shape `p × num_experts`):**

The expert matrix encodes biochemical lipid class membership as a sparse binary prior. The assignment is constructed in `lgpmoe_experiment.py::create_mask()` using lipid name prefixes:

| Expert index | Lipid class | Prefixes |
|---|---|---|
| 0 | Glycerophospholipids | `PC`, `PE`, `PG`, `PA`, `PS`, `PI` |
| 1 | Lysophospholipids | `LPC`, `LPE`, `LPA`, `LPS`, `LPG` |
| 2 | Glycosphingolipids | `HexCer`, `Hex2Cer` |
| 3 | Simple sphingolipids | `SM`, `Cer` |
| 4 | Neutral storage lipids | `TG`, `DG` |
| 5+ | Residual | all others |

**Forward pass:**
1. GP posterior → latent mean `z ∈ R^d`
2. Gate network: `z → softmax → global_gate ∈ R^{num_experts}` (soft routing)
3. Per-feature weighting: `feature_weights[i, e] ∝ global_gate[e] · A[i, e]` (combines global gate with lipid-class prior)
4. Output: weighted sum of expert predictions

This allows different regions of the brain to route lipid predictions through different expert networks while respecting the biochemical class structure.

---

### 4.3 Multimodal Extension — `lgpmultimodal.py`

**`LGPMultiModal`** combines two modalities (e.g., MALDI lipids and gene expression) under a shared spatial GP framework.

**Architecture:**
- Two independent sparse GPs, one per modality
- Both GPs take the same 3-D CCF coordinate as input
- Latent representations are fused via **bidirectional cross-attention** (`MultiModalCrossAttention`)
- Two separate MLP decoders produce reconstructions for each modality

**Cross-attention module:**
- `CrossAttention`: standard scaled dot-product multi-head attention
- `MultiModalCrossAttention`: applies cross-attention in both directions (modality 1 → 2 and 2 → 1), then applies layer normalisation and feed-forward sub-layers (transformer-style)
- The fused latent codes `(attended_z1, attended_z2)` are passed to separate decoders

**Loss function:**
```
L = α · NLL_mod1 + (1-α) · NLL_mod2 + β · (KL_GP1 + KL_GP2)
```
where `α` balances the reconstruction losses between modalities.

---

### 4.4 Voxel Diffusion — `lgp_vdiffusion.py`

A **conditional DDPM** (Denoising Diffusion Probabilistic Model) adapted for 1-D lipidomic spectra. This model is used as a second stage after an LGP has been trained, to refine and diversify predictions through iterative denoising.

**Key design differences from standard DDPM:**

1. **Conditioning**: the model is always conditional. It receives two conditioning signals:
   - `low_res` (shape `[B, 1, L]`): the LGP's decoded mean prediction (serves as a low-resolution anchor)
   - `z` (shape `[B, d]`): the LGP's latent mean (spatial context)

2. **Modified forward process**: noisy input incorporates the conditioning signal:
   ```
   x_t = x_0 · sqrt(ᾱ_t) + low_res + ε · sqrt(1 - ᾱ_t)
   ```
   (standard DDPM omits the `low_res` term)

3. **Modified posterior mean**: the denoising step includes conditioning:
   ```
   μ_post = c1 · x̂_0 + c2 · x_t + c3 · low_res
   ```

4. **Classifier-free guidance**: supported via `guidance_weight > 0`, which interpolates between conditional and unconditional (zero-conditioned) model outputs.

#### `LGPVoxelUNet`

The noise-prediction backbone. A lightweight 1-D UNet with:
- Conv1d encoder (×2 downsampling stages)
- Bottleneck
- ConvTranspose1d decoder (×2 upsampling stages)
- Time embedding (scalar `t` → `model_channels`-dim MLP)
- Z embedding (`z` → `model_channels`-dim MLP)
- Embeddings added to features after the initial convolution

#### `LGPVoxelDiffusion`

The DDPM wrapper around `LGPVoxelUNet`. Handles:
- Noise schedule registration (`β_1` to `β_T`, `T=1000` by default)
- Pre-computed diffusion constants (`α`, `ᾱ`, `sqrt(ᾱ)`, etc.)
- Forward pass for training: `compute_noisy_input` then predict ε
- Reverse sampling loop with optional guidance
- Model save/load

---

### 4.5 Simple 1-D DDPM — `simple_ddpm1d.py`

A nearly-identical standalone implementation of the conditional DDPM (`SimpleDDPM1D`) and its UNet backbone (`SimpleUNet1D`). This module was developed independently for prototyping and shares the same interface as `lgp_vdiffusion.py`. It is used in `voxel_diffusion.py` for the whole-brain reconstruction notebook.

The two implementations (`lgp_vdiffusion.py` vs `simple_ddpm1d.py`) are functionally equivalent; `lgp_vdiffusion.py` is the version integrated into the production pipeline under the `LGPVoxelDiffusion` class.

---

## 5. The `maldi/` Experiment Module

The `maldi/` directory is a self-contained experiment application that uses `l3di` to run spatial lipidomics experiments on MALDI-MSI data from mouse brain sections.

### 5.1 Data Format

The primary data source is a **Parquet file** (`maindata_minimal.parquet`). Each row is one MALDI pixel and contains:

| Column group | Columns | Description |
|---|---|---|
| Identifiers | `Sample`, `Section` | Biological sample name and section index |
| Pixel coordinates | `x`, `y` | In-section pixel position |
| CCF coordinates | `xccf`, `yccf`, `zccf` | Registration to Allen CCF (mm, 3-D) |
| Lipid intensities | `{lipid_name}` (×hundreds) | Raw MALDI signal per lipid species |

A companion `.npy` file (`maindata_minimal_available_lipids.npy`) lists the names of all available lipid channels in the order they appear in the Parquet file.

A **reference image** (`reference_image.npy`) is a 3-D binary mask of the Allen CCF volume at a given resolution; it is used to generate inducing points that cover the brain interior.

### 5.2 Cross-Validation Splits

Splits are stored under `maldi/data/splits/` as JSON files. Each file defines which `Sample` values belong to the training set, the test set, and the ignored set.

**Example — `fold_1.json`:**
```json
{
    "train": [
        ["Sample", "==", "ReferenceAtlas"],
        ["Sample", "==", "SecondAtlas"],
        ["Sample", "==", "Female3"],
        ["Sample", "==", "Male1"],
        ["Sample", "==", "Male2"],
        ["Sample", "==", "Male3"]
    ],
    "test": [
        ["Sample", "==", "Female1"],
        ["Sample", "==", "Female2"]
    ],
    "ignore": [
        ["Sample", "==", "GBA1"],
        ["Sample", "==", "Pregnant1"],
        ...
    ]
}
```

These filters are applied directly to `pd.read_parquet(..., filters=...)` via PyArrow's predicate pushdown, so only the relevant rows are loaded into memory. There are **8 folds** defined. The CV runs in `submit_cv.sh` / `do_cv.sh` use folds 1–5. The `all.json` split assigns all non-ignored samples to training (used for the final model).

The held-out samples in each fold are always from different biological replicates than the training samples, so the evaluation genuinely tests generalisation across individuals.

### 5.3 Configuration — `config.py`

**`MaldiConfig`** is a frozen `dataclass` that stores all experiment parameters. It is created from CLI arguments via `MaldiConfig.from_args(args)` which also:

- creates the output directory tree (`exp_path/checkpoints/`, `exp_path/train/`, `exp_path/test/`, `exp_path/volume/`, optionally `exp_path/diffusion_checkpoints/`, `exp_path/volume_diffusion/`)
- saves `args.npy` to the experiment folder for reproducibility
- appends `{n_pixels}` and optionally `_log` to the experiment name

Key fields:

| Field | Type | Description |
|---|---|---|
| `mode` | `str` | `"lgp"`, `"lgpmoe"`, `"test"`, etc. |
| `kernel` | `str` | `"rbf"`, `"matern"`, `"symmetric"` |
| `latent_dim` | `int` | Dimensionality of the GP latent space |
| `num_inducing` | `int` | Number of sparse GP inducing points |
| `epochs` | `int` | Training epochs |
| `batch_size` | `int` | Mini-batch size |
| `learning_rate` | `float` | AdamW learning rate |
| `log_transform` | `bool` | Apply `log(x + 1e-10)` before training |
| `nu` | `float` | Matern smoothness parameter |
| `n_pixels` | `int` | Minimum resolution in voxels (used to compute `minimal_length_scale`) |
| `use_diffusion` | `bool` | Enable diffusion pipeline |
| `section_filter` | `list` | PyArrow-style filter for training rows |
| `test_filter` | `list` | PyArrow-style filter for test rows |
| `selected_lipids_names` | `list[str]` | Names of selected lipid channels |
| `device` | `str` | `"cuda"` if available, else `"cpu"` |

### 5.4 Experiment Orchestrator — `experiment.py`

**`MaldiExperiment`** is the central class that manages data loading, training, prediction, and evaluation. Its `run()` method implements the full pipeline:

```
run()
 ├── If model.pth exists: load weights (skip training)
 ├── Else:
 │    ├── load_train_sections()      # load section IDs
 │    ├── load_train_coordinates()   # load 3-D CCF coords, normalise
 │    ├── load_train_samples()       # load sample IDs
 │    ├── load_train_data()          # load lipid intensities, log-transform, z-score
 │    ├── load_checkpoint()          # resume from latest checkpoint if present
 │    ├── wandb.init(...)
 │    └── train_fit()                # call model.train_model(...)
 ├── predict training set → train/predictions.npy, train/true_values.npy
 ├── predict test set     → test/predictions.npy,  test/true_values.npy
 ├── scatter_comparison() → per-lipid scatter plots (train and test)
 └── whole_brain_reconstruction() [if model.pth present]
      ├── load Allen CCF template volume
      ├── iterate all non-zero CCF voxels in batches
      └── save per-batch lipid predictions → volume/batch_{i}.pth
```

**Data preprocessing pipeline (inside `load_train_data`):**

1. Load lipid intensities from Parquet for the training filter
2. Replace negative values with 0 (MALDI artefact correction)
3. Optional log transform: `log(x + 1e-10)`
4. Z-score normalise per lipid: `(x - μ) / σ` where statistics are computed once and cached to disk
5. Load CCF coordinates and normalise by coordinate mean/std (cached)
6. Build `TensorDataset(lipids_normalised, coordinates_normalised)`

**Prediction post-processing:**

Predictions are saved in normalised space and also converted back to original scale by reversing the z-score and (if applicable) the log transform: `exp(x̂_norm · σ + μ) - 1e-10`.

**Scatter comparison (`scatter_comparison`):**

Generates per-lipid scatter plots of true vs predicted intensities for both train and test sets. Pearson correlation is annotated on each plot and an overall `{dataset}_correlations.npy` is saved.

**Whole-brain reconstruction (`whole_brain_reconstruction`):**

Uses the Allen SDK (`allensdk.core.mouse_connectivity_cache`) to download/load the 25 µm CCF reference template volume. All non-zero voxels are converted to CCF coordinates in mm, normalised, and fed through the trained LGP in batches. Predictions for each lipid species are assembled into a full volumetric array and saved as `.npy`.

### 5.5 Entry Points

#### `lgp_experiment.py` — Standard LGP

```bash
python maldi/lgp_experiment.py \
  --mode lgp \
  --exp-name MY_EXPERIMENT \
  --dataset-path /path/to/data/ \
  --maldi-file /path/to/maindata_minimal.parquet \
  --available-lipids-file /path/to/available_lipids.npy \
  --output-dir /path/to/output/ \
  --slices-dataset-file maldi/data/splits/fold_1.json \
  --num-inducing 500 \
  --latent-dim 10 \
  --kernel symmetric \
  --epochs 20 \
  --batch-size 2000 \
  --learning-rate 0.001 \
  --seed 42 \
  --n-pixels 10 \
  --device cuda \
  --log-transform \
  [--use-diffusion]
```

| Flag | Effect |
|---|---|
| `--kernel symmetric` | Use coronal-plane-symmetric Matern kernel |
| `--kernel matern` | Standard Matern with configurable `--nu` |
| `--kernel rbf` | RBF kernel |
| `--log-transform` | Apply `log(x+1e-10)` before training |
| `--use-diffusion` | Enable diffusion pipeline (creates `diffusion_checkpoints/` and `volume_diffusion/` dirs) |
| `--n-pixels N` | Sets `minimal_length_scale = N * 0.025 / mean(coord_std)` |
| `--num-inducing M` | Number of sparse GP pseudo-inputs (must be even; symmetric placement) |
| `--load-args` | Load args from previously saved `args.npy` instead of CLI |

#### `lgpmoe_experiment.py` — Mixture of Experts LGP

Same flags as above plus:

| Flag | Effect |
|---|---|
| `--num-experts K` | Number of expert networks in the MoE decoder |

The expert-to-lipid assignment matrix is constructed automatically from lipid name prefixes (see [Section 4.2](#42-mixture-of-experts-decoder--lgpmopy)).

### 5.6 Utilities — `utils.py`

#### `get_inducing_points(exp_path, dataset_path, num_inducing)`

Generates inducing points for the sparse GP. Results are cached to disk (`inducing_points_{M}.pth`).

**Algorithm:**
1. Load `reference_image.npy` (3-D binary CCF mask at 40 voxels/mm)
2. Extract non-zero voxel indices, convert to mm
3. Z-score normalise coordinates; cache `colmean.pth` and `colstd.pth`
4. Extract the left hemisphere (`x <= median_x`)
5. Run **MiniBatch KMeans** with `M/2` clusters on left-hemisphere points
6. Mirror the `M/2` centroids across the coronal midplane to generate `M/2` symmetric right-hemisphere points
7. Return `M` inducing points, with left and right hemispheres represented equally

This symmetric placement is paired with the `Custom3DKernel` (which also enforces coronal symmetry) to produce a model that respects bilateral symmetry of the mouse brain.

### 5.7 Whole-Brain Volumetric Reconstruction — `voxel_diffusion.py`

This Jupytext-formatted Python script (runnable as a notebook via `jupytext`) demonstrates the two-stage reconstruction pipeline:

**Stage 1 — LGP reconstruction (standard):**
The trained LGP predicts lipid intensities at every voxel of the CCF atlas. This is already handled by `experiment.whole_brain_reconstruction()`.

**Stage 2 — Diffusion refinement:**
A `SimpleDDPM1D` is trained on top of the LGP-encoded training data:
- The LGP acts as the VAE: `encode(coord)` → `(μ, logvar)`, `reparametrize`, `decode` → `cond`
- The diffusion model learns to denoise lipid spectra conditioned on `(cond, z)`
- At inference, the denoising chain is run over the entire CCF volume

**Reconstruction at atlas scale:**
- Allen CCF template volume downloaded via `allensdk` (25 µm resolution, shape ~456×528×320)
- Non-zero voxels extracted (~4 million voxels)
- Processed in batches; each batch stored as `volume_diffusion/batch_{i}.pth` with coordinates, CCF indices, and predictions
- Individual lipid volumes assembled from batch files and saved as `{lipid_name}_volume_diffusion.npy` and `{lipid_name}_volume255_diffusion.npy` (normalised to [0, 255])

### 5.8 Visualisation — `visualisations.py`

A standalone napari-based visualisation script for 3-D lipid volumes. For each input volume (`.npy` file):

1. Loads the 3-D array and normalises to [0, 1]
2. Opens a **napari viewer** in volumetric rendering mode (`mip`)
3. Captures a 4-row multi-panel figure:
   - Row 1: 10 **coronal** slices (X-Y plane, stepping along Y)
   - Row 2: 10 **axial** slices (X-Z plane, stepping along Z)
   - Row 3: 10 **sagittal** slices (Y-Z plane, stepping along X)
   - Row 4: 10 **3-D rotation frames** (36° increments, full 360°)
4. Saves a composite PNG (`output_images/{name}_multi_panel.png`)
5. Optionally generates a **rotation video** (`output_images/{name}_brain_rotation.mp4`) using `napari-animation`

### 5.9 CV Result Collection — `collect-cv-lgp.py`

A Jupytext notebook that aggregates and analyses cross-validation results from multiple fold directories. It:

1. Scans the output directory for folders matching `CV-FOLD-{fold}-{latent}-{inducing}-{batch}_log`
2. Loads `test/predictions.npy` and `test/true_values.npy` for each fold
3. Computes per-lipid metrics: **MSE**, **MAPE** (Mean Absolute Percentage Error), **MRE** (Mean Relative Error), **Pearson correlation**
4. Builds a tidy DataFrame indexed by `(FOLD, LATENT, INDUCING, BATCH_SIZE)`
5. Generates seaborn boxplots for each metric vs each hyperparameter, coloured by fold
6. Prints summary tables sorted by mean MSE, MRE, and correlation

This allows visual and numerical identification of the best hyperparameter configuration.

### 5.10 Output Directory Layout

For an experiment named `CV-FOLD-1-10-500-200010_log`:

```
{OUTPUT_DIR}/CV-FOLD-1-10-500-200010_log/
├── args.npy                         # Saved CLI arguments
├── model.pth                        # Final trained model weights
├── colmean.pth                      # CCF coordinate means (shared across runs)
├── colstd.pth                       # CCF coordinate standard deviations
├── inducing_points_500.pth          # KMeans-derived GP inducing points
├── labels_500.pth                   # KMeans cluster labels
├── reference_image_500.pth          # Symmetric reference image tensor
├── train_means.pth                  # Per-lipid training set means
├── train_stds.pth                   # Per-lipid training set standard deviations
├── checkpoints/
│   ├── model_0.pth
│   ├── model_1.pth
│   └── ...
├── train/
│   ├── predictions.npy              # Lipid predictions on training pixels
│   ├── true_values.npy              # True lipid values on training pixels
│   └── scatter_{lipid}.png          # Per-lipid scatter plots
├── test/
│   ├── predictions.npy              # Lipid predictions on held-out test pixels
│   ├── true_values.npy              # True lipid values on test pixels
│   └── scatter_{lipid}.png
├── volume/
│   ├── template_volume.npy          # Allen CCF template (cached)
│   ├── batch_0.pth … batch_N.pth   # Per-batch volumetric predictions
│   └── {lipid}_volume.npy           # Full 3-D volume per lipid
├── diffusion_checkpoints/           # [if --use-diffusion]
│   └── model_{epoch}.pth
└── volume_diffusion/                # [if --use-diffusion]
    ├── template_volume.npy
    ├── batch_0.pth … batch_N.pth
    ├── {lipid}_volume_diffusion.npy
    └── {lipid}_volume255_diffusion.npy
```

---

## 6. The `multimodal/` Extension

The `multimodal/` directory extends the single-modality MALDI experiment to jointly model **MALDI lipid data and gene expression data** from spatially registered measurements.

It mirrors the structure of `maldi/`:
- `config.py`: `MultiModalConfig` (adds `gene_file`, `selected_genes_names`, `alpha` balance parameter)
- `experiment.py`: `MultiModalExperiment` — handles two data modalities, trains `LGPMultiModal`, and saves per-modality predictions and scatter plots
- `multimodallgp_experiment.py`: CLI entry point
- `utils.py`: shared helper functions

The multimodal model uses `LGPMultiModal` from `l3di/lgpmultimodal.py` (see [Section 4.3](#43-multimodal-extension--lgpmultimodalpy)).

---

## 7. The `manifold/` Extension — Riemann Manifold GP

The `manifold/` directory replaces the Euclidean kernel of the baseline LGP with a **graph-Laplacian Riemann–Matérn kernel** built on the brain's own geometry, so covariance is measured along tissue rather than straight through it. Everything downstream — data loading, CV splits, the MLP decoder, prediction, whole-brain reconstruction — is shared with `maldi/`, which makes the two directly comparable on the same folds and metrics.

> **Full documentation:** [`manifold/README.md`](manifold/README.md). This chapter is the installation / submission quick start.

### 7.1 What It Changes

Instead of `k(x, y) = Matérn(‖x − y‖)`, the manifold path:

1. Discretises the CCF reference template into a graph — nodes are voxels above `--threshold`, subsampled by `--stride`.
2. Connects each node to its `k` nearest neighbours with **FAISS**, so edges follow tissue and not air.
3. Optionally biases the graph by anatomy: edges crossing an Allen atlas region boundary get their length inflated (soft prior, `--cross-region-inflation`) or are removed outright (hard prune, `--prune-cross-region`).
4. Forms the **graph Laplacian** and solves for its lowest `--num-modes` eigenpairs `(λ_k, φ_k)`.
5. Defines the kernel as a Matérn spectral filter over those eigenpairs — `k(x, y) = Σ_k S(λ_k) φ_k(x) φ_k(y)` — with a Nyström bump extension for queries that are not exact graph nodes.
6. Plugs that kernel into the same latent-GP-plus-MLP-decoder stack. Because an MLP sits between the GP latent and the observations, training uses a Monte-Carlo ELBO (`rsample` + decode) rather than `gpytorch.VariationalELBO`, plus MAP-II hyperparameter priors.

Two model stacks are provided: `LatentRiemannGP` + `ManifoldLGP` (inducing points) and `SpectralLatentGP` + `SpectralManifoldLGP` (weight space — no inducing points, diagonal posterior directly over the spectrum). They share every graph/eigenpair cache, so running both separates "does the geometry help?" from "does the inducing approximation hurt?".

### 7.2 Module Layout

```
manifold/
├── lgp_manifold.py                      # LatentRiemannGP, ManifoldLGP,
│                                        #   SpectralLatentGP, SpectralManifoldLGP
├── lgp_manifold_experiment.py           # ENTRY POINT — inducing-point run
├── spectral_lgp_manifold_experiment.py  # ENTRY POINT — weight-space (spectral) run
├── manifold_kernel_builder.py           # Shared graph → Laplacian → eigenpairs → kernel
├── compute_eigenvectors.py              # Offline GPU (cupy) eigensolve into the shared cache
├── check_eigencache.py                  # Which eigenvector caches exist / are missing
├── model_hyperparams.py                 # Dump trained hyperparameters from checkpoints
├── download_bg_atlas.py                 # Fetch + depth-coarsen the Allen/BrainGlobe annotation
├── manifold_gp/                         # Vendored IMGP package (kernels, operators, utils)
├── slepc/                               # MPI/SLEPc eigensolver + PETSc build script
├── docker/                              # Training / SLEPc / notebook images + entrypoint
├── benchmarks/                          # Bump support, Nyström, bandwidth, spectral sweeps
├── visualizations/                      # napari manifold-vs-Euclidean explorer + its engine
├── notebooks/                           # Narrative notebooks (boundary story, distance study)
└── toy_example/                         # Synthetic manifolds (swiss roll, box + cylinder)
```

`manifold/` has no `__init__.py` — it is a script directory. Only `manifold/manifold_gp` is installed, as the **top-level** module `manifold_gp` (it is a fork of [Implicit Manifold GP Regression](https://arxiv.org/abs/2310.19390), so it keeps the upstream name).

Three further working directories — `viz/` (the napari explorer fleet), `experiments/` (one-off validation probes) and `analysis/` (figure notebooks and their cache builders) — are `.gitignore`d and absent from a clone. Nothing in the training or submission path depends on them; see [`manifold/README.md`](manifold/README.md) for what they contain.

### 7.3 Installation

```bash
python -m venv .venv && source .venv/bin/activate

# Installs BOTH l3di and the vendored manifold_gp — pyproject.toml lists
# `manifold` as a second package-discovery root for exactly this reason.
pip install -e .
pip install -r requirements.txt

# FAISS is NOT in requirements.txt (the container compiles it from source with
# GPU support). Locally, install a wheel:
pip install faiss-gpu-cu12          # or: pip install faiss-cpu

# Optional — napari explorers under manifold/visualizations and manifold/toy_example
pip install napari napari-animation

# Optional — only for manifold/slepc/slepc_eigensolve.py (needs gfortran + MPI)
PREFIX=/opt WITH_MUMPS=1 bash manifold/slepc/build_slepc_petsc.sh
export PETSC_DIR=/opt/petsc SLEPC_DIR=/opt/slepc
export LD_LIBRARY_PATH=/opt/petsc/lib:/opt/slepc/lib:$LD_LIBRARY_PATH
```

Do **not** `pip install manifold-gp` from PyPI — it would shadow the fork.

Beyond the baseline data files, the atlas-weighted graph methods need an annotation volume (`level_15annot.npy`, `level_5annot.npy`, or a `ccf_depth{N}annot.npy` produced by `manifold/download_bg_atlas.py --max-depth N`).

**Docker** — build from the repo root, since the `COPY` paths are repo-relative:

```bash
docker build -f manifold/docker/Dockerfile -t artiomartiom/sdsc:maldi_manifold_all_latest .
```

The image is `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel` plus `requirements.txt`, FAISS built from source with GPU support, PETSc + SLEPc + MUMPS, an sshd on port 2222, and a dual-mode entrypoint: **no arguments** starts sshd for interactive/VS Code Remote work, **with arguments** it activates the venv, drops to `appuser` via `gosu`, and execs the command. Every boot clones/updates the repo at `/myhome/mlibra` and `pip install -e`s it.

### 7.4 Running Locally

The recommended path is `local_run/run_manifold.sh` — the *same* script the cluster executes, with every knob defaulted as an environment variable. It composes `EXP_NAME` from all swept hyperparameters (so two configs can never share an output directory), translates env vars into CLI flags, and calls the Python entry point:

```bash
DATA_PATH=/path/to/mlibra_data \
OUTPUT_DIR=/path/to/output \
EIGENVECTOR_DIR=/path/to/output/eigenvectors \
SLICES_DATASET_FILE=$PWD/maldi/data/splits/fold_2.json \
STRIDE=4 NUM_MODES=300 KNN_K=15 \
KNN_METHOD=faiss_atlas_weighted CROSS_REGION_INFLATION=50 \
GRAPHBANDWIDTH=0.05 LAPLACIAN_NORM=randomwalk \
N_EPOCHS=30 BATCH_SIZE=1000 LATENT_DIM=5 \
  ./local_run/run_manifold.sh
```

The spectral twin is `local_run/run_spectral_manifold.sh`. Or call Python directly for a smoke run:

```bash
python manifold/lgp_manifold_experiment.py \
  --mode lgp --exp-name manifold_smoke \
  --dataset-path /path/to/mlibra_data \
  --maldi-file /path/to/maindata_minimal.parquet \
  --available-lipids-file /path/to/maindata_minimal_available_lipids.npy \
  --reference-file /path/to/reference_image.npy \
  --annotations-file /path/to/level_15annot.npy \
  --template-name reference \
  --eigenvector-dir /path/to/output/eigenvectors \
  --output-dir /path/to/output \
  --slices-dataset-file maldi/data/splits/fold_2.json \
  --stride 8 --threshold 5 --knn-k 15 --num-modes 100 \
  --knn-method faiss --laplacian-norm randomwalk \
  --graphbandwidth-init 0.05 --nu 2 \
  --num-inducing 200 --latent-dim 5 \
  --epochs 2 --batch-size 1000 --learning-rate 0.001 --seed 42
```

Start coarse (large stride, few modes): the first run pays for the graph build and the eigensolve, every later run on the same geometry is a cache hit. Metrics go to Weights & Biases — export `WANDB_API_KEY`, or set `WANDB_MODE=offline`.

The main geometry flags are `--stride`, `--threshold`, `--knn-k`, `--knn-method` (`faiss` / `anatomical_atlas` / `faiss_atlas_weighted` / `faiss_cluster_weighted`), `--laplacian-norm`, `--graphbandwidth-init`, `--num-modes`; the anatomy prior is `--cross-region-inflation`, `--root-handling`, `--denoise-labels`, `--prune-cross-region`; the kernel is `--nu`, `--bump-scale`, `--bump-decay`, `--per-task-lengthscale`, `--diffusion-scale-init` / `--learn-diffusion-scale`, `--product-ard-matern`. See [`manifold/README.md`](manifold/README.md) for the complete tables.

### 7.5 Cluster Submission

Jobs go to the same **RunAI** cluster as the baseline. The chain is:

```
./submit/run_manifold_batch.sh                       (your machine)
   └── runai training submit ... -e KEY=VAL ...  --  ./local_run/run_manifold.sh
          └── python manifold/lgp_manifold_experiment.py    (in the container)
```

Everything the container needs travels as environment variables, and `run_manifold.sh` is the single env→CLI translation layer — so local and cluster runs are the same code path producing the same experiment names.

**Prerequisite:** a `.env` at the repo root containing `export WANDB_API_KEY=...`; the batch scripts refuse to run without it.

```bash
DRY_RUN=1 ./submit/run_manifold_batch.sh    # print the runai commands, submit nothing
./submit/run_manifold_batch.sh              # submit the sweep
```

The sweep grid is a block of bash arrays in the middle of the script — `FOLDS`, `STRIDE_NUM_MODES` (`"stride:modes"` pairs), `MAN_KNN_METHODS`, `MAN_INFLATIONS`, `ROOT_HANDLINGS`, `MAN_DENOISE_LABELS`, `MAN_PRUNE_CROSS_REGIONS`, `LAPLACIAN_NORMS`, `GRAPH_BANDWIDTHS`, `NU`, `BUMP_SCALES`, `BUMP_DECAYS`, `THRESHOLDS`, `KNN_K`, `IND_SOURCES`, `DIFFUSION_SCALES`, `PRODUCT_ARD`. Edit and resubmit. Refinement knobs that only apply to the weighted graph methods collapse to their no-op value for plain `faiss`, so nothing is submitted redundantly.

Useful submit-time overrides:

```bash
ATLAS_LEVEL=5 ./submit/run_manifold_batch.sh                        # other annotation volume
BETA=0.1 ./submit/run_manifold_batch.sh                             # KL weight (tags the output dir)
S3_OUTPUT_DIR=/s3/.../my_batch ./submit/run_manifold_batch.sh
FAISS_CPU=1 FORCE_RECOMPUTE_GRAPH=1 ./submit/run_manifold_batch.sh  # CPU-only FAISS timing
N_LIST=sqrt N_PROBE=8 FAISS_CPU=1 ./submit/run_manifold_batch.sh    # fast approximate CPU build
RECONSTRUCTION_LIPIDS_FILE=/path/lipids.txt ./submit/run_manifold_batch.sh
```

Each job defaults to 4 CPU / 48 GB / 0.5 GPU on `artiomartiom/sdsc:maldi_manifold_all_latest`.

| Script | Purpose |
|---|---|
| `submit/run_manifold_batch.sh` | The manifold sweep (inducing-point runner) |
| `submit/run_spectral_manifold_batch.sh` | The spectral twin; shares all graph/eigenvector caches |
| `submit/run_slepc_cache_prepare.sh` | GPU-free eigenpair pre-computation via MPI + SLEPc shift-invert |
| `submit/run_lgp_batch.sh` | The Euclidean LGP baseline on the same folds |
| `submit/run_parcel_lgp_batch.sh`, `run_sota_batch.sh`, `run_submit_baselines_sweep.sh`, `run_submit_per_lipid.sh` | The other comparison sweeps on the same folds |

The surrounding cluster-ops helpers (keep-alive pod, resumable rsync over the port-forward, run-directory downloads, job query / cancel / retry) are `.gitignore`d local-only tooling and are not in the repo — nothing in the training or submission path needs them. Working against a pod interactively is just:

```bash
runai workspace port-forward <job> --port 2222:2222 &
ssh -p 2222 appuser@localhost          # or: runai workspace exec <job> -it -- bash
rsync -avP -e 'ssh -p 2222' ./localdir/ appuser@localhost:/dest/
```

with the pod submitted **without a command**, so the image's entrypoint starts `sshd` rather than dropping into job mode.

### 7.6 Geometry Caches

The expensive geometry is computed once and reused. Under `--eigenvector-dir`:

```
<eigenvector-dir>/
├── knn/     <graph-key>.{npz,json}      # edge_index, edge_value, node coords
└── eigvecs/ <eig-key>.eigpairs.npz      # eigval (M,), eigvec (N, M)  + .meta.json
```

Keys come from the `make_key` helpers in `manifold_gp.utils`, which **every** producer and consumer calls — so a cache written by a training run, by `manifold/compute_eigenvectors.py` (offline GPU), or by `manifold/slepc/slepc_eigensolve.py` (MPI/SLEPc) is a drop-in hit for the others. Eigenpairs are nested, so a cached 2300-mode solve satisfies a 300-mode request by truncation, and the annotation filename stem is part of the key so different atlas levels never collide.

Pre-compute ahead of a sweep, then check what you have:

```bash
python manifold/compute_eigenvectors.py \
  --reference-volume /path/reference_image.npy \
  --annotations-volume /path/level_15annot.npy \
  --output-path /path/eigenvectors \
  --stride 4 --threshold 5 --knn-k 15 --modes 2300 \
  --nlist 1 --bandwidth 0.05 --normalization randomwalk \
  --knn-method faiss_atlas_weighted --cross-region-inflation 50 \
  --root-handling dissolve --denoise-labels 3 --prune-cross-region 0.95

python manifold/check_eigencache.py --eigenvector-dir /path/eigenvectors --list
```

### 7.7 Output Layout

`EXP_NAME` encodes every swept hyperparameter; conditional suffixes are appended only when a knob is non-default, so historical directory names stay valid:

```
{EXP_PREFIX}-MANIFOLD-{RSAMPLE|MEAN}-{latent}-{stride}-K{modes}-{template}-{threshold}
  -{ind_source}-{num_inducing}-{batch}-{knn_method}-{inflation}-{knn_k}-{lap_norm}
  -{nu}-{bump_scale}-{bump_decay}-{bandwidth}
  [-learndiff{init}] [-prodard{nu}] [-blend{frac}] [-{atlas_stem}]
  [-root{mode}] [-dn{N}] [-prune{F}] [-beta{β}]
```

```
{OUTPUT_DIR}/{EXP_NAME}/
├── args.npy, config.json     # exact arguments + human-readable snapshot
├── model.pth                 # trained weights (also the source of truth for inducing geometry)
├── checkpoints/              # model_{epoch}.pth
├── train/, test/             # predictions.npy, true_values.npy, scatter plots
├── volume/                   # dense per-lipid whole-brain volumes
├── volume_sparse/            # written instead of volume/ under --render-voxels-only
└── renders/                  # composite figures, diagnostics, error slices
```

`--render-voxels-only` (on by default in `run_manifold.sh`) reconstructs only the voxels the composite render reads — ~5.5× fewer for a near-identical figure. Set `RENDER_VOXELS_ONLY=0` when you need dense volumes for napari or the analysis scripts.

### 7.8 Tooling

All of it reads the same caches as training, so nothing is recomputed:

- **`manifold/benchmarks/`** — `bump_support_report.py` (are your points inside the kernel's bump support, or silently falling back to the prior?), `nystrom_benchmark.py` (accuracy of the out-of-sample extension), `graph_bandwidth_sweep.py` (pick a bandwidth that is neither saturated nor collapsed), `spectral_distance_sweep.py` + `spectral_sweep_report.py` (which distance metric the lipid covariance actually decays along), `eigensolver_compare.py`. Each has a matching `.sh` with a worked invocation.
- **`manifold/visualizations/`** — a napari explorer that fits a Euclidean and a manifold kernel side by side and reports the border-budget numbers, its standalone numeric engine, and a raw-MALDI-over-the-template viewer.
- **`manifold/notebooks/`** — the boundary-behaviour and distance-metric narrative notebooks, with their pooled anchor caches checked in so they run without a rebuild.
- **`manifold/toy_example/`** — synthetic folded geometries where a manifold kernel should win, for sanity-checking the whole stack.
- Local-only (`.gitignore`d): `manifold/viz/` (the wider napari explorer fleet), `manifold/experiments/` (validation probes), `manifold/analysis/` (figure notebooks and cache builders).

---

## 8. Shell Scripts & Use Cases

### 8.1 `deploy.sh` — Code Sync

Synchronises the local repository to a remote compute node over SSH (port 2222):

```bash
./maldi/deploy.sh
```

Uses `rsync` with exclusions for `.git/`, `.gitignore`, and build artefacts. The remote target is `root@localhost:/myhome/` (tunnelled via a local port forward). This is intended to be run before submitting jobs to the cluster.

### 8.2 `run_all.sh` — Single LGP Fold Run

**This is the core worker script** executed by each cluster job. It accepts four positional arguments:

```bash
bash maldi/run_all.sh <NUM_INDUCING> <LATENT_DIM> <FOLD> <BATCH_SIZE>
```

**Example:**
```bash
bash maldi/run_all.sh 500 10 1 2000
```

**What it does:**
1. Sources the virtual environment
2. Installs `l3di` into the user environment (`pip install --user`)
3. Runs `lgp_experiment.py` with hard-coded paths pointing to the remote S3 data storage
4. Uses the **symmetric kernel** by default
5. Applies **log transform**
6. Enables **diffusion** (`--use-diffusion`)
7. Runs for **20 epochs**

**Fixed defaults inside the script:**
- `DATA_PATH`: S3 mount at `/s3/mlibra/mlibra-data/maldi/`
- `OUTPUT_DIR`: `/s3/mlibra/mlibra-data/maldi/lmmvae`
- `KERNEL`: `symmetric`
- `N_EPOCHS`: `20`
- `LEARNING_RATE`: `0.001`
- `SEED`: `416465`

**Experiment name generated:**
```
CV-FOLD-{FOLD}-{LATENT_DIM}-{NUM_INDUCING}-{BATCH_SIZE}{N_PIXELS}_log
```

### 8.3 `run_all_moe.sh` — Single LGPMOE Fold Run

Same as `run_all.sh` but calls `lgpmoe_experiment.py` with an additional parameter:

```bash
bash maldi/run_all_moe.sh <NUM_INDUCING> <LATENT_DIM> <FOLD> <BATCH_SIZE> <NUM_EXPERTS> [LEARNING_RATE]
```

**Example:**
```bash
bash maldi/run_all_moe.sh 500 10 1 2000 5
```

Outputs go to `/s3/mlibra/mlibra-data/maldi/lgpmoe`. The experiment name includes the number of experts:
```
lgpmoe_CV-FOLD-{FOLD}-{LATENT_DIM}-{NUM_INDUCING}-{BATCH_SIZE}-MOE-{NUM_EXPERTS}{N_PIXELS}_log
```

Runs for **40 epochs** and does **not** include `--use-diffusion`.

### 8.4 `run_final.sh` — Full-Data Final Model

Trains the LGP on **all available data** (`all.json` split, no held-out test set) for **50 epochs**. Intended to produce the best-quality model for downstream whole-brain reconstruction.

```bash
bash local_run/run_lgp.sh
```

**Fixed configuration:**
- `NUM_INDUCING_POINTS`: 500
- `LATENT_DIM`: 5
- `BATCH_SIZE`: 1000
- `KERNEL`: `symmetric`
- `N_EPOCHS`: 50
- `SLICES_DATASET_FILE`: `all.json`

No `--use-diffusion` flag. Experiment name: `LGPALL-{LATENT_DIM}-{NUM_INDUCING}-{BATCH_SIZE}{N_PIXELS}_log`

### 8.5 `do_cv.sh` — Local CV Grid

Runs the full CV grid sequentially **without** cluster submission. Iterates the same hyperparameter grid as `submit_cv.sh`:

```bash
bash maldi/do_cv.sh
```

Grid:
```
NUM_INDUCING  in {500, 1500}
LATENT_DIM    in {5, 20}
BATCH_SIZE    in {1000, 2000}
FOLD          in {1, 2, 3, 4, 5}
```

Total: **40 runs** (2 × 2 × 2 × 5). Useful for debugging on a machine with sufficient memory and GPU.

### 8.6 `submit_cv.sh` — Cluster CV Grid

Submits the same 40-run grid to a **RunAI** Kubernetes cluster:

```bash
bash maldi/submit_cv.sh
```

Sequence:
1. Runs `deploy.sh` to sync code to the remote
2. For each combination, calls `runai workspace submit` with:
   - CPU: 15 cores
   - Memory: 50 GB
   - GPU: 0.5 (fractional GPU sharing)
   - Preemptible jobs
   - Container: `registry.renkulab.io/daniel.trejobanos1/mlibra`
   - Command: `bash /myhome/mlibra/maldi/run_all.sh <args>`
3. Job name format: `gp-postpred-cv-{fold}-{inducing}-{latent}-{batch_size}`

---

## 9. Cross-Validation Protocol

The CV scheme evaluates **spatial generalisation**: given training data from several brain sections (from multiple biological replicates), can the model predict lipid intensities at unseen sections from different replicates?

| Aspect | Detail |
|---|---|
| Number of folds defined | 8 (in `data/splits/`) |
| Folds used in grid search | 5 (folds 1–5) |
| Train samples (fold 1) | `ReferenceAtlas`, `SecondAtlas`, `Female3`, `Male1`, `Male2`, `Male3` |
| Test samples (fold 1) | `Female1`, `Female2` |
| Ignored samples | `GBA1`, `Pregnant1`, `Pregnant2`, `Pregnant4` (disease/special models) |
| Evaluation metrics | MSE, MAPE, MRE, Pearson correlation — per lipid channel |

Predictions are always made by querying the trained GP at the 3-D CCF coordinates of the test pixels; no information from the test section's lipid measurements is used during inference.

---

## 10. Experiment Naming Convention

Experiment names are constructed programmatically from the configuration:

| Experiment type | Name pattern |
|---|---|
| CV fold (LGP) | `CV-FOLD-{fold}-{latent_dim}-{num_inducing}-{batch_size}{n_pixels}_log` |
| CV fold (LGPMOE) | `lgpmoe_CV-FOLD-{fold}-{latent_dim}-{num_inducing}-{batch_size}-MOE-{num_experts}{n_pixels}_log` |
| Final model | `LGPALL-{latent_dim}-{num_inducing}-{batch_size}{n_pixels}_log` |

The `{n_pixels}` suffix is appended by `MaldiConfig.from_args()` and the `_log` suffix is appended when `--log-transform` is active.

**Example:**
`NUM_INDUCING=500`, `LATENT_DIM=10`, `FOLD=3`, `BATCH_SIZE=2000`, `N_PIXELS=10`, log-transform on → `CV-FOLD-3-10-500-200010_log`

---

## 11. Installation

### Prerequisites

- Python 3.10+ (Python 3.12 used in development)
- CUDA-compatible GPU recommended for training (CPU fallback is automatic)

### Install the `l3di` Package

From the repository root:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install l3di in editable mode
pip install -e .

# Install all experiment dependencies
pip install -r requirements.txt
```

### MALDI-Specific Dependencies

```bash
pip install -r maldi/requirements.txt
```

### Optional: Allen SDK (for whole-brain reconstruction)

```bash
pip install allensdk
```

### Docker

A `Dockerfile` is provided for the Renku/RunAI environment. It is based on `jupyter/base-notebook:python-3.10` and installs all requirements from `requirements.txt`:

```bash
docker build -t mlibra .
docker run -p 8888:8888 mlibra
```

The container also configures an SSH server (port 2222) to support the `deploy.sh` workflow.

---

## 12. Running an Experiment

### Minimal Local Test

```bash
cd maldi/

python lgp_experiment.py \
  --mode lgp \
  --exp-name test_run \
  --dataset-path /path/to/maldi_data/ \
  --maldi-file /path/to/maindata_minimal.parquet \
  --available-lipids-file /path/to/available_lipids.npy \
  --output-dir /path/to/output/ \
  --slices-dataset-file data/splits/fold_1.json \
  --num-inducing 100 \
  --latent-dim 5 \
  --kernel rbf \
  --epochs 5 \
  --batch-size 500 \
  --learning-rate 0.001 \
  --seed 42 \
  --n-pixels 10 \
  --device cpu \
  --log-transform
```

### Full CV Grid on Cluster

```bash
# From the repository root on your local machine
bash maldi/submit_cv.sh
```

This will first rsync the code, then submit 40 RunAI jobs.

### Final Model Training

```bash
# On the remote compute node (after deploy.sh)
bash /myhome/mlibra/local_run/run_lgp.sh
```

### LGPMOE Experiment

```bash
python maldi/lgpmoe_experiment.py \
  --mode lgpmoe \
  --num-experts 5 \
  --exp-name moe_test \
  ... (same required flags as lgp_experiment.py) ...
```

### Multimodal Experiment

```bash
cd multimodal/

python multimodallgp_experiment.py \
  --mode lgp_multimodal \
  ... (extended flag set; see multimodal/config.py) ...
```

---

## 13. Testing

Tests are located in `tests/` and use **pytest**.

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/test_lgp_kernel.py -v
```

### Test Suite Contents

| File | Tests |
|---|---|
| `test_infrastructure.py` | Verifies pytest fixtures return correct shapes/types |
| `test_lgp_kernel.py` | `Custom3DKernel`: initialisation, output shape, symmetry, self-similarity > cross-similarity |
| `test_lgp_gpmodel.py` | `IndependentMultitaskGPModel`: initialisation, forward pass shape `(batch, num_tasks)`, device transfer |

### Shared Fixtures (`conftest.py`)

| Fixture | Shape | Description |
|---|---|---|
| `device` | — | `torch.device("cuda")` or `"cpu"` |
| `mock_1d_data` | `(4, 1, 128)` | Random 1-D lipid spectra |
| `mock_3d_coords` | `(10, 3)` | Random 3-D coordinates |
| `mock_latent_z` | `(4, 32)` | Random latent vectors |

---

## 14. Infrastructure & Deployment

### Compute Environment

Jobs are submitted to a **RunAI** cluster backed by Kubernetes. The container image is hosted at `registry.renkulab.io/daniel.trejobanos1/mlibra`. Data is stored on an S3-compatible object store mounted at `/s3/mlibra/mlibra-data/`.

### Local Development Workflow

```
[Local machine]
  └── edit code
  └── bash maldi/deploy.sh          # rsync to remote (via SSH tunnel on port 2222)

[Remote compute node / cluster job]
  └── pip install --user /myhome/mlibra   # install l3di
  └── python lgp_experiment.py ...        # run experiment
  └── results saved to /s3/mlibra/mlibra-data/maldi/lmmvae/
```

### Experiment Tracking

All training metrics are logged to **Weights & Biases** (`wandb`). The following metrics are tracked per epoch:

| Metric | Description |
|---|---|
| `loss` | Total ELBO loss (NLL + β·KL) |
| `reconstruction_loss` | NLL term only |
| `kl_loss` | GP KL divergence |
| `mse_loss` | Mean squared error (monitoring metric) |
| `loss_batch` | Batch-level loss (logged every 10 batches) |
| `mse_loss_batch` | Batch-level MSE |

---

## 15. Dependencies

### Core (`l3di`)

| Package | Role |
|---|---|
| `torch` | Neural network backbone and autograd |
| `gpytorch` | Gaussian Process primitives (kernels, variational inference, likelihoods) |
| `numpy` | Array manipulation |
| `tqdm` | Progress bars |
| `wandb` | Experiment tracking |

### Experiment (`maldi/`)

| Package | Role |
|---|---|
| `pandas` + `pyarrow` | Parquet data loading with predicate pushdown |
| `scikit-learn` | MiniBatchKMeans for inducing point generation |

### Full Requirements

| Package | Role |
|---|---|
| `matplotlib` | Scatter plots and volumetric slice figures |
| `scipy` | Statistical utilities (Spearman correlation, etc.) |
| `scikit-image` + `imageio` | Image I/O for visualisation |
| `numba` | JIT compilation for performance-critical loops |
| `torch_geometric` | Graph neural network utilities (potential future extension) |
| `napari` + `napari-animation` | Interactive 3-D brain volume viewer and rotation video export |
| `allensdk` | Allen Mouse Brain Atlas CCF template volume download |

---

## 16. The EUCLID Baseline

[EUCLID](https://github.com/lamanno-epfl/EUCLID) is the Lipid Brain Atlas
pipeline. Its `anatomical_interpolation`
routine reconstructs a lipid's whole-brain volume, which makes it the closest
published comparison to the models here — so it is wired in as
`--model euclid` in `baselines/experiment_baselines.py`.

**Nothing about the estimator is reimplemented.** `baselines/euclid_kernel.py`
extracts three functions from a clone of their repo and executes them as
written:

| Function | Role |
|---|---|
| `anatomical_interpolation` | driver: log → per-voxel mean → rescale → `w` clip → interpolate → box fill |
| `fill_array_interpolation` | the kernel: structure-gated `exp(-d)` weighted mean |
| `normalize_to_255` | their per-lipid intensity rescale |

We supply only an AnnData-shaped object for their driver to read and a sampler
that reads the resulting volumes at held-out pixels. The clone is fetched
automatically by `local_run/run_baseline.sh` if `$EUCLID_REPO` is absent.

### What the method is

For each 100 µm voxel, a weighted mean of the *measured* voxels within 26 voxels
(2.6 mm) that carry the **same Allen leaf label**, weighted `exp(-d)` with `d` in
voxel units; leftover holes take a 10³ box mean. There is nothing to train — no
fitted lengthscale, no uncertainty. It is a fixed linear smoother whose sparsity
pattern is set by the atlas.

### Running it

```bash
# locally, one fold
MODEL=euclid EUCLID_W=0 EUCLID_JOBS=16 \
    EXP_PREFIX=FOLD-2 SLICES_DATASET_FILE=maldi/data/splits/fold_2.json \
    sh local_run/run_baseline.sh --reconstruct whole_brain

# the cluster sweep: atlas x fold x w x normalisation
./submit/run_submit_euclid.sh                 # DRY_RUN=1 to preview
```

Their kernel is a single-threaded numba loop costing ~242 s per lipid, so
`EUCLID_JOBS` (one process per lipid) is the only speed lever; 173 lipids take
~11.6 h serially and ~30 min at 25-way. Completed per-lipid volumes are written
to `<run>/euclid_volumes/` and are **resumable** — a re-run skips lipids whose
volume already exists, and truncated files (from a killed job) are detected and
recomputed.

### Two caveats that matter for interpretation

**Their `w` default is a rendering threshold, not a modelling one.** `w=50`
drops voxels below that value on their 0–255 rescale; it is calibrated for
uMAIA-normalised intensities. At this dataset's scale every `log(x)` is
negative, the rescale never reaches 255, and `w=50` removes ~92 % of donor
voxels. Use `EUCLID_W=0` for a like-for-like comparison.

**`anatomical_interpolation` is not the paper's imputation claim.** In EUCLID it
feeds the 3-D rendering code; their imputation of failed acquisitions is
`xgboost_feature_restoration`, which is cross-lipid (predict a lipid from the
other lipids at the same pixel), not spatial. Describe the comparison as against
EUCLID's *anatomical interpolation*, not against "EUCLID's imputation".

---

## Licence

MIT — see `LICENSE` for details.
