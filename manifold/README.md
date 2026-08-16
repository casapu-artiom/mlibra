# `manifold/` — Riemann Manifold Gaussian Processes on the Mouse Brain

> The manifold module replaces the Euclidean kernel of the baseline LGP (`maldi/lgp_experiment.py`) with a **graph-Laplacian Riemann–Matérn kernel** built on the brain's own geometry. Instead of measuring similarity with straight-line distance through tissue and ventricles alike, the GP measures it along a k-nearest-neighbour graph over the Allen CCF reference volume, optionally biased by an anatomical atlas. Everything downstream — the MLP decoder, the data loading, the CV splits, the whole-brain reconstruction — is shared with `maldi/`.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Directory Layout](#3-directory-layout)
4. [The Vendored `manifold_gp` Package](#4-the-vendored-manifold_gp-package)
5. [Models — `lgp_manifold.py`](#5-models--lgp_manifoldpy)
6. [Entry Points](#6-entry-points)
7. [Graph Construction & Refinement](#7-graph-construction--refinement)
8. [Eigenpairs & the Cache Layout](#8-eigenpairs--the-cache-layout)
9. [The Riemann–Matérn Kernel](#9-the-riemannmatérn-kernel)
10. [Inducing Points](#10-inducing-points)
11. [Installation](#11-installation)
12. [Running Locally](#12-running-locally)
13. [Cluster Submission](#13-cluster-submission)
14. [Output Directory Layout](#14-output-directory-layout)
15. [Atlas Volumes](#15-atlas-volumes)
16. [Tooling — Benchmarks, Visualisation, Analysis](#16-tooling--benchmarks-visualisation-analysis)
17. [Known Pitfalls](#17-known-pitfalls)

---

## 1. Motivation

A mouse brain is not a Euclidean ball. Two voxels can be millimetres apart along the tissue while sitting adjacent across a ventricle or a sulcus, and an isotropic RBF/Matérn kernel will happily smear lipid intensity across that gap. The manifold approach:

1. Discretises the tissue into a graph — nodes are voxels of the CCF reference template above an intensity `--threshold`, subsampled by `--stride`.
2. Connects each node to its `k` nearest neighbours (FAISS), so edges follow tissue rather than air.
3. Forms the **graph Laplacian** `L` of that graph (symmetric or random-walk normalised) and solves for its lowest `--num-modes` eigenpairs `(λ_k, φ_k)`.
4. Defines a **Riemann–Matérn kernel** as a spectral filter over those eigenpairs — the graph analogue of a Matérn kernel on a Riemannian manifold (Borovitskiy et al.; implementation vendored from Fichera et al., [Implicit Manifold Gaussian Process Regression](https://arxiv.org/abs/2310.19390)).
5. Uses that kernel in the same latent-GP-plus-MLP-decoder architecture as the baseline LGP, so the two are directly comparable on the same folds and metrics.

Optionally the graph can be biased by anatomy: edges crossing an Allen atlas region boundary get their length inflated (soft prior) or removed outright (hard prune), so heat diffuses within regions far more readily than across them.

---

## 2. Pipeline Overview

```
reference_image.npy  (+ level_*annot.npy)
        │
        │  crop_or_stride_volume(stride)  →  sub_volume, sub_atlas
        │  reference_ccf_from_subvolume(threshold)
        ▼
  graph nodes (N × 3, standardised CCF mm)
        │
        │  FAISS kNN (k = --knn-k, IVF nlist/nprobe)              ← cached: <eigvec-dir>/knn/
        │  [+ atlas / cluster region labels]
        │  [+ root dissolve → label denoise → cross-region inflate → hard prune]
        ▼
  edge_index, edge_value
        │
        │  GraphLaplacianOperator(bandwidth, normalisation)
        │  LaplacianEigensolver(num_modes)                        ← cached: <eigvec-dir>/eigvecs/
        ▼
  (λ_k, φ_k)  k = 1 … num_modes
        │
        │  RiemannMaternKernel(nu, bump_scale, bump_decay, …)
        │  [Nyström out-of-sample extension for non-node queries]
        ▼
  LatentRiemannGP  (M inducing points, d latent tasks)   ── or ──  SpectralLatentGP (weight space)
        │
        │  rsample → MLP decoder [256, 256, 128]
        ▼
  x̂ ∈ R^p  (p lipids)      loss = NLL + β·KL(q‖p) − log p(θ)
```

The three caches — kNN graph, eigenpairs, and inducing points — are keyed by content so that a sweep over training hyperparameters never recomputes the geometry.

---

## 3. Directory Layout

```
manifold/
├── lgp_manifold.py                      # Models: LatentRiemannGP, ManifoldLGP,
│                                        #   SpectralLatentGP, SpectralManifoldLGP
├── lgp_manifold_experiment.py           # ENTRY POINT — inducing-point manifold run
├── spectral_lgp_manifold_experiment.py  # ENTRY POINT — weight-space (spectral) run
├── manifold_kernel_builder.py           # Shared graph→Laplacian→eigenpairs→kernel builder
│                                        #   (+ add_manifold_args for other runners, e.g. GPLFR)
├── compute_eigenvectors.py              # Offline GPU (cupy) eigensolve into the shared cache
├── check_eigencache.py                  # Which eigenvector caches exist / are missing
├── model_hyperparams.py                 # Dump trained kernel/GP hyperparameters from checkpoints
├── download_bg_atlas.py                 # Fetch + depth-coarsen the Allen/BrainGlobe annotation
│
├── manifold_gp/                         # Vendored IMGP package (installed as top-level module)
│   ├── kernels/    riemann_kernel.py, riemann_matern_kernel.py, surface_kernels.py
│   ├── models/     riemann_gp.py, spectral_riemann_gp.py, vanilla_gp.py
│   ├── operators/  graph_laplacian_operator.py, precision_matern_operator.py, …
│   ├── priors/     inverse_gamma_prior.py
│   └── utils/      nearest_neighbors.py, compute_eigenvectors.py, anatomical_knn.py,
│                   surface_depth.py, …
│
├── slepc/                               # MPI eigensolver
│   ├── slepc_eigensolve.py              #   SLEPc Krylov-Schur / shift-invert (MUMPS)
│   └── build_slepc_petsc.sh             #   PETSc + SLEPc + MUMPS source build
│
├── docker/
│   ├── Dockerfile                       # Training image (torch + FAISS + PETSc/SLEPc)
│   ├── Dockerfile.withslepc             # SLEPc layer on top of the older withfaiss image
│   ├── Dockerfile.notebook              # Jupyter/Renku fork of the root Dockerfile
│   └── entrypoint.sh                    # Dual-mode: sshd (interactive) vs job
│
├── benchmarks/                          # Quantitative sweeps over the geometry (see §16)
├── visualizations/                      # Manifold-vs-Euclidean explorer + its numeric engine
├── notebooks/                           # Narrative notebooks (boundary story, distance study)
└── toy_example/                         # Synthetic manifolds (swiss roll, box+cylinder)
```

`manifold/` deliberately has **no `__init__.py`** — it is a script directory, not a package. Only `manifold/manifold_gp` is installed, and it keeps its upstream top-level name (see [§4](#4-the-vendored-manifold_gp-package)).

**Local-only directories.** Three working directories are `.gitignore`d and therefore absent from a fresh clone — `viz/` (the napari explorer fleet), `experiments/` (one-off validation probes), and `analysis/` (the figure notebooks and their cache builders). They live in the author's working copy; a curated subset of the explorers is re-published separately. Everything the training and submission pipeline needs is tracked, so their absence never blocks a run — [§16](#16-tooling--benchmarks-visualisation-analysis) marks which tools they hold.

---

## 4. The Vendored `manifold_gp` Package

`manifold/manifold_gp/` is a fork of the upstream [`manifold-gp`](https://github.com/nash169/manifold-gp) implementation of Implicit Manifold GP Regression, extended here with the caching, atlas-weighting and out-of-sample machinery this project needs. It is imported as the **top-level** `manifold_gp`, which is why the root `pyproject.toml` lists a second package-discovery root:

```toml
[tool.setuptools.packages.find]
where = [".", "manifold"]
include = ["l3di*", "manifold_gp*", "other_experiments*"]
```

So `pip install -e .` from the repo root installs `l3di` **and** `manifold_gp`. Do not `pip install manifold-gp` from PyPI — it would shadow the fork.

Pieces that carry most of the weight:

| Module | Role |
|---|---|
| `utils/nearest_neighbors.py` | `KnnGraphCache` (FAISS build + on-disk cache), key construction, `resolve_nlist` / `resolve_nprobe`, the `--faiss-cpu-*` switches |
| `utils/compute_eigenvectors.py` | `LaplacianEigensolver` (cupy backend), cache load/store, `resolve_ncv_min` |
| `utils/anatomical_knn.py` | Node→region labelling from the atlas or from template clustering, `inflate_cross_region_edges`, `dissolve_root_labels`, `denoise_labels_majority_vote`, `prune_cross_region_edges` |
| `operators/graph_laplacian_operator.py` | The Laplacian as a `linear_operator`, plus the Nyström `out_of_sample` extension |
| `kernels/riemann_matern_kernel.py` | `RiemannMaternKernel` — the spectral Matérn filter over `(λ_k, φ_k)` |
| `utils/surface_depth.py` | EDT distance-to-surface feature for the (optional) surface kernel |

---

## 5. Models — `lgp_manifold.py`

Four classes, two complete model stacks.

### `LatentRiemannGP` + `ManifoldLGP` (inducing-point path)

- `LatentRiemannGP` is an `ApproximateGP` with `d` independent latent tasks over `M` inducing points that have been **snapped to exact graph nodes**. The covariance is the `RiemannMaternKernel`; with `--per-task-lengthscale` each latent dimension gets its own kernel instance (sharing eigenpairs and graph tensors, so the memory cost is one copy) for a multi-scale latent basis. With `--product-ard-matern` the geodesic kernel is multiplied by an ambient per-axis Euclidean Matérn, `k = k_geo · k_ARD`, restoring the directional anisotropy a pure Riemann kernel structurally lacks.
- `ManifoldLGP` wraps it with the MLP decoder (`[256, 256, 128]`, SiLU, dropout 0.1) and per-lipid learnable observation noise.

Because an MLP sits between the GP latent `f` and the observations, `p(x | f)` is **not** Gaussian in `f`, so training cannot use `gpytorch.VariationalELBO`. Instead the loss is a Monte-Carlo ELBO: `rsample()` the GP posterior, decode, then

```
L = NLL(x, decode(z), log_var_x) + β · KL(q(u) ‖ p(u)) − log p(θ)
```

`--beta` accepts a float or the literal `elbo` (which resolves to batch/N). The `− log p(θ)` term is MAP-II regularisation over the kernel hyperparameters (Gamma priors on outputscale and, when a centre is given, lengthscale; a noise prior). These prior objects live **outside** the `nn.Module` graph, so `state_dict()` is byte-identical to older checkpoints and they still load.

### `SpectralLatentGP` + `SpectralManifoldLGP` (weight-space path)

The spectral twin parametrises a diagonal variational posterior **directly over the manifold spectrum** — there are no inducing points at all, and hence no snapping, no blend, no `--inducing-*` flags. Graph, Laplacian and eigenpair handling are identical (same cache keys), and the decoder is the same architecture, so the two runners differ only in the GP. Run both to separate "does the geometry help?" from "does the inducing approximation hurt?".

### Helper wrappers

`BatchedRiemannWrapper` and `PerTaskRiemannWrapper` adapt a single kernel (or a list of per-task kernels) to GPyTorch's batched-kernel calling convention.

---

## 6. Entry Points

| Script | Model | Notes |
|---|---|---|
| `lgp_manifold_experiment.py` | `LatentRiemannGP` + `ManifoldLGP` | The main runner. Drives `MaldiExperiment` from `maldi/experiment.py`, so data loading, CV filters, prediction, scatter diagnostics and whole-brain reconstruction are shared with the baseline. |
| `spectral_lgp_manifold_experiment.py` | `SpectralLatentGP` + `SpectralManifoldLGP` | Same flags minus everything inducing-related. |
| `compute_eigenvectors.py` | — | Offline GPU eigensolve; writes into the same cache the runners read. |
| `slepc/slepc_eigensolve.py` | — | MPI/SLEPc eigensolve for the regimes where the GPU path runs out of memory. |
| `check_eigencache.py` | — | Grid or inventory report of which eigenpair caches exist. |
| `model_hyperparams.py` | — | Reads trained checkpoints and prints/CSV-dumps the constrained hyperparameter values. |

All of them bootstrap `sys.path` to the repo root and `maldi/` at import time, so they run from anywhere without `PYTHONPATH` fiddling.

### Key CLI flags (`lgp_manifold_experiment.py`)

**Geometry**

| Flag | Default | Meaning |
|---|---|---|
| `--stride N` | 4 | Subsample the reference volume by `N` in each axis before building the graph |
| `--threshold T` | 5 | Template intensity above which a voxel counts as tissue |
| `--knn-k K` | 15 | Neighbours per node |
| `--knn-method` | `faiss` | `faiss`, `anatomical_atlas`, `faiss_atlas_weighted`, `faiss_cluster_weighted` |
| `--n-list`, `--n-probe` | `1`, `1` | FAISS IVF sizing; `sqrt` resolves to `round(√N)` / `round(√nlist)` |
| `--laplacian-norm` | `symmetric` | `symmetric` or `randomwalk` |
| `--graphbandwidth-init` | 1.0 | Heat-kernel bandwidth used for edge weights **and** the eigensolve |
| `--num-modes M` | 200 | Laplacian eigenpairs to compute/use |
| `--ncv-min` | -1 | Lanczos Krylov subspace floor; `-1` = `max(1500, 3·modes + 20)` |

**Anatomy prior** (weighted methods only)

| Flag | Default | Meaning |
|---|---|---|
| `--cross-region-inflation X` | 10.0 | Multiply the length of edges crossing a region boundary by `X` |
| `--root-handling` | `dissolve` | `dissolve` (fold atlas label-0 "root" into the nearest region), `ignore`, `cross` (legacy) |
| `--denoise-labels N` | 0 | Majority-vote label-smoothing passes before the prune |
| `--prune-cross-region F` | 0.0 | Hard-remove fraction `F` of cross-region edges |
| `--cluster-k`, `--cluster-spatial-weight`, `--cluster-fit-subsample`, `--cluster-seed` | 64, 1.0, 40000, 0 | Template-clustering labels (atlas-free alternative) |

**Kernel**

| Flag | Default | Meaning |
|---|---|---|
| `--nu` | 1.0 | Matérn smoothness of the spectral filter |
| `--bump-scale`, `--bump-decay` | 3.0, 0.05 | Bump-function support and decay for the Nyström out-of-sample extension |
| `--lengthscale-init` | — | Initial kernel lengthscale |
| `--per-task-lengthscale` | off | One kernel per latent dimension |
| `--diffusion-scale-init`, `--learn-diffusion-scale` | 1.0, off | Multiplicative scale on the frozen spectrum (`λ_k → s·λ_k`); no eigenpair recompute |
| `--product-ard-matern`, `--product-ard-nu` | off, 2.5 | Multiply by an ambient ARD Euclidean Matérn |
| `--surface-kernel`, `--surface-depth-lengthscale` | off | Add the EDT distance-to-surface feature |

**Model / training**

`--latent-dim`, `--num-inducing`, `--inducing-source {reference,data}`, `--inducing-method`, `--inducing-from-maldi-nodes`, `--inducing-density-frac`, `--epochs`, `--batch-size`, `--learning-rate`, `--beta`, `--no-rsample`, `--seed`.

**IO / behaviour**

`--dataset-path`, `--maldi-file`, `--available-lipids-file`, `--reference-file`, `--annotations-file`, `--template-name`, `--eigenvector-dir`, `--output-dir`, `--exp-name`, `--slices-dataset-file`, `--do-brain-reconstruction`, `--render-voxels-only`, `--reconstruction-lipids`, `--force-recompute-graph`, `--faiss-cpu-graph|-search|-recon`.

---

## 7. Graph Construction & Refinement

| `--knn-method` | Topology | Weights | Needs atlas? |
|---|---|---|---|
| `faiss` | Plain kNN over tissue nodes | Heat kernel on Euclidean edge length | no |
| `anatomical_atlas` | Per-region connectivity-3 adjacency | — | yes |
| `faiss_atlas_weighted` | Plain kNN (cache shared with `faiss`) | Cross-region edges inflated ×`--cross-region-inflation` | yes |
| `faiss_cluster_weighted` | Plain kNN (cache shared with `faiss`) | Same inflation, but labels come from k-means clustering of the template itself | no |

`faiss_atlas_weighted` is the workhorse. Its refinement chain, in order:

1. **Root handling** — the Allen label `0` ("root") is real tissue, not background, and in the shipped partial atlases it covers a large fraction of the brain. `dissolve` reassigns each root node to its nearest labelled region so the prior does not spray through interiors; `cross` is the legacy behaviour that treated root boundaries as region crossings.
2. **Inflation** — `inflate_cross_region_edges` multiplies the *length* of every cross-region edge, so the heat-kernel affinity across it collapses. This is a *soft* prior: the topology is untouched.
3. **Denoise** — `--denoise-labels N` runs `N` majority-vote passes over the graph to clean speckle in the labels the prune will act on.
4. **Prune** — `--prune-cross-region F` hard-removes the top fraction `F` of cross-region edges. This changes the topology, hence the eigenpairs, hence the cache key.

Interactions worth knowing before you sweep these are listed in [§17](#17-known-pitfalls).

---

## 8. Eigenpairs & the Cache Layout

Everything geometric is cached under `--eigenvector-dir`:

```
<eigenvector-dir>/
├── knn/
│   └── <graph-key>.{npz,json}        # edge_index, edge_value, node coords
└── eigvecs/
    ├── <eig-key>.eigpairs.npz        # eigval (M,), eigvec (N, M)
    └── <eig-key>.meta.json
```

Keys are built by `make_key` in `manifold_gp.utils.{nearest_neighbors,compute_eigenvectors}` — **the same helpers every producer and consumer calls**, so a cache written by `compute_eigenvectors.py`, by the SLEPc solver, or by a training run is a drop-in hit for the others.

The graph key covers template, stride, threshold, method, `k`, `nlist` (only when ≠ 1), atlas stem, and the weighting/prune tags. The eigenvector key adds normalisation, bandwidth and mode count. Two conveniences:

- `allow_larger_modes=True` — eigenpairs are nested, so a cached 2300-mode solve satisfies a 300-mode request by truncation.
- Cache keys carry the annotation filename stem, so `level_5` and `level_15` runs never collide. The historical default (`level_15annot`) keeps its un-suffixed keys so pre-existing caches stay valid.

### Three ways to produce eigenpairs

```bash
# 1. In-process (default): cupy Lanczos, computed on first use and cached.
#    Nothing to do — just run the experiment.

# 2. Offline GPU, ahead of a sweep:
python manifold/compute_eigenvectors.py \
  --reference-volume   /path/reference_image.npy \
  --annotations-volume /path/level_15annot.npy \
  --output-path        /path/eigenvectors \
  --stride 4 --threshold 5 --knn-k 15 --modes 2300 \
  --nlist 1 --bandwidth 0.05 --normalization randomwalk \
  --knn-method faiss_atlas_weighted --cross-region-inflation 50 \
  --root-handling dissolve --denoise-labels 3 --prune-cross-region 0.95

# 3. CPU cluster, MPI + SLEPc shift-invert (for small stride, where the GPU
#    path and scipy's 32-bit SuperLU both run out of room):
mpirun -n 16 python manifold/slepc/slepc_eigensolve.py \
  --stride 1 --threshold 5 --knn-k 15 --modes 300 --shift-invert \
  --eigenvector-dir /path/eigenvectors \
  --reference-file  /path/reference_image.npy \
  --annotations-file /path/level_15annot.npy --build-if-missing
```

Then check what you have:

```bash
python manifold/check_eigencache.py --eigenvector-dir /path/eigenvectors --list

python manifold/check_eigencache.py --eigenvector-dir /path/eigenvectors \
  --knn-k 15 --threshold 5 --num-modes 2300 --stride 4 \
  --nlist sqrt --method faiss faiss_atlas_weighted --inflation 10 50
```

---

## 9. The Riemann–Matérn Kernel

`RiemannMaternKernel` evaluates

```
k(x, y) = Σ_k  S(λ_k)  φ_k(x) φ_k(y),      S(λ) = (2ν/ℓ² + s·λ)^(−ν)
```

with `(λ_k, φ_k)` the cached Laplacian eigenpairs. Two practical consequences:

- **Finite rank.** The kernel's rank is exactly `--num-modes`. Asking for more inducing points than modes makes `K_uu` singular and adds no capacity.
- **Resolution ceiling.** The usable mode count is bounded by node density: `--stride` and `--threshold` decide how fine a mode the graph can represent, and modes beyond that are aliased noise rather than signal. More modes is not automatically better.

Queries that are not exact graph nodes (MALDI voxels, inducing points before snapping, reconstruction targets) are handled by the **Nyström out-of-sample extension**, gated by a bump function of radius `bump_scale · graphbandwidth`. Points outside every bump get zeroed manifold features and silently fall back to the prior — `benchmarks/bump_support_report.py` exists precisely to check that this is not happening to your train/test set.

The lengthscale enters as an additive floor `2ν/ℓ²`; its multiplicative companion `s` is the **diffusion scale** (`--diffusion-scale-init`, learnable with `--learn-diffusion-scale`, `s = 1` is the identity). Rescaling the frozen spectrum leaves the eigenvectors unchanged, so `s` reshapes the filter with no eigensolve recompute.

---

## 10. Inducing Points

Three placement strategies, all followed by a **snap to the nearest graph node** (the kernel is only exact on nodes):

| `--inducing-source` | Behaviour |
|---|---|
| `reference` | k-means over the reference tissue image (the `maldi/utils.py` symmetric scheme) |
| `data` | k-means / FPS / random over the *measured* MALDI voxels, so placement follows where data actually is |
| `--inducing-from-maldi-nodes` (blend) | **Overrides** the source: `--inducing-density-frac` (default 0.8) of the points come from the densest graph nodes, the rest from the measured MALDI voxels that snap onto the graph most cheaply. Both sources are already exact graph nodes, so the snap is a no-op. |

Snapping deduplicates, so the realised count is ≤ `--num-inducing` and varies run to run (the FAISS graph is not bit-reproducible). If a checkpoint already exists in the output dir, the inducing geometry is read straight from it instead of re-snapping, otherwise `load_state_dict` would fail on the shape mismatch.

---

## 11. Installation

### Prerequisites

- Python 3.12 (3.10+ works)
- A CUDA GPU for FAISS-GPU, cupy and training. CPU-only runs are possible (`--faiss-cpu-graph/-search/-recon`) but the eigensolve wants either a GPU or the SLEPc path.

### From the repo root

```bash
python -m venv .venv && source .venv/bin/activate

# Installs BOTH l3di and the vendored manifold_gp (see pyproject.toml)
pip install -e .

pip install -r requirements.txt        # torch, gpytorch, cupy-cuda12x, wandb, bg-atlasapi, …
```

### FAISS

FAISS is **not** in `requirements.txt` — in the container it is compiled from source with GPU support (see `manifold/docker/Dockerfile`, `FAISS_VERSION`/`CUDA_ARCHS` build args). Locally, install a wheel:

```bash
pip install faiss-gpu-cu12     # or: pip install faiss-cpu
```

### Optional: SLEPc/PETSc (only for `slepc/slepc_eigensolve.py`)

```bash
# needs gfortran + an MPI; installs petsc4py/slepc4py into the active venv
PREFIX=/opt WITH_MUMPS=1 bash manifold/slepc/build_slepc_petsc.sh
export PETSC_DIR=/opt/petsc SLEPC_DIR=/opt/slepc
export LD_LIBRARY_PATH=/opt/petsc/lib:/opt/slepc/lib:$LD_LIBRARY_PATH
```

### Optional: napari (for `visualizations/` and the toy explorer)

```bash
pip install napari napari-animation
```

### Docker

```bash
# Build from the REPO ROOT — the COPY paths are repo-relative
docker build -f manifold/docker/Dockerfile -t artiomartiom/sdsc:maldi_manifold_all_latest .
```

The image is `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel` plus: `requirements.txt`, FAISS built from source with GPU support, PETSc + SLEPc + MUMPS, an sshd on port 2222, and the dual-mode `entrypoint.sh` — **no arguments** starts sshd for interactive/VS Code Remote work; **with arguments** it activates the venv, drops to `appuser` via `gosu` and execs the command. On every boot it clones/updates the repo at `/myhome/mlibra` and `pip install -e`s it.

`Dockerfile.withslepc` adds only the SLEPc layer on top of the older `artiomartiom/sdsc:withfaiss` image; `Dockerfile.notebook` is a Jupyter/Renku fork (it needs an `sshd_config` at the repo root, which is gitignored and currently absent).

### Data files you need

| File | Used for |
|---|---|
| `maindata_minimal.parquet` | MALDI pixels + CCF coordinates |
| `maindata_minimal_available_lipids.npy` | Lipid channel names |
| `reference_image.npy` | CCF reference template → graph nodes |
| `level_15annot.npy` / `level_5annot.npy` / `ccf_depth{N}annot.npy` | Region labels for the atlas-weighted methods ([§15](#15-atlas-volumes)) |
| `maldi/data/splits/fold_*.json` | CV filters (shared with the baseline) |
| `maldi/data/lipid_subset.txt` | Which lipids get reconstructed/rendered |

---

## 12. Running Locally

The convenient path is the shared runner script, which is the *same* script the cluster executes — every knob is an environment variable with a local default:

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

`run_manifold.sh` composes `EXP_NAME` from every swept hyperparameter (so two configs can never collide in one output dir), translates the env vars into CLI flags, appends the reconstruction lipid list, and calls the Python entry point. The spectral twin is `local_run/run_spectral_manifold.sh`.

Or call Python directly for a minimal smoke run:

```bash
python manifold/lgp_manifold_experiment.py \
  --mode lgp \
  --exp-name manifold_smoke \
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

Start at a coarse stride and a small mode count — the first run pays for the graph build and the eigensolve, every later run with the same geometry is a cache hit.

Training metrics go to **Weights & Biases** (`l3di_maldi_knn_eig` for the graph/eigen phase, the experiment project for training); export `WANDB_API_KEY`, or `WANDB_MODE=offline` to run without it.

---

## 13. Cluster Submission

Jobs go to a **RunAI** cluster. The chain is:

```
./submit/run_manifold_batch.sh          (your laptop)
   └── runai training submit ... -e KEY=VAL ...  -- ./local_run/run_manifold.sh
          └── manifold/lgp_manifold_experiment.py   (in the container)
```

Everything the container needs travels as environment variables; `run_manifold.sh` is the single translation layer from env to CLI, and it is also what you run locally — so local and cluster runs are the same code path with the same experiment names.

**Prerequisite:** a `.env` at the repo root with `export WANDB_API_KEY=...`. The batch scripts refuse to run without it.

```bash
DRY_RUN=1 ./submit/run_manifold_batch.sh    # print the runai commands, submit nothing
./submit/run_manifold_batch.sh              # submit the sweep
```

The sweep grid lives in the middle of the script — edit the arrays and resubmit:

| Array | Sweeps |
|---|---|
| `FOLDS` | `fold-1` … `fold-8` (mapped to `maldi/data/splits/fold_N.json` and to the wandb prefix) |
| `STRIDE_NUM_MODES` | `"stride:modes"` pairs, e.g. `("4:300" "8:6000")` |
| `MAN_KNN_METHODS`, `MAN_INFLATIONS` | Graph method and cross-region inflation |
| `ROOT_HANDLINGS`, `MAN_DENOISE_LABELS`, `MAN_PRUNE_CROSS_REGIONS` | Graph refinement (weighted methods only; plain `faiss` collapses to the no-op value so it is not submitted redundantly) |
| `LAPLACIAN_NORMS`, `GRAPH_BANDWIDTHS`, `NU`, `BUMP_SCALES`, `BUMP_DECAYS`, `THRESHOLDS`, `KNN_K` | Laplacian and kernel |
| `IND_SOURCES` | `reference` / `data` / `blend` |
| `DIFFUSION_SCALES`, `PRODUCT_ARD` | `"LEARN:INIT"` and `"ENABLE:NU"` pairs |

Useful submit-time overrides:

```bash
ATLAS_LEVEL=5 ./submit/run_manifold_batch.sh                 # other annotation volume
BETA=0.1 ./submit/run_manifold_batch.sh                      # KL weight (tags the output dir)
S3_OUTPUT_DIR=/s3/.../my_batch ./submit/run_manifold_batch.sh
FAISS_CPU=1 FORCE_RECOMPUTE_GRAPH=1 ./submit/run_manifold_batch.sh   # CPU-only FAISS timing
N_LIST=sqrt N_PROBE=8 FAISS_CPU=1 ./submit/run_manifold_batch.sh     # fast approximate CPU build
RECONSTRUCTION_LIPIDS_FILE=/path/lipids.txt ./submit/run_manifold_batch.sh
```

Resources per job default to 4 CPU / 48 GB / 0.5 GPU on `artiomartiom/sdsc:maldi_manifold_all_latest`.

### Related submit scripts

| Script | Purpose |
|---|---|
| `submit/run_manifold_batch.sh` | The manifold sweep (inducing-point runner) |
| `submit/run_spectral_manifold_batch.sh` | The spectral twin — no inducing/product-ARD sweeps, adds a `LENGTHSCALE_INIT` sweep; shares all graph/eigvec caches |
| `submit/run_slepc_cache_prepare.sh` | CPU-only, GPU-free eigenpair pre-computation via MPI + SLEPc shift-invert |
| `submit/run_lgp_batch.sh` | The Euclidean LGP baseline, same folds |
| `submit/run_parcel_lgp_batch.sh`, `run_sota_batch.sh`, `run_submit_baselines_sweep.sh`, `run_submit_per_lipid.sh` | The other comparison sweeps on the same folds |

The cluster-ops helpers that surround these (a keep-alive pod for SSH / VS Code Remote, resumable rsync over the pod's port-forward, run-directory downloads, and job query / cancel / retry utilities) are `.gitignore`d local-only tooling and are not part of the repo. Nothing in the training or submission path depends on them: the interactive workflow they wrap is just

```bash
runai workspace port-forward <job> --port 2222:2222 &
ssh -p 2222 appuser@localhost          # or: runai workspace exec <job> -it -- bash
rsync -avP -e 'ssh -p 2222' ./localdir/ appuser@localhost:/dest/
```

with the pod submitted **without a command**, so the image's entrypoint starts `sshd` instead of dropping into job mode.

---

## 14. Output Directory Layout

`EXP_NAME` is assembled by `run_manifold.sh` so that every swept hyperparameter appears in it — otherwise two configs would share a directory and clobber each other's checkpoints:

```
{EXP_PREFIX}-MANIFOLD-{RSAMPLE|MEAN}-{latent}-{stride}-K{modes}-{template}-{threshold}
  -{ind_source}-{num_inducing}-{batch}-{knn_method}-{inflation}-{knn_k}-{lap_norm}
  -{nu}-{bump_scale}-{bump_decay}-{bandwidth}
  [-learndiff{init} | -diff{init}] [-prodard{nu}] [-blend{frac}]
  [-clk{K}-sw{w}-cs{seed}] [-{atlas_stem}] [-root{mode}] [-dn{N}] [-prune{F}] [-beta{β}]
```

Conditional suffixes are only appended when the knob is non-default, so historical directory names (and the checkpoints inside them) stay valid.

```
{OUTPUT_DIR}/{EXP_NAME}/
├── args.npy                  # exact CLI arguments
├── config.json               # human-readable snapshot of the same
├── model.pth                 # trained weights (also the source of truth for inducing geometry)
├── train_means.pth, train_stds.pth, colmean.pth, colstd.pth
├── checkpoints/model_{epoch}.pth
├── train/                    # predictions.npy, true_values.npy, scatter plots
├── test/                     # predictions.npy, true_values.npy, scatter plots
├── volume/                   # dense per-lipid whole-brain volumes ({lipid}_volume.npy)
├── volume_sparse/            # written instead of volume/ under --render-voxels-only
└── renders/                  # composite figures, {lipid}_diagnostics.png, {lipid}_error_slice.png
```

`--render-voxels-only` (the default in `run_manifold.sh`, `RENDER_VOXELS_ONLY=1`) reconstructs only the voxels the composite render actually reads — the slice planes plus the 3-D MIP stride — roughly 5.5× fewer voxels for a near-identical figure. Set `RENDER_VOXELS_ONLY=0` if you need dense volumes for napari or the analysis scripts.

---

## 15. Atlas Volumes

Two families of annotation volume, selected by `ATLAS_LEVEL`:

| `ATLAS_LEVEL` | File | What it is |
|---|---|---|
| `5`, `15` | `level_{N}annot.npy` | The LBAE-shipped **partial** cuts. Note that `level_15` has only 68 regions and its `root` (id 997) covers ~57% of the brain — which is why `--root-handling dissolve` matters so much here. |
| `d5`, `d7` | `ccf_depth{N}annot.npy` | True CCF depth-cuts produced by `download_bg_atlas.py --max-depth N` (175 / 476 regions; root shrinks to well under 1%) |

Generate the depth-cut volumes yourself:

```bash
python manifold/download_bg_atlas.py \
  --reference-file   /path/bg_template.npy \
  --annotations-file /path/ccf_depth7annot.npy \
  --max-depth 7
```

It pulls `allen_mouse_25um` through the BrainGlobe API and writes both volumes (skipping the download if both files already exist). `coarsen_annotation` walks each structure's `structure_id_path` and collapses leaves to the requested ancestor depth: `1 → 5` labels, `4 → 86`, `5 → 176`, `7 → 477`, `9+ → 672` (the raw leaves). The Allen tree bottoms out at depth 9 — anything deeper is identical.

Because the atlas filename stem is part of every cache key and of `EXP_NAME`, switching levels never collides with existing caches or output directories.

---

## 16. Tooling — Benchmarks, Visualisation, Analysis

Each `benchmarks/*.py` has a matching `.sh` with a worked invocation; every tool reads the **same** graph/eigvec caches as training, so nothing is recomputed.

### `benchmarks/`

| Tool | Question it answers |
|---|---|
| `bump_support_report.py` | How many train/test MALDI points fall inside the kernel's bump support? (Points outside get zeroed manifold features.) |
| `nystrom_benchmark.py` | How accurate is the Nyström out-of-sample extension, scored against a finer-stride ground truth or against real MALDI coordinates? |
| `graph_bandwidth_sweep.py` | Which bandwidth keeps the heat-kernel affinity neither saturated nor collapsed, given the edge-length distribution? |
| `spectral_distance_sweep.py` + `spectral_sweep_report.py` | How well does the measured lipid covariance decay along each candidate distance metric? Sweep, then rank. |
| `eigensolver_compare.py` | cupy Lanczos vs the alternatives on the bottom of the spectrum. |

### `visualizations/` (napari)

| Tool | Shows |
|---|---|
| `manifold_vs_euclidean_explorer.py` | Hand-tune (or load from a trained run) a Euclidean and a manifold kernel side by side, hit **Refit**, and read off the border-budget numbers the analysis notebooks produce |
| `graph_visualization_utils.py` | The numeric engine behind that explorer — runs the border-budget analysis on any kernel, and is usable standalone via its own CLI |
| `visualize_raw_data.py` | The raw MALDI voxels on the CCF template — no model, no kernel, just where the data actually lives |

### `notebooks/`

`manifold_boundary_narrative.ipynb` (why the kernel behaves the way it does at region boundaries) and `laplacian_distance_distributions.ipynb` (which distance metric actually explains the lipid data), with their pooled anchor caches (`pools_fold2.npz`, `fold2_test_pool.npz`) checked in so they run without a rebuild.

### `toy_example/`

Synthetic geometries where the manifold kernel *should* win: `toy_manifold.py` (a folded 2-D sheet in 3-D, so geodesically-distant points come close in Euclidean space), `toy_box_cylinder.py`, and `toy_gp_explorer_napari.py` to explore them interactively. Check the sampling density against the fold gap before drawing conclusions — an undersampled roll produces kNN edges that jump layers, and then you are measuring a graph artefact rather than a kernel property.

### Local-only (`.gitignore`d, absent from a clone)

Three directories stay in the author's working copy rather than the repo. They are listed here so the pipeline they exercise is documented, not because a clone can run them:

- **`viz/`** — the napari explorer fleet: the FAISS graph over the brain with click-to-drop ROIs, the three nested kNN-fabric → Laplacian → kernel layers, how much of the Laplacian's action on a field the first N eigenmodes recover, a MALDI ↔ kernel correlation explorer, per-lipid GP predictions read straight from disk, and batch-wide render/re-inference fillers for missing `renders/`.
- **`experiments/`** — one-off probes: does the graph short-circuit across anatomical gaps, do cross-region edges land on real interfaces, are the cached eigenpairs sane (including the `D^{-1/2}` reweighting between `L_sym` and random-walk eigenvectors), and do the Euclidean and manifold models actually differ (with a constant-collapse check).
- **`analysis/`** — the figure notebooks and the scripts that build their caches: the headline benchmark ranking, the distance-metric study, the border story, and the spectral-truncation capacity work. The two narrative notebooks that ship live in `notebooks/` instead.

---

## 17. Known Pitfalls

- **Modes vs node density.** The usable mode count is capped by `--stride`/`--threshold`. More modes than the node spacing supports adds aliased noise, not resolution.
- **Inducing points vs modes.** The kernel is finite-rank (`= --num-modes`). `--num-inducing > --num-modes` makes `K_uu` singular and buys nothing.
- **Bandwidth.** At `--graphbandwidth-init 0.1` the graph is close to unweighted (affinity contrast ≈ 1.03), which throttles a ×50 atlas inflation down to a ~21× effective prior. ≈0.05 keeps the prior meaningful — check with `benchmarks/graph_bandwidth_sweep.py`.
- **Denoise before prune.** Aggressive `--denoise-labels` erases a large share of the true boundary *before* the prune sees it, so the prune then removes far fewer real cross-region edges than the fraction suggests.
- **Prune and connectivity.** A high `--prune-cross-region` can shatter the graph into several components; the eigensolve will happily return the resulting near-zero modes.
- **Root catch-all.** With the shipped partial atlases, leaving `--root-handling cross` lets the root label act as a giant pseudo-region and undoes the confinement the inflation is meant to produce. `dissolve` is the default for a reason.
- **Bump support.** Test points outside every bump get zeroed manifold features and fall back to the prior — silently. Run `benchmarks/bump_support_report.py` before trusting a comparison.
- **Snap-count drift.** The FAISS graph is not bit-reproducible, so the deduplicated inducing count varies between runs; that is why an existing checkpoint's inducing geometry always wins over a fresh snap.

---

## See Also

- [`../README.md`](../README.md) — the project overview, the `l3di` library, and the Euclidean `maldi/` baseline
- [`manifold_gp/README.md`](manifold_gp/README.md) — the upstream IMGP readme and citation
