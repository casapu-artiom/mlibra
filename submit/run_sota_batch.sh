#!/usr/bin/env bash
# Submit the SOTA 3D-reconstruction papers (NTF / Spa3D / DeepSpatial / GPLFR) on
# MALDI to run:ai, one job per (fold, model, sweep-config). Mirrors
# run_submit_baselines.sh.
#
# Structure:
#   * FOLDS is a single CV list, shared across ALL models (outer loop).
#   * each model has its OWN sweep grid (sweep_for_model) over the knobs that
#     actually matter for it -- so a CV run also sweeps the right hyperparams.
#   * the sweep TAG is folded into the job name AND EXP_PREFIX, so every
#     (fold, model, config) writes to a distinct output dir (no clobbering).
#
# The runner (local_run/run_sota.sh) already specifies every input/output path with a
# LOCAL default; this script overrides the I/O env vars to point at the
# S3-mounted dirs. Each job does whole-brain reconstruction + renders + per-lipid
# diagnostics (RECONSTRUCT=whole_brain), comparable to the manifold/baseline runs.
#
#   ./submit/run_sota_batch.sh                          # default models, folds, full sweep
#   MODELS="ntf" ./submit/run_sota_batch.sh             # just NTF (still swept)
#   MODELS="ntf spa3d deepspatial gplfr" ...            # add GPLFR (needs EIGENVECTOR_DIR)
#   FOLDS="fold-1 fold-2 fold-3" ./submit/run_sota_batch.sh
#   SWEEP=0 ./submit/run_sota_batch.sh                  # one default config per model (no sweep)
#   DRY_RUN=1 ./submit/run_sota_batch.sh                # print the runai commands + job count only
set -euo pipefail

DRY_RUN=${DRY_RUN:-0}
SWEEP=${SWEEP:-1}                     # 1 = full per-model grid; 0 = single default config
N_JOBS=0

run_or_echo() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY] $*"
    else
        "$@"
    fi
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
elif [ "$DRY_RUN" = "1" ]; then
    echo "[DRY] .env not found; continuing with a placeholder WANDB_API_KEY" >&2
    WANDB_API_KEY=${WANDB_API_KEY:-DRY_RUN_PLACEHOLDER}
else
    echo "ERROR: .env not found at $SCRIPT_DIR/.env" >&2
    echo "Create it with: export WANDB_API_KEY=..." >&2
    exit 1
fi

# ---- resource tiers (per model) --------------------------------------------
# spa3d gets the HIGH tier (heavier 3D graph); every other model uses LOW.
# All env-overridable.
LOW_GPU=${LOW_GPU:-0.2};  LOW_CPU=${LOW_CPU:-2};  LOW_MEM=${LOW_MEM:-32G}
HIGH_GPU=${HIGH_GPU:-0.5}; HIGH_CPU=${HIGH_CPU:-2}; HIGH_MEM=${HIGH_MEM:-48G}

# W&B on by default for cluster runs (WANDB_API_KEY comes from .env). WANDB=0 disables.
WANDB=${WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-sota_maldi}

# ---- S3-mounted I/O overrides (the runner defaults are LOCAL) --------------
S3_DATA_PATH="/s3/mlibra/mlibra-data/maldi/"
S3_OUTPUT_DIR=${S3_OUTPUT_DIR:-"/s3/mlibra/mlibra-data/artiom/sota_batch_cv"}
S3_MALDI_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet"
S3_REFERENCE_FILE="/s3/mlibra/mlibra-data/reference_image.npy"
S3_ANNOTATION_FILE="/s3/mlibra/mlibra-data/level_15annot.npy"
S3_AVAILABLE_LIPIDS_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy"
# Precomputed eigenvectors -- only GPLFR's riemann/spectral bases need these.
S3_EIGENVECTOR_DIR=${S3_EIGENVECTOR_DIR:-"/s3/mlibra/mlibra-data/artiom/eigenvectors"}
SRC_PATH="/myhome/mlibra"
# job-name-only suffix -- kept short because runai caps workload names at 50 chars.
EXP_SUFFIX="$(date +'%m%d-%H%M')"

# ---------------------------------------------------------------------------
# Per-model sweep grids. One config per "echo" line: "TAG:ENV1=v1 ENV2=v2 ...".
#   - TAG (lowercase, dash-safe) is folded into the job name + EXP_PREFIX.
#   - every param (incl. N_EPOCHS + BATCH_SIZE) is spelled out ON THE LINE, so a
#     line is fully self-contained: comment it out to drop that config, copy it to
#     add one, edit any value in place. Duplication is intentional (readability).
# SWEEP=0 runs only the FIRST line of each model's grid.
# Knobs chosen (see the design discussion): NTF -> regularization (weight-decay/TV);
# Spa3D -> z_weight (3D-ness) + SPE on/off; DeepSpatial -> pairing + UOT sharpness;
# GPLFR -> the base GP (riemann/spectral need EIGENVECTOR_DIR).
# ---------------------------------------------------------------------------
sweep_for_model() {
    case "$1" in
        ntf)
            echo "res128-tv05:N_EPOCHS=50 BATCH_SIZE=16384 NTF_MAX_RES=128 NTF_WEIGHT_DECAY=0.0001 NTF_TV_WEIGHT=0.05"
            echo "res256-tv05:N_EPOCHS=50 BATCH_SIZE=16384 NTF_MAX_RES=256 NTF_WEIGHT_DECAY=0.0001 NTF_TV_WEIGHT=0.05"
            echo "res512-tv05:N_EPOCHS=50 BATCH_SIZE=16384 NTF_MAX_RES=512 NTF_WEIGHT_DECAY=0.0001 NTF_TV_WEIGHT=0.05"
            echo "res256-tv20:N_EPOCHS=50 BATCH_SIZE=16384 NTF_MAX_RES=256 NTF_WEIGHT_DECAY=0.0001 NTF_TV_WEIGHT=0.2"
            echo "res256-tv00:N_EPOCHS=50 BATCH_SIZE=16384 NTF_MAX_RES=256 NTF_WEIGHT_DECAY=0.0001 NTF_TV_WEIGHT=0.0"
            ;;
        spa3d)
            echo "zw03-none:N_EPOCHS=50 BATCH_SIZE=4096 SPA3D_Z_WEIGHT=0.3 SPA3D_SPE=none"
            echo "zw01-none:N_EPOCHS=50 BATCH_SIZE=4096 SPA3D_Z_WEIGHT=0.1 SPA3D_SPE=none"
            echo "zw05-none:N_EPOCHS=50 BATCH_SIZE=4096 SPA3D_Z_WEIGHT=0.5 SPA3D_SPE=none"
            echo "zw03-none-nodes150k:N_EPOCHS=50 BATCH_SIZE=4096 SPA3D_Z_WEIGHT=0.3 SPA3D_SPE=none SPA3D_GRAPH_NODES=150000"
            echo "zw03-alft:N_EPOCHS=50 BATCH_SIZE=4096 SPA3D_Z_WEIGHT=0.3 SPA3D_SPE=alft"
            echo "zw03-hilbert:N_EPOCHS=50 BATCH_SIZE=4096 SPA3D_Z_WEIGHT=0.3 SPA3D_SPE=hilbert"
            ;;
        deepspatial)
            echo "cross-reg03:N_EPOCHS=100 BATCH_SIZE=256 DS_PAIRING=cross-mouse DS_UOT_REG=0.3 DS_MAX_CELLS=8000"
            echo "cross-reg08:N_EPOCHS=100 BATCH_SIZE=256 DS_PAIRING=cross-mouse DS_UOT_REG=0.8 DS_MAX_CELLS=8000"
            echo "within-reg03:N_EPOCHS=100 BATCH_SIZE=256 DS_PAIRING=within-mouse DS_UOT_REG=0.3 DS_MAX_CELLS=8000"
            ;;
        gplfr)
            echo "euclidean:N_EPOCHS=30 BATCH_SIZE=2000 BASE_GP=euclidean"
            echo "riemann:N_EPOCHS=30 BATCH_SIZE=2000 BASE_GP=riemann"
            echo "spectral:N_EPOCHS=30 BATCH_SIZE=2000 BASE_GP=spectral"
            ;;
        *)
            echo "ERROR: no sweep grid defined for model '$1'" >&2
            return 1
            ;;
    esac
}


submit() {
    local job_name=$1 model=$2 slices=$3 prefix=$4 env_str=$5
    shift 5
    local extra_args=("$@")           # forwarded verbatim to run_sota.sh
    # env_str carries N_EPOCHS / BATCH_SIZE (from the per-model base) plus the
    # config's own knobs -- all passed straight through as -e vars.
    local sweep_env=()
    for kv in $env_str; do sweep_env+=(-e "$kv"); done
    # per-model resource tier: spa3d -> high, everything else -> low
    local gpu cpu mem tier
    if [ "$model" = "spa3d" ]; then
        tier=high; gpu=$HIGH_GPU; cpu=$HIGH_CPU; mem=$HIGH_MEM
    else
        tier=low;  gpu=$LOW_GPU;  cpu=$LOW_CPU;  mem=$LOW_MEM
    fi
    echo ">>> Submitting $job_name (model=$model, tier=$tier: gpu=$gpu cpu=$cpu mem=$mem, sweep='$env_str')"
    runai training submit "$job_name" \
        -i artiomartiom/sdsc:maldi_manifold_all_latest \
        --cpu-core-limit "$cpu" --cpu-core-request "$cpu" \
        --cpu-memory-limit "$mem" --cpu-memory-request "$mem" \
        --gpu-request-type portion --gpu-portion-request "$gpu" \
        -e EXP_PREFIX="$prefix" \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e MODEL="$model" \
        -e DATA_PATH="$S3_DATA_PATH" \
        -e OUTPUT_DIR="$S3_OUTPUT_DIR" \
        -e MALDI_FILE="$S3_MALDI_FILE" \
        -e SLICES_DATASET_FILE="$slices" \
        -e AVAILABLE_LIPIDS_FILE="$S3_AVAILABLE_LIPIDS_FILE" \
        -e TEMPLATE_NAME="reference" \
        -e REFERENCE_FILE="$S3_REFERENCE_FILE" \
        -e ANNOTATION_FILE="$S3_ANNOTATION_FILE" \
        -e EIGENVECTOR_DIR="$S3_EIGENVECTOR_DIR" \
        -e SRC_PATH="$SRC_PATH" \
        -e WANDB="$WANDB" \
        -e WANDB_PROJECT="$WANDB_PROJECT" \
        "${sweep_env[@]}" \
        -- ./local_run/run_sota.sh "${extra_args[@]}"
}
MODELS=${MODELS:-"ntf spa3d deepspatial gplfr"}   # add 'gplfr' to also sweep the latent-GP
FOLDS=("fold-1" "fold-2" "fold-3" "fold-4" "fold-5" "fold-6" "fold-7" "fold-8")

# Outer loop: folds shared across all models. Inner: per-model sweep grid.
for fold in "${FOLDS[@]}"; do
    fold_upper=${fold^^}
    fold_file=${fold//-/_}
    SLICES_DATASET_FILE="/myhome/mlibra/maldi/data/splits/${fold_file}.json"
    for model in $MODELS; do
        configs=$(sweep_for_model "$model")
        [ "$SWEEP" = "0" ] && configs=$(echo "$configs" | head -1)   # first config only
        while IFS= read -r cfg; do
            [ -z "$cfg" ] && continue
            tag=${cfg%%:*}                 # part before the first ':'
            env_str=${cfg#*:}              # "N_EPOCHS=.. BATCH_SIZE=.. KEY=VAL .."
            tag_upper=${tag^^}
            prefix="${fold_upper}-${tag_upper}"
            # Job name stays short (runai caps workload names at 50): the sweep
            # tag lives in EXP_PREFIX / the output dir, so the job name uses a
            # running counter for uniqueness instead. e.g. sota-deepspatial-fold3-0718-2322-7
            job="sota-${model}-${fold//-/}-${EXP_SUFFIX}-${N_JOBS}"
            run_or_echo submit "$job" "$model" "$SLICES_DATASET_FILE" \
                "$prefix" "$env_str" "$@"
            N_JOBS=$((N_JOBS + 1))
        done <<<"$configs"
    done
done

echo "=== ${N_JOBS} job(s) $([ "$DRY_RUN" = "1" ] && echo 'would be ' )submitted"\
     "(models='$MODELS', folds='${FOLDS[*]}', sweep=$SWEEP) ==="
