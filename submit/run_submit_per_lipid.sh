#!/usr/bin/env bash
# =============================================================================
# Per-lipid GP runai batch submitter.
#
# Mirrors run_manifold_batch.sh but for the per-lipid pipeline
# (./maldi/run_lgp_per_lipid.sh → lgp_experiment_per_lipid.py).
#
# What it does:
#   - Loops over a small cross-product of hyperparameters
#     (KNN_METHOD × KNN_K × INFLATION × …)
#   - Submits one runai job per config, each running
#     ./maldi/run_lgp_per_lipid.sh inside the standard container
#   - Each job trains only the lipids listed in $LIPIDS_FILE
#     (default: 10 lipids — fast enough for parallel sweeps)
#
# Usage:
#     DRY_RUN=1 ./run_lgp_per_lipid_batch.sh   # print commands only
#     ./run_lgp_per_lipid_batch.sh             # actually submit
# =============================================================================
set -euo pipefail

DRY_RUN=${DRY_RUN:-0}
run_or_echo() {
    if [ "$DRY_RUN" = "1" ]; then echo "[DRY] $*"; else "$@"; fi
}

# ---- load secrets ---------------------------------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
else
    echo "ERROR: .env not found at $SCRIPT_DIR/.env" >&2
    echo "Create it with: export WANDB_API_KEY=..." >&2
    exit 1
fi

# ---- resource limits ------------------------------------------------------
MEM=48G
CPU=4
GPU=0.5

# ---- training hyperparameters (constant across sweep) ---------------------
# Per-lipid run knobs. EPOCHS=20 is the published default; drop to 5 for
# quick smoke-tests in the early-sweep phase. LIPID_BATCH_SIZE=10 fits
# our 10-lipid subset in a single GP fit.
EPOCHS=10
LIPID_BATCH_SIZE=10
NUM_INDUCING=1000
LEARNING_RATE=0.005
BATCH_SIZE=2048
STRIDE_BUMP=20.0   # bump_scale — fixed for this sweep
STRIDE_DECAY=0.01  # bump_decay — fixed for this sweep

# ---- (stride : num_modes) pairs to sweep (manifold only) ------------------
# Coarser strides need fewer eigenmodes to span the graph; finer strides need
# more. Each entry is "STRIDE:NUM_MODES" and gets its own job.
MAN_STRIDE_MODES=("4:1300" "8:6000")

# ---- lengthscale modes (manifold only) ------------------------------------
# 1 → fixed lengthscale (--lengthscale-init 8.0 --lengthscale-no-decay)
# 0 → learned lengthscale (no flags; GP trains it).
# Both modes are submitted so they can be compared head-to-head.
MAN_FIXED_LS=(1 0)

# ---- subset of lipids to train -------------------------------------------
# A small subset is what makes a 30-job sweep practical. The file lives
# inside the container at the same path the run script expects.
LIPIDS_FILE="/myhome/mlibra/maldi/data/lipid_subset.txt"

# ---- S3-mounted paths inside the container --------------------------------
S3_DATA_PATH="/s3/mlibra/mlibra-data/maldi/"
S3_EIGENVECTOR_DIR="/s3/mlibra/mlibra-data/artiom/eigenvectors"
S3_OUTPUT_DIR="/s3/mlibra/mlibra-data/artiom/per_lipid_batch_5"
S3_MALDI_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal.parquet"
S3_TEMPLATE_NAME="reference"
S3_REFERENCE_FILE="/s3/mlibra/mlibra-data/reference_image.npy"
S3_ANNOTATION_FILE="/s3/mlibra/mlibra-data/level_15annot.npy"
S3_SLICES_DATASET_FILE="/myhome/mlibra/maldi/data/splits/fold_3.json"
S3_AVAILABLE_LIPIDS_FILE="/s3/mlibra/mlibra-data/maldi/maindata_minimal_available_lipids.npy"
SRC_PATH="/myhome/mlibra"

EXP_SUFFIX="artiom-$(date +'%y%m%d-%H-%M')"

# ---- one submission -------------------------------------------------------
# All hyperparameters get passed as -e env vars; run_lgp_per_lipid.sh
# inside the container reads them via ``: "${X:=default}"`` and forwards
# to lgp_experiment_per_lipid.py.
submit() {
    local job_name=$1
    local family=$2        # euclidean | manifold
    local nu=$3
    local knn_method=$4
    local knn_k=$5
    local laplacian_norm=$6
    local graphbandwidth=$7
    local inflation=$8
    local threshold=$9
    local prefix=${10}
    local slice=${11}
    local stride=${12}
    local modes=${13}
    local fixed_ls=${14}
    shift 14
    local extra_args=("$@")

    echo ">>> Submitting $job_name"
    run_or_echo runai training submit "$job_name" \
        -i artiomartiom/sdsc:maldi_manifold_latest \
        --cpu-core-limit "$CPU" --cpu-core-request "$CPU" \
        --cpu-memory-limit "$MEM" --cpu-memory-request "$MEM" \
        --gpu-request-type portion --gpu-portion-request "$GPU" \
        -e EXP_PREFIX="$prefix" \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e SRC_PATH="$SRC_PATH" \
        -e DATA_PATH="$S3_DATA_PATH" \
        -e OUTPUT_DIR="$S3_OUTPUT_DIR" \
        -e EIGENVECTOR_DIR="$S3_EIGENVECTOR_DIR" \
        -e MALDI_FILE="$S3_MALDI_FILE" \
        -e SLICES_DATASET_FILE="$slice" \
        -e AVAILABLE_LIPIDS_FILE="$S3_AVAILABLE_LIPIDS_FILE" \
        -e TEMPLATE_NAME="$S3_TEMPLATE_NAME" \
        -e REFERENCE_FILE="$S3_REFERENCE_FILE" \
        -e ANNOTATION_FILE="$S3_ANNOTATION_FILE" \
        -e LIPIDS_FILE="$LIPIDS_FILE" \
        -e KERNEL_FAMILY="$family" \
        -e NU="$nu" \
        -e KNN_METHOD="$knn_method" \
        -e KNN_K="$knn_k" \
        -e LAPLACIAN_NORM="$laplacian_norm" \
        -e GRAPHBANDWIDTH="$graphbandwidth" \
        -e CROSS_REGION_INFLATION="$inflation" \
        -e BUMP_SCALE="$STRIDE_BUMP" \
        -e BUMP_DECAY="$STRIDE_DECAY" \
        -e NUM_MODES="$modes" \
        -e STRIDE="$stride" \
        -e FIXED_LENGTHSCALE="$fixed_ls" \
        -e THRESHOLD="$threshold" \
        -e NUM_INDUCING="$NUM_INDUCING" \
        -e LIPID_BATCH_SIZE="$LIPID_BATCH_SIZE" \
        -e EPOCHS="$EPOCHS" \
        -e LEARNING_RATE="$LEARNING_RATE" \
        -e BATCH_SIZE="$BATCH_SIZE" \
        -- ./maldi/run_lgp_per_lipid.sh "${extra_args[@]}"
}

# =============================================================================
# Sweep definitions
#
# Two separate loops below: one for the Euclidean baseline (only the
# kernel + nu vary; no kNN / Laplacian / inflation knobs apply), and
# one for the Manifold family (all the graph-spectral knobs). This
# keeps the per-family job count honest — submitting `nu=2` for
# euclidean only submits one job, not one per (knn_k, knn_method, ...)
# permutation that the value would otherwise duplicate over.
# =============================================================================

FOLDS=("fold-1" "fold-2" "fold-3" "fold-4" "fold-5" "fold-6" "fold-7" "fold-8")

# ---- Euclidean sweep ------------------------------------------------------
# Euclidean Matern only has nu as a meaningful hyperparameter — the GP
# learns its own lengthscale per dimension. Other knobs (kernel="matern"
# etc.) are fixed in run_lgp_per_lipid.sh.
EUC_NU=(1.5)
RUN_EUCLIDEAN=1   # 0 = skip the euclidean loop entirely

# ---- Manifold sweep -------------------------------------------------------
MAN_NU=(2)
MAN_THRESHOLDS=(5 40)
MAN_LAPLACIAN_NORMS=("randomwalk")
MAN_GRAPH_BANDWIDTHS=(0.1)
MAN_KNN_K=(15)
MAN_KNN_METHODS=("faiss" "faiss_atlas_weighted")
MAN_INFLATIONS=(50)   # only used when knn_method=faiss_atlas_weighted
RUN_MANIFOLD=1

exp_num=1
for fold in "${FOLDS[@]}"; do
    fold_upper=${fold^^}
    fold_file=${fold//-/_}
    SLICES_DATASET_FILE="/myhome/mlibra/maldi/data/splits/${fold_file}.json"

    # ---- EUCLIDEAN loop ---------------------------------------------------
    if [ "$RUN_EUCLIDEAN" = "1" ]; then
        for nu in "${EUC_NU[@]}"; do
            job_name="gp-perlipid-${EXP_SUFFIX}-${exp_num}"
            printf "  exp %2d: %-30s nu=%-4s fold=%s\n" \
                "$exp_num" "euclidean" "$nu" "$fold"
            # Euclidean ignores knn_method / laplacian / inflation /
            # threshold / graphbandwidth, but the submit() function takes
            # them positionally — pass dummy values that the python side
            # will silently ignore for kernel_family=euclidean.
            # stride/modes/fixed_ls are manifold-only; pass placeholders.
            submit "$job_name" "euclidean" "$nu" \
                "faiss" "15" "randomwalk" "0.1" "1" "5" \
                "$fold_upper" "$SLICES_DATASET_FILE" "4" "1300" "1"
            exp_num=$((exp_num + 1))
        done
    fi

    # ---- MANIFOLD loop ----------------------------------------------------
    if [ "$RUN_MANIFOLD" = "1" ]; then
        for nu in "${MAN_NU[@]}"; do
            for threshold in "${MAN_THRESHOLDS[@]}"; do
                for ln in "${MAN_LAPLACIAN_NORMS[@]}"; do
                    for gb in "${MAN_GRAPH_BANDWIDTHS[@]}"; do
                        for knn in "${MAN_KNN_K[@]}"; do
                            for km in "${MAN_KNN_METHODS[@]}"; do
                                # Inflation is only meaningful for
                                # faiss_atlas_weighted; for the other
                                # methods, sweep just one placeholder so
                                # we don't submit redundant jobs.
                                if [ "$km" = "faiss_atlas_weighted" ]; then
                                    infl_list=("${MAN_INFLATIONS[@]}")
                                else
                                    infl_list=(1)
                                fi
                                for infl in "${infl_list[@]}"; do
                                  for sm in "${MAN_STRIDE_MODES[@]}"; do
                                    stride="${sm%%:*}"
                                    modes="${sm##*:}"
                                    for fixed_ls in "${MAN_FIXED_LS[@]}"; do
                                        job_name="gp-perlipid-${EXP_SUFFIX}-${exp_num}"
                                        # Fold the lengthscale mode into the
                                        # prefix so the two modes (and their
                                        # output dirs) never collide.
                                        if [ "$fixed_ls" = "1" ]; then
                                            ls_tag="fixls"
                                        else
                                            ls_tag="learnls"
                                        fi
                                        prefix="${fold_upper}-${ls_tag}"
                                        printf "  exp %2d: %-22s nu=%s knn_k=%-3s ln=%s gb=%s infl=%s stride=%s modes=%s ls=%s\n" \
                                            "$exp_num" "$km" "$nu" "$knn" "$ln" "$gb" "$infl" "$stride" "$modes" "$ls_tag"
                                        submit "$job_name" "manifold" "$nu" \
                                            "$km" "$knn" "$ln" "$gb" "$infl" "$threshold" \
                                            "$prefix" "$SLICES_DATASET_FILE" \
                                            "$stride" "$modes" "$fixed_ls"
                                        exp_num=$((exp_num + 1))
                                    done
                                  done
                                done
                            done
                        done
                    done
                done
            done
        done
    fi
done

echo "Submitted $((exp_num - 1)) jobs (DRY_RUN=$DRY_RUN)."
echo "Results land under $S3_OUTPUT_DIR/<EXP_NAME>/"
echo ""
echo "To visualise once done:"
echo "  RUN_DIR=$S3_OUTPUT_DIR/<one-run-name> ./maldi/visualize_lipid_gp.sh"
echo ""
echo "To aggregate metrics across runs:"
echo "  python -c 'import pandas as pd, pathlib as P; \\"
echo "    print(pd.concat([pd.read_csv(d/\"metrics.csv\").assign(run=d.name) \\"
echo "      for d in P.Path(\"$S3_OUTPUT_DIR\").iterdir() \\"
echo "      if (d/\"metrics.csv\").exists()]).groupby(\"run\")[[\"test_corr\", \"test_rmse_z\"]].mean())'"