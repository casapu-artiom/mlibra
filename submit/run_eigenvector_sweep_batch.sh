#!/usr/bin/env bash
# Batch submitter: loops the sweep grid and submits one runai job per
# (knn_method, normalization, threshold, knn_k, graphbandwidth,
# cross_region_inflation) cell. Each job runs run_eigenvector_sweep.sh
# with its sweep values pushed in as env vars.
#
# Override any axis from the calling shell, e.g.
#     THRESHOLDS="5 20" KNN_KS="60 120" ./run_eigenvector_sweep_batch.sh
# or edit the defaults inline.

set -euo pipefail

DRY_RUN=${DRY_RUN:-0}

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
else
    echo "ERROR: .env not found at $SCRIPT_DIR/.env" >&2
    echo "Create it with: export WANDB_API_KEY=..." >&2
    exit 1
fi

# -------------------------------------------------------------------------
# Cluster resources
# -------------------------------------------------------------------------
MEM=${MEM:-48G}
CPU=${CPU:-4}
GPU=${GPU:-0.5}
IMAGE=${IMAGE:-artiomartiom/sdsc:maldi_manifold_latest}

# -------------------------------------------------------------------------
# Paths (S3 mounts on the cluster)
# -------------------------------------------------------------------------
S3_EIGENVECTOR_DIR="${S3_EIGENVECTOR_DIR:-/s3/mlibra/mlibra-data/eigenvectors}"
S3_OUTPUT_DIR="${S3_OUTPUT_DIR:-/s3/mlibra/mlibra-data/artiom/psd_sweep}"
S3_TEMPLATE_NAME="${S3_TEMPLATE_NAME:-reference}"
S3_REFERENCE_FILE="${S3_REFERENCE_FILE:-/s3/mlibra/mlibra-data/reference_image.npy}"
S3_ANNOTATION_FILE="${S3_ANNOTATION_FILE:-/s3/mlibra/mlibra-data/level_15annot.npy}"
SRC_PATH="${SRC_PATH:-/myhome/mlibra}"

# -------------------------------------------------------------------------
# Sweep grid. Override any list from the env (space-separated).
#   KNN_METHODS="faiss anatomical_atlas" ./run_eigenvector_sweep_batch.sh
# -------------------------------------------------------------------------
KNN_METHODS=${KNN_METHODS:-"faiss anatomical_atlas faiss_atlas_weighted"}
NORMS=${NORMS:-"symmetric randomwalk"}
THRESHOLDS=${THRESHOLDS:-"5 20 40 150"}
KNN_KS=${KNN_KS:-"15 60 120 180"}
BANDWIDTHS=${BANDWIDTHS:-"0.05 0.1 0.5 1.0"}
# Inflation only applies to faiss_atlas_weighted; iterated only for it below.
INFLATIONS=${INFLATIONS:-"1 5 10 100"}
# Non-swept defaults (set in env to override globally for this batch)
NUM_MODES=${NUM_MODES:-1300}
NU=${NU:-2}
LENGTHSCALE=${LENGTHSCALE:-1.0}
BUMP_SCALE=${BUMP_SCALE:-3.0}
BUMP_DECAY=${BUMP_DECAY:-0.05}
N_TEST_ON=${N_TEST_ON:-200}
N_TEST_OFF=${N_TEST_OFF:-200}
TEST_SEED=${TEST_SEED:-42}

EXP_SUFFIX="${EXP_SUFFIX:-$(date +'%y%m%d-%H%M')}"

# -------------------------------------------------------------------------
# Slug helpers (job names & filenames must be k8s/filesystem-safe)
# -------------------------------------------------------------------------
slug() { echo "$1" | sed 's/\./p/g; s/-/n/g'; }
method_short() {
    case "$1" in
        faiss)                echo "fa" ;;
        anatomical_atlas)     echo "aa" ;;
        faiss_atlas_weighted) echo "fw" ;;
        *) echo "$1" ;;
    esac
}
norm_short() {
    case "$1" in
        symmetric)  echo "sym" ;;
        randomwalk) echo "rw"  ;;
        *) echo "$1" ;;
    esac
}

# -------------------------------------------------------------------------
# Submit one runai job. Per-config sweep values are forwarded to the inner
# shell as env vars (-e KEY=VALUE) — the inner shell consumes them via the
# `: "${KEY:=default}"` pattern, so anything we set here overrides the local
# default and anything we don't set falls back to the local default.
# -------------------------------------------------------------------------
submit_one() {
    local job_name=$1; shift
    local out_csv=$1;  shift
    local method=$1; norm=$2; thr=$3; k=$4; bw=$5; infl=$6
    runai training submit "$job_name" \
        -i "$IMAGE" \
        --cpu-core-limit "$CPU"     --cpu-core-request   "$CPU" \
        --cpu-memory-limit "$MEM"   --cpu-memory-request "$MEM" \
        --gpu-request-type portion  --gpu-portion-request "$GPU" \
        -e WANDB_API_KEY="$WANDB_API_KEY" \
        -e OUTPUT_DIR="$S3_OUTPUT_DIR" \
        -e EIGENVECTOR_DIR="$S3_EIGENVECTOR_DIR" \
        -e TEMPLATE_NAME="$S3_TEMPLATE_NAME" \
        -e REFERENCE_FILE="$S3_REFERENCE_FILE" \
        -e ANNOTATION_FILE="$S3_ANNOTATION_FILE" \
        -e SRC_PATH="$SRC_PATH" \
        -e OUT_CSV="$out_csv" \
        -e KNN_METHOD="$method" \
        -e NORMALIZATION="$norm" \
        -e THRESHOLD="$thr" \
        -e KNN_K="$k" \
        -e GRAPHBANDWIDTH="$bw" \
        -e CROSS_REGION_INFLATION="$infl" \
        -e NUM_MODES="$NUM_MODES" \
        -e NU="$NU" \
        -e LENGTHSCALE="$LENGTHSCALE" \
        -e BUMP_SCALE="$BUMP_SCALE" \
        -e BUMP_DECAY="$BUMP_DECAY" \
        -e N_TEST_ON="$N_TEST_ON" \
        -e N_TEST_OFF="$N_TEST_OFF" \
        -e TEST_SEED="$TEST_SEED" \
        -- ./maldi/run_laplacian_test.sh
}

# -------------------------------------------------------------------------
# Sweep loop
# -------------------------------------------------------------------------
n_submitted=0
for method in $KNN_METHODS; do
    m_short=$(method_short "$method")
    for norm in $NORMS; do
        n_short=$(norm_short "$norm")
        for thr in $THRESHOLDS; do
            for k in $KNN_KS; do
                for bw in $BANDWIDTHS; do
                    # Only iterate the inflation axis for faiss_atlas_weighted;
                    # for the other methods we submit a single job (inflation
                    # is unused and excluded from their cache keys).
                    if [ "$method" = "faiss_atlas_weighted" ]; then
                        inflation_list="$INFLATIONS"
                    else
                        inflation_list="default"   # sentinel; not used in slug
                    fi
                    for infl in $inflation_list; do
                        bw_slug=$(slug "$bw")
                        if [ "$infl" = "default" ]; then
                            graph_slug="${m_short}-${n_short}-t${thr}-k${k}-bw${bw_slug}"
                            infl_val="10.0"   # placeholder; ignored for non-weighted methods
                        else
                            infl_slug=$(slug "$infl")
                            graph_slug="${m_short}-${n_short}-t${thr}-k${k}-bw${bw_slug}-i${infl_slug}"
                            infl_val="$infl"
                        fi
                        # Filename slug carries every sweep-able param so the
                        # output dir is self-documenting. The job name doesn't
                        # — it's just `lap-<batch suffix>-<counter>`, which is
                        # short and guaranteed unique within the batch.
                        full_slug="${graph_slug}-bs$(slug "$BUMP_SCALE")-bd$(slug "$BUMP_DECAY")-nu${NU}-ls$(slug "$LENGTHSCALE")-nm${NUM_MODES}"

                        n_submitted=$((n_submitted + 1))
                        job_name="lap-${EXP_SUFFIX}-$(printf '%04d' "$n_submitted")"
                        out_csv="${S3_OUTPUT_DIR}/${EXP_SUFFIX}/${full_slug}.csv"

                        echo ">>> [$n_submitted] $job_name -> ${full_slug}.csv"
                        run_or_echo submit_one "$job_name" "$out_csv" \
                            "$method" "$norm" "$thr" "$k" "$bw" "$infl_val"
                    done
                done
            done
        done
    done
done

echo "Submitted $n_submitted jobs. Suffix: $EXP_SUFFIX"
echo "Outputs: $S3_OUTPUT_DIR/$EXP_SUFFIX/*.csv"