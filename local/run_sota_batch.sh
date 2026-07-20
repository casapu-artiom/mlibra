#!/usr/bin/env bash
# Local counterpart of submit/run_sota_batch.sh: run the SOTA sweep (NTF / Spa3D /
# DeepSpatial / GPLFR) on THIS machine instead of submitting to run:ai.
#
# The sweep grid, fold loop, TAG -> EXP_PREFIX naming and job ordering are kept
# IDENTICAL to the submit script, so a local run and a cluster run of the same
# (fold, model, config) land in the same-named experiment dir and stay directly
# comparable. The only differences are mechanical:
#
#   * runs SEQUENTIALLY in this shell (one at a time) instead of N parallel pods;
#   * I/O keeps sota/run_sota.sh's LOCAL defaults instead of the /s3 + /myhome
#     mounts (override any of them from the environment -- see "Paths" below);
#   * W&B is OFF by default (no .env / API key needed to try things locally);
#   * a failing run does NOT abort the sweep -- failures are counted and listed
#     at the end, so an overnight sweep survives one bad config;
#   * every run's stdout+stderr is tee'd to LOG_DIR/<job>.log.
#
#   ./local/run_sota_batch.sh                       # default models, folds, full sweep
#   MODELS="ntf" ./local/run_sota_batch.sh          # just NTF (still swept)
#   FOLDS="fold-2" MODELS="ntf" SWEEP=0 ./local/run_sota_batch.sh   # single smoke run
#   DRY_RUN=1 ./local/run_sota_batch.sh             # print what would run + count
#   WANDB=1 ./local/run_sota_batch.sh               # log to W&B (needs .env)
#   RETRIES=2 ./local/run_sota_batch.sh             # retry each failing run twice
#   RESUME=1 ./local/run_sota_batch.sh              # skip runs already completed
#   RESUME=1 RETRIES=2 ./local/run_sota_batch.sh    # <- re-run ONLY the failures
#
# Resume/retry: a completed run's stem is appended to STATE_FILE
# ($LOG_DIR/completed.txt); RESUME=1 skips those. Only successes are recorded, so
# RESUME=1 on a second pass re-attempts exactly the runs that failed. Delete the
# ledger (or a line in it) to force a re-run.
set -uo pipefail          # NOT -e: one failing run must not kill the sweep

DRY_RUN=${DRY_RUN:-0}
SWEEP=${SWEEP:-1}                     # 1 = full per-model grid; 0 = single default config
# Retry a run that exits non-zero, RETRIES extra times (0 = no retry). Only worth
# >0 for transient failures (OOM under load, a flaky mount); a deterministic bug
# will just fail RETRIES+1 times and slow the sweep down.
RETRIES=${RETRIES:-0}
RETRY_DELAY=${RETRY_DELAY:-10}        # seconds between attempts
# RESUME=1 skips runs already recorded complete in STATE_FILE, so an interrupted
# sweep can be re-launched and pick up where it stopped.
#
# NOTE this is a SWEEP-level skip, not a mid-run resume: it only avoids re-running
# a job that already finished. ntf/spa3d never persist a model (sota/run_sota.py
# has no torch.save and its --skip-training flag is dead), so a job that died
# part-way retrains from scratch regardless. gplfr resumes for free
# (MaldiExperiment.run() reloads model.pth) and deepspatial has its own *.ckpt
# reload; both keep whatever the reconstruct cache / render skip-if-exists saved.
RESUME=${RESUME:-0}
N_JOBS=0
N_FAIL=0
N_SKIP=0
FAILED=()

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)

# W&B off by default locally: the submit script needs .env for the cluster, but a
# local smoke run shouldn't. WANDB=1 opts in and then .env is sourced if present.
WANDB=${WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-sota_maldi}
if [ "$WANDB" = "1" ]; then
    if [ -f "$REPO_ROOT/.env" ]; then
        # shellcheck disable=SC1091
        source "$REPO_ROOT/.env"
    else
        echo "WARNING: WANDB=1 but no .env at $REPO_ROOT/.env; W&B will likely fail." >&2
    fi
fi

# ---- Paths (LOCAL defaults; every one overridable from the environment) -----
# These mirror sota/run_sota.sh's own defaults. Left unset here, the runner would
# use the same values anyway -- they're spelled out so a local sweep is
# self-describing and easy to repoint at a different data dir.
DATA_PATH=${DATA_PATH:-/home/casap/mlibra/mlibra_data}
OUTPUT_DIR=${OUTPUT_DIR:-/home/casap/mlibra/output/sota_batch}
MALDI_FILE=${MALDI_FILE:-/home/casap/mlibra/mlibra_data/maindata_minimal.parquet}
REFERENCE_FILE=${REFERENCE_FILE:-/home/casap/mlibra/mlibra_data/reference_image.npy}
ANNOTATION_FILE=${ANNOTATION_FILE:-/home/casap/mlibra/mlibra_data/level_15annot.npy}
AVAILABLE_LIPIDS_FILE=${AVAILABLE_LIPIDS_FILE:-/home/casap/mlibra/mlibra_data/maindata_minimal_available_lipids.npy}
# Only GPLFR's riemann/spectral bases need these.
EIGENVECTOR_DIR=${EIGENVECTOR_DIR:-/home/casap/mlibra/output/eigenvectors}
SRC_PATH=${SRC_PATH:-$REPO_ROOT}
# GPLFR euclidean base GP: per-axis ARD ON by default for the local sweep (set
# GPLFR_ARD=0 to go back to a single isotropic lengthscale). Ignored by the
# riemann/spectral bases. Kept as a script-level default (not a grid value) so the
# per-model sweep grid stays byte-identical to submit/run_sota_batch.sh.
GPLFR_ARD=${GPLFR_ARD:-1}
SPLITS_DIR=${SPLITS_DIR:-$REPO_ROOT/maldi/data/splits}
LOG_DIR=${LOG_DIR:-$OUTPUT_DIR/logs}
EXP_SUFFIX=${EXP_SUFFIX:-"local-$(date +'%y%m%d-%H-%M')"}
# Completion ledger for RESUME. Keyed by the job STEM (sota-<model>-<tag>-<fold>),
# which is stable across invocations -- unlike the job name, which carries
# EXP_SUFFIX's timestamp. Deterministic EXP_NAME means a re-run of the same stem
# targets the same output dir, so skipping it is safe.
STATE_FILE=${STATE_FILE:-$LOG_DIR/completed.txt}

# ---------------------------------------------------------------------------
# Per-model sweep grids -- kept byte-identical to submit/run_sota_batch.sh so the
# two produce the same TAGs (and therefore the same EXP_PREFIX / output dirs).
# If you edit a grid, edit it in BOTH files.
# ---------------------------------------------------------------------------
sweep_for_model() {
    case "$1" in
        ntf)
            echo "res128-tv05:N_EPOCHS=30 BATCH_SIZE=16384 NTF_MAX_RES=128 NTF_WEIGHT_DECAY=0.0001 NTF_TV_WEIGHT=0.05"
            echo "res256-tv05:N_EPOCHS=30 BATCH_SIZE=16384 NTF_MAX_RES=256 NTF_WEIGHT_DECAY=0.0001 NTF_TV_WEIGHT=0.05"
            echo "res512-tv05:N_EPOCHS=30 BATCH_SIZE=16384 NTF_MAX_RES=512 NTF_WEIGHT_DECAY=0.0001 NTF_TV_WEIGHT=0.05"
            echo "res256-tv20:N_EPOCHS=30 BATCH_SIZE=16384 NTF_MAX_RES=256 NTF_WEIGHT_DECAY=0.0001 NTF_TV_WEIGHT=0.2"
            echo "res256-tv00:N_EPOCHS=30 BATCH_SIZE=16384 NTF_MAX_RES=256 NTF_WEIGHT_DECAY=0.0001 NTF_TV_WEIGHT=0.0"
            ;;
        spa3d)
            echo "zw03-none:N_EPOCHS=30 BATCH_SIZE=4096 SPA3D_Z_WEIGHT=0.3 SPA3D_SPE=none"
            echo "zw01-none:N_EPOCHS=30 BATCH_SIZE=4096 SPA3D_Z_WEIGHT=0.1 SPA3D_SPE=none"
            echo "zw05-none:N_EPOCHS=30 BATCH_SIZE=4096 SPA3D_Z_WEIGHT=0.5 SPA3D_SPE=none"
            echo "zw03-none-nodes150k:N_EPOCHS=30 BATCH_SIZE=4096 SPA3D_Z_WEIGHT=0.3 SPA3D_SPE=none SPA3D_GRAPH_NODES=150000"
            echo "zw03-alft:N_EPOCHS=30 BATCH_SIZE=4096 SPA3D_Z_WEIGHT=0.3 SPA3D_SPE=alft"
            echo "zw03-hilbert:N_EPOCHS=30 BATCH_SIZE=4096 SPA3D_Z_WEIGHT=0.3 SPA3D_SPE=hilbert"
            ;;
        deepspatial)
            echo "cross-reg03:N_EPOCHS=100 BATCH_SIZE=256 DS_PAIRING=cross-mouse DS_UOT_REG=0.3 DS_MAX_CELLS=8000"
            echo "cross-reg08:N_EPOCHS=100 BATCH_SIZE=256 DS_PAIRING=cross-mouse DS_UOT_REG=0.8 DS_MAX_CELLS=8000"
            echo "within-reg03:N_EPOCHS=100 BATCH_SIZE=256 DS_PAIRING=within-mouse DS_UOT_REG=0.3 DS_MAX_CELLS=8000"
            ;;
        gplfr)
            echo "euclidean:N_EPOCHS=20 BATCH_SIZE=2000 BASE_GP=euclidean"
            echo "riemann:N_EPOCHS=20 BATCH_SIZE=2000 BASE_GP=riemann"
            echo "spectral:N_EPOCHS=20 BATCH_SIZE=2000 BASE_GP=spectral"
            ;;
        *)
            echo "ERROR: no sweep grid defined for model '$1'" >&2
            return 1
            ;;
    esac
}

is_done() {   # stem already recorded complete?
    [ "$RESUME" = "1" ] && [ -f "$STATE_FILE" ] && grep -qxF "$1" "$STATE_FILE"
}

# run_one <job_stem> <job_name> <model> <slices> <prefix> <env_str> [extra args...]
run_one() {
    local stem=$1 job_name=$2 model=$3 slices=$4 prefix=$5 env_str=$6
    shift 6
    local extra_args=("$@")
    local log="$LOG_DIR/${job_name}.log"

    if is_done "$stem"; then
        echo ">>> [$((N_JOBS + 1))] $stem  -- already complete (RESUME=1); skipping"
        N_SKIP=$((N_SKIP + 1))
        return 0
    fi

    echo ">>> [$((N_JOBS + 1))] $job_name  (model=$model, sweep='$env_str')"
    if [ "$DRY_RUN" = "1" ]; then
        echo "    [DRY] env $env_str MODEL=$model EXP_PREFIX=$prefix \\"
        [ "$model" = "gplfr" ] && echo "          GPLFR_ARD=$GPLFR_ARD \\"
        echo "          SLICES_DATASET_FILE=$slices OUTPUT_DIR=$OUTPUT_DIR \\"
        echo "          ./sota/run_sota.sh ${extra_args[*]}"
        return 0
    fi

    mkdir -p "$LOG_DIR"
    local attempt=0 rc=0
    while : ; do
        # env_str is "K=V K=V ..." from the grid line; pass it plus the shared I/O
        # to the runner's environment. Word-splitting it is intentional (same as
        # the submit script turning it into -e pairs).
        # shellcheck disable=SC2086
        ( cd "$REPO_ROOT" && env $env_str \
            MODEL="$model" \
            EXP_PREFIX="$prefix" \
            DATA_PATH="$DATA_PATH" \
            OUTPUT_DIR="$OUTPUT_DIR" \
            MALDI_FILE="$MALDI_FILE" \
            SLICES_DATASET_FILE="$slices" \
            AVAILABLE_LIPIDS_FILE="$AVAILABLE_LIPIDS_FILE" \
            TEMPLATE_NAME="reference" \
            REFERENCE_FILE="$REFERENCE_FILE" \
            ANNOTATION_FILE="$ANNOTATION_FILE" \
            EIGENVECTOR_DIR="$EIGENVECTOR_DIR" \
            SRC_PATH="$SRC_PATH" \
            GPLFR_ARD="$GPLFR_ARD" \
            WANDB="$WANDB" \
            WANDB_PROJECT="$WANDB_PROJECT" \
            ./sota/run_sota.sh "${extra_args[@]}" ) 2>&1 | tee -a "$log"
        # tee is last in the pipe, so grab the runner's status, not tee's.
        rc=${PIPESTATUS[0]}
        [ "$rc" -eq 0 ] && break
        [ "$attempt" -ge "$RETRIES" ] && break
        attempt=$((attempt + 1))
        echo "    !! exit $rc -- retry $attempt/$RETRIES in ${RETRY_DELAY}s"
        sleep "$RETRY_DELAY"
    done

    if [ "$rc" -eq 0 ]; then
        # Record only on success, so a re-run with RESUME=1 retries the failures.
        mkdir -p "$(dirname "$STATE_FILE")"
        echo "$stem" >> "$STATE_FILE"
        [ "$attempt" -gt 0 ] && echo "    ok after $attempt retry/retries"
    else
        echo "    !! FAILED (exit $rc after $attempt retry/retries) -- see $log"
        FAILED+=("$stem (exit $rc)")
        N_FAIL=$((N_FAIL + 1))
    fi
    return 0
}

MODELS=${MODELS:-"ntf spa3d deepspatial gplfr"}   # add 'gplfr' to also sweep the latent-GP
FOLDS=("fold-2" "fold-7")    # shared CV list

for fold in "${FOLDS[@]}"; do
    fold_upper=${fold^^}
    fold_file=${fold//-/_}
    SLICES_DATASET_FILE="${SPLITS_DIR}/${fold_file}.json"
    if [ ! -f "$SLICES_DATASET_FILE" ] && [ "$DRY_RUN" != "1" ]; then
        echo "ERROR: split file not found: $SLICES_DATASET_FILE" >&2
        exit 1
    fi
    for model in $MODELS; do
        configs=$(sweep_for_model "$model") || exit 1
        [ "$SWEEP" = "0" ] && configs=$(echo "$configs" | head -1)
        while IFS= read -r cfg; do
            [ -z "$cfg" ] && continue
            tag=${cfg%%:*}
            env_str=${cfg#*:}
            tag_upper=${tag^^}
            prefix="${fold_upper}-${tag_upper}"
            stem="sota-${model}-${tag}-${fold}"
            job="${stem}-${EXP_SUFFIX}"
            run_one "$stem" "$job" "$model" "$SLICES_DATASET_FILE" "$prefix" "$env_str" "$@"
            N_JOBS=$((N_JOBS + 1))
        done <<<"$configs"
    done
done

echo
echo "=== ${N_JOBS} run(s) $([ "$DRY_RUN" = "1" ] && echo 'would run' || echo 'attempted')," \
     "${N_SKIP} skipped (already complete)" \
     "(models='$MODELS', folds='${FOLDS[*]}', sweep=$SWEEP) ==="
if [ "$N_FAIL" -gt 0 ]; then
    echo "=== ${N_FAIL} FAILED: ==="
    for f in "${FAILED[@]}"; do echo "    - $f"; done
    echo "    logs:  $LOG_DIR"
    echo "    retry: RESUME=1 RETRIES=2 $0    # re-runs ONLY the failures above"
    exit 1
fi
[ "$DRY_RUN" = "1" ] || echo "    logs: $LOG_DIR    ledger: $STATE_FILE"
