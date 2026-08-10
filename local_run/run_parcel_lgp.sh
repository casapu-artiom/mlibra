#!/usr/bin/env bash
# =============================================================================
# LGP path: LATENT_DIM latent GPs -> MLP decoder -> all lipid channels, with and
# without the reference-only parcel factor.
#
#   ./local_run/run_parcel_lgp.sh                    # baseline + parcel, back to back
#   MODE=parcel   ./local_run/run_parcel_lgp.sh
#   MODE=baseline ./local_run/run_parcel_lgp.sh
#
# Runs other_experiments/parcelgp/lgp_parcel_experiment.py, which is maldi/lgp_experiment.py with
# the parcel factor inserted into the kernel -- so training, checkpoint resume,
# normalization, prediction, whole-brain reconstruction and rendering are all the
# EXISTING pipeline, and the numbers are comparable to your other LGP runs.
# Without --parcel-field it is byte-for-byte the standard LGP, which is what
# makes the baseline arm a true ablation rather than an approximation.
#
# Directory naming: every parameter that changes the model is in EXP_NAME (plus
# the parcel settings, appended by the python entrypoint). This matters more than
# it looks -- MaldiExperiment.run() loads and SKIPS TRAINING when it finds a
# model.pth in the experiment directory, so two configs sharing a directory means
# the second silently reports the first's results. That is exactly the class of
# bug that made a stride-1 parcel field return stride-4 numbers.
#
# Compare arms with:
#   python -m other_experiments.parcelgp.compare --baseline base=DIR --run parcel=DIR --metric corr
# (per-lipid paired; ~60x tighter than comparing run means)
# =============================================================================
set -euo pipefail

# ---- data ----
: "${DATA_PATH:=/home/casap/mlibra/mlibra_data}"
: "${MALDI_FILE:=$DATA_PATH/maindata_minimal.parquet}"
: "${REFERENCE_FILE:=$DATA_PATH/reference_image.npy}"
: "${ANNOTATION_FILE:=$DATA_PATH/level_15annot.npy}"
: "${AVAILABLE_LIPIDS_FILE:=$DATA_PATH/maindata_minimal_available_lipids.npy}"
: "${SRC_PATH:=/home/casap/mlibra_git}"
: "${SLICES_DATASET_FILE:=$SRC_PATH/maldi/data/splits/fold_2.json}"
: "${OUTPUT_DIR:=/home/casap/mlibra/output/parcel_lgp}"
: "${EXP_PREFIX:=FOLD-2}"

# ---- parcellation (built/verified by parcelgp.build; see its --force) ----
: "${PARCEL_DIR:=$DATA_PATH/parcels}"
: "${N_PARCELS:=128}"
: "${PARCEL_FEATURES:=full}"        # full | simple | spatial
: "${SPATIAL_WEIGHT:=3.0}"          # 2.3 balances geometry vs appearance 50/50
: "${STRIDE:=4}"
: "${THRESHOLD:=5}"
# 1 = weight each descriptor BLOCK equally in the k-means distance instead of
# each feature. Off = current behaviour (hessian 37.5% of the distance, depth
# 6.3%, purely by feature count); on = 20% each. Neither is known to be better.
: "${NORMALIZE_BLOCKS:=0}"
NB_ARG=""; NB_TAG=""
[ "$NORMALIZE_BLOCKS" = "1" ] && { NB_ARG="--normalize-blocks"; NB_TAG="_nb"; }
# Filename carries every parameter that changes the parcellation; parcelgp.build
# additionally verifies the cached field's recorded parameters and refuses to
# reuse a mismatched one.
: "${PARCEL_FIELD:=$PARCEL_DIR/${PARCEL_FEATURES}_k${N_PARCELS}_sw${SPATIAL_WEIGHT}_s${STRIDE}_t${THRESHOLD}${NB_TAG}.npz}"

# ---- parcel factor ----
: "${PARCEL_RANK:=8}"               # width of the learned parcel embedding
: "${PARCEL_INIT_SCALE:=0.05}"      # std of B init; 0 = exact no-op at init
: "${PARCEL_SHARED_B:=0}"           # 1 = one B shared across the latents

# ---- model / training (names mirror local_run/run_lgp.sh) ----
: "${LATENT_DIM:=5}"
: "${NUM_INDUCING_POINTS:=1000}"
: "${INDUCING_SOURCE:=reference}"   # reference | data
: "${KERNEL:=matern}"               # matern | rbf | symmetric
: "${NU:=1.5}"                      # Euclidean Matern accepts 0.5 | 1.5 | 2.5
: "${N_PIXELS:=10}"                 # sets the lengthscale floor
: "${BATCH_SIZE:=1000}"
: "${N_EPOCHS:=2}"
: "${LEARNING_RATE:=0.001}"
: "${SEED:=416465}"
: "${MODE_ARG:=lgp}"                # MaldiConfig 'mode'
: "${NO_RSAMPLE:=true}"             # true -> decode the posterior mean
: "${LEARN_INDUCING:=true}"
: "${ARD:=true}"
: "${LOG_TRANSFORM:=true}"
: "${DO_BRAIN_RECONSTRUCTION:=1}"   # 1 = whole-brain volumes + rendered figures
                                    # (restricted to RECONSTRUCTION_LIPIDS_FILE below;
                                    #  set 0 for a metrics-only sweep)
: "${RENDER_VOXELS_ONLY:=1}"        # (reconstruction) only the voxels the figure reads
# Which lipids get reconstructed/rendered. Reconstruction is O(voxels x lipids)
# over the whole brain -- 34M voxels at stride 1 -- so it MUST be restricted to a
# subset or it dominates the run. One lipid name per line; '#' comments ignored.
: "${RECONSTRUCTION_LIPIDS_FILE:=$SRC_PATH/maldi/data/lipid_subset.txt}"

: "${MODE:=parcel}"                   # both | parcel | baseline

# =============================================================================
mkdir -p "$PARCEL_DIR"
# Always invoke the builder: it verifies the cached field's recorded build
# parameters and errors on a mismatch rather than trusting the filename.
echo "== parcel field: $PARCEL_FIELD"
python -m other_experiments.parcelgp.build --reference-file "$REFERENCE_FILE" \
    --out "$PARCEL_FIELD" --n-parcels "$N_PARCELS" \
    --features "$PARCEL_FEATURES" --spatial-weight "$SPATIAL_WEIGHT" \
    --stride "$STRIDE" --threshold "$THRESHOLD" $NB_ARG

flags=()
tag_bits=()
if [ "$NO_RSAMPLE" = "true" ] || [ "$NO_RSAMPLE" = "1" ]; then
    flags+=(--no-rsample); tag_bits+=("MEAN")
else
    tag_bits+=("RSAMPLE")
fi
if [ "$LEARN_INDUCING" = "true" ] || [ "$LEARN_INDUCING" = "1" ]; then
    flags+=(--learn-inducing); tag_bits+=("learnind")
else
    tag_bits+=("fixind")
fi
if [ "$ARD" = "true" ] || [ "$ARD" = "1" ]; then
    flags+=(--ard); tag_bits+=("ard")
else
    tag_bits+=("iso")
fi
# NOTE: MaldiConfig itself appends N_PIXELS and '_log' to exp_name, so those two
# are already in the directory name and are deliberately not repeated here.
[ "$LOG_TRANSFORM" = "true" ] || [ "$LOG_TRANSFORM" = "1" ] && flags+=(--log-transform)
if [ "$DO_BRAIN_RECONSTRUCTION" = "1" ]; then
    flags+=(--do-brain-reconstruction)
    [ "$RENDER_VOXELS_ONLY" = "1" ] && flags+=(--render-voxels-only)
    # --reconstruction-lipids is nargs='+', and lipid names contain spaces
    # ("PC 35:1 PE 38:1"), so each line must become ONE argv entry. Reading into
    # a bash array does that correctly; run_final.sh achieves the same with an
    # IFS=newline trick, which is easier to break.
    if [ -f "$RECONSTRUCTION_LIPIDS_FILE" ]; then
        recon=()
        while IFS= read -r _line || [ -n "$_line" ]; do
            # Trim CR (Windows line endings) and surrounding whitespace, matching
            # run_final.sh's python line.strip(). lipid_subset.txt contains a
            # whitespace-ONLY line, which would otherwise be passed through as a
            # lipid named " " and silently match nothing.
            _line="${_line%$'\r'}"
            _line="${_line#"${_line%%[![:space:]]*}"}"
            _line="${_line%"${_line##*[![:space:]]}"}"
            case "$_line" in ''|\#*) continue ;; esac
            recon+=("$_line")
        done < "$RECONSTRUCTION_LIPIDS_FILE"
        if [ ${#recon[@]} -gt 0 ]; then
            flags+=(--reconstruction-lipids "${recon[@]}")
            echo "== reconstructing ${#recon[@]} lipid(s) from $(basename "$RECONSTRUCTION_LIPIDS_FILE")"
        fi
    else
        echo "WARNING: RECONSTRUCTION_LIPIDS_FILE=$RECONSTRUCTION_LIPIDS_FILE not found;" >&2
        echo "         reconstruction will run over ALL lipids, which is very slow." >&2
    fi
fi

# Every knob above that changes the fitted model appears here. The fold comes
# from the splits file's basename so runs on different folds never collide.
FOLD_TAG="$(basename "$SLICES_DATASET_FILE" .json)"
IFS=- eval 'TAGS="${tag_bits[*]}"'
BASE_NAME="${EXP_PREFIX}-LGP-${FOLD_TAG}-d${LATENT_DIM}-${INDUCING_SOURCE}${NUM_INDUCING_POINTS}"
BASE_NAME="${BASE_NAME}-${KERNEL}nu${NU}-bs${BATCH_SIZE}-lr${LEARNING_RATE}-ep${N_EPOCHS}"
BASE_NAME="${BASE_NAME}-${TAGS}-s${SEED}"

common=(
    --exp-name "$BASE_NAME"
    --mode "$MODE_ARG"
    --dataset-path "$DATA_PATH"
    --maldi-file "$MALDI_FILE"
    --output-dir "$OUTPUT_DIR"
    --available-lipids-file "$AVAILABLE_LIPIDS_FILE"
    --slices-dataset-file "$SLICES_DATASET_FILE"
    --template-name reference
    --reference-file "$REFERENCE_FILE"
    --annotations-file "$ANNOTATION_FILE"
    --latent-dim "$LATENT_DIM"
    --num-inducing "$NUM_INDUCING_POINTS"
    --inducing-source "$INDUCING_SOURCE"
    --kernel "$KERNEL"
    --nu "$NU"
    --n-pixels "$N_PIXELS"
    --batch-size "$BATCH_SIZE"
    --epochs "$N_EPOCHS"
    --learning-rate "$LEARNING_RATE"
    --seed "$SEED"
    "${flags[@]}"
)

run_baseline() {
    echo "== BASELINE (no parcel factor)"
    python -m other_experiments.parcelgp.lgp_parcel_experiment "${common[@]}"
}
run_parcel() {
    echo "== PARCEL ($PARCEL_FEATURES k=$N_PARCELS rank=$PARCEL_RANK)"
    local extra=(--parcel-field "$PARCEL_FIELD"
                 --parcel-rank "$PARCEL_RANK"
                 --parcel-init-scale "$PARCEL_INIT_SCALE")
    [ "$PARCEL_SHARED_B" = "1" ] && extra+=(--parcel-shared-B)
    # The entrypoint appends the parcel settings to --exp-name itself, so the two
    # arms cannot share a directory.
    python -m other_experiments.parcelgp.lgp_parcel_experiment "${common[@]}" "${extra[@]}"
}

cd "$SRC_PATH"
case "$MODE" in
    baseline) run_baseline ;;
    parcel)   run_parcel ;;
    both)     run_baseline; run_parcel ;;
    *) echo "MODE must be both|parcel|baseline, got '$MODE'" >&2; exit 2 ;;
esac

_suffix="${N_PIXELS}"
{ [ "$LOG_TRANSFORM" = "true" ] || [ "$LOG_TRANSFORM" = "1" ]; } && _suffix="${_suffix}_log"
_parcel_tag="parcel$(basename "$PARCEL_FIELD" .npz)-r${PARCEL_RANK}"
[ "$PARCEL_SHARED_B" = "1" ] && _parcel_tag="${_parcel_tag}-sharedB"
echo
echo "python -m other_experiments.parcelgp.compare \\"
echo "    --baseline base=$OUTPUT_DIR/${BASE_NAME}${_suffix} \\"
echo "    --run parcel=$OUTPUT_DIR/${BASE_NAME}-${_parcel_tag}${_suffix} --metric corr"
