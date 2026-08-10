#!/usr/bin/env bash
# =============================================================================
# Run the per-lipid GP with the reference-only parcel factor — and its ablation.
#
#   ./local_run/run_parcel_per_lipid.sh              # baseline + parcel, back to back
#   MODE=parcel   ./local_run/run_parcel_per_lipid.sh
#   MODE=baseline ./local_run/run_parcel_per_lipid.sh
#   LIMIT=4 EPOCHS=1 ./local_run/run_parcel_per_lipid.sh      # quick smoke test
#
# Defaults to running BOTH arms because a parcel run on its own tells you
# nothing: the only question is whether it beats the identical model without the
# factor. Both arms share every hyperparameter, seed and split — the single
# difference is --parcel-field.
#
# The parcel field is built on first use (~10s) and cached. It never sees lipid
# data, so it is built once and reused by every lipid, fold and arm.
# =============================================================================
set -euo pipefail

# ---- data ----
: "${DATA_PATH:=/home/casap/mlibra/mlibra_data}"
: "${MALDI_FILE:=$DATA_PATH/maindata_minimal.parquet}"
: "${REFERENCE_FILE:=$DATA_PATH/reference_image.npy}"
: "${AVAILABLE_LIPIDS_FILE:=$DATA_PATH/maindata_minimal_available_lipids.npy}"
: "${SRC_PATH:=/home/casap/mlibra_git}"
: "${SLICES_DATASET_FILE:=$SRC_PATH/maldi/data/splits/fold_2.json}"
: "${OUTPUT_DIR:=/home/casap/mlibra/output/parcel}"
: "${EXP_PREFIX:=FOLD-2}"

# ---- parcellation (see other_experiments/parcelgp/README.md for what these do) ----
: "${PARCEL_DIR:=$DATA_PATH/parcels}"
: "${N_PARCELS:=128}"
: "${PARCEL_FEATURES:=full}"        # full | simple | spatial
: "${SPATIAL_WEIGHT:=3.0}"          # higher = more spatially compact parcels
: "${STRIDE:=2}"
: "${THRESHOLD:=5}"
# 1 = weight each descriptor BLOCK equally in the k-means distance instead of
# each feature. Off = current behaviour (hessian 37.5% of the distance, depth
# 6.3%, purely by feature count); on = 20% each. Neither is known to be better.
: "${NORMALIZE_BLOCKS:=0}"
NB_ARG=""; NB_TAG=""
[ "$NORMALIZE_BLOCKS" = "1" ] && { NB_ARG="--normalize-blocks"; NB_TAG="_nb"; }
# The filename carries every build parameter that changes the parcellation.
# parcelgp.build ALSO verifies the cached field's recorded parameters and refuses
# to reuse a mismatched one, so a knob missing from this name can no longer
# silently train against the wrong field.
: "${PARCEL_FIELD:=$PARCEL_DIR/${PARCEL_FEATURES}_k${N_PARCELS}_sw${SPATIAL_WEIGHT}_s${STRIDE}_t${THRESHOLD}${NB_TAG}.npz}"

# ---- parcel factor ----
: "${PARCEL_RANK:=8}"               # width of the learned parcel embedding
: "${PARCEL_INIT_SCALE:=0.05}"      # std of B init; 0 = exact no-op at init
: "${PARCEL_SHARED_B:=0}"           # 1 = one B shared across the lipid batch

# ---- GP ----
: "${KERNEL_FAMILY:=euclidean}"
: "${KERNEL:=matern}"
# NOTE: nu must be 0.5/1.5/2.5 for the Euclidean MaternKernel. The per-lipid
# runner's own default of 2.0 is only valid for the Riemann kernel and crashes
# the euclidean family.
: "${NU:=2.5}"
: "${NO_ARD:=0}"                    # 1 = isotropic lengthscale (matches the existing baseline)
: "${NUM_INDUCING:=1000}"
: "${INDUCING_SOURCE:=data}"
: "${LIPID_BATCH_SIZE:=10}"
: "${EPOCHS:=10}"
: "${LEARNING_RATE:=0.005}"
: "${BATCH_SIZE:=2048}"
: "${SEED:=42}"
: "${DEVICE:=cuda}"

# ---- which lipids ----------------------------------------------------------
# Default is ALL lipids in AVAILABLE_LIPIDS_FILE (173). Note this differs from
# local_run/run_lgp_per_lipid.sh, which defaults to the 5-lipid smoke subset — runs
# from the two scripts are not comparable unless you set LIPIDS_FILE to match.
#
#   LIPIDS_FILE=$SRC_PATH/maldi/data/lipid_subset.txt   # the 5-lipid subset
#   LIMIT=20                                            # first 20 lipids
#
# For the ablation itself, prefer a real subset over LIMIT: 5 lipids is too few
# to separate a small effect from fold noise, and both arms need the SAME set
# (they get it automatically — the selection is in common_args).
: "${LIPIDS_FILE:=$SRC_PATH/maldi/data/lipid_subset.txt}"                # text file, one lipid name per line
: "${LIMIT:=}"                      # fit only the first N lipids (smoke tests)
# ---- rendering ----
# lgp_experiment_per_lipid.py renders by DEFAULT; stated explicitly here so both
# parcelgp runners declare the same behaviour instead of one inheriting it
# silently. RENDER_MAX_LIPIDS caps how many trained lipids get a PNG.
: "${RENDER:=1}"
: "${RENDER_MAX_LIPIDS:=8}"

: "${WANDB:=0}"
: "${WANDB_PROJECT:=l3di_parcel}"

: "${MODE:=both}"                   # both | parcel | baseline

# =============================================================================
mkdir -p "$PARCEL_DIR"
# Always invoke the builder: it decides whether the cached field matches these
# parameters, reuses it if so, and ERRORS if it was built with different ones.
# The old `if [ ! -f ]` guard trusted the filename, which silently reused a
# stride-4 field for a stride-1 run.
echo "== parcel field: $PARCEL_FIELD"
python -m other_experiments.parcelgp.build \
    --reference-file "$REFERENCE_FILE" \
    --out "$PARCEL_FIELD" \
    --n-parcels "$N_PARCELS" \
    --features "$PARCEL_FEATURES" \
    --spatial-weight "$SPATIAL_WEIGHT" \
    --stride "$STRIDE" \
    --threshold "$THRESHOLD" $NB_ARG

common_args=(
    --kernel-family "$KERNEL_FAMILY"
    --output-dir "$OUTPUT_DIR"
    --dataset-path "$DATA_PATH"
    --maldi-file "$MALDI_FILE"
    --available-lipids-file "$AVAILABLE_LIPIDS_FILE"
    --slices-dataset-file "$SLICES_DATASET_FILE"
    --template-name reference
    --reference-file "$REFERENCE_FILE"
    --kernel "$KERNEL"
    --nu "$NU"
    --num-inducing "$NUM_INDUCING"
    --inducing-source "$INDUCING_SOURCE"
    --lipid-batch-size "$LIPID_BATCH_SIZE"
    --epochs "$EPOCHS"
    --learning-rate "$LEARNING_RATE"
    --batch-size "$BATCH_SIZE"
    --seed "$SEED"
    --device "$DEVICE"
)
[ "$NO_ARD" = "1" ] && common_args+=(--no-ard)
[ -n "$LIPIDS_FILE" ] && common_args+=(--lipids-file "$LIPIDS_FILE")
[ -n "$LIMIT" ] && common_args+=(--limit "$LIMIT")
if [ "$RENDER" = "1" ]; then
    common_args+=(--render --render-max-lipids "$RENDER_MAX_LIPIDS")
else
    common_args+=(--no-render)
fi
[ "$WANDB" = "1" ] && common_args+=(--wandb --wandb-project "$WANDB_PROJECT")

# Both arms encode the shared hyperparameters in their name, so a sweep over
# N_PARCELS / PARCEL_RANK / EPOCHS never overwrites an earlier comparison.
base_tag="${KERNEL_FAMILY}-nu${NU}-ind${NUM_INDUCING}-lr${LEARNING_RATE}-ep${EPOCHS}-lbs${LIPID_BATCH_SIZE}"
parcel_tag="${base_tag}-parcel$(basename "$PARCEL_FIELD" .npz)-r${PARCEL_RANK}-stride${STRIDE}"
[ "$PARCEL_SHARED_B" = "1" ] && parcel_tag="${parcel_tag}-sharedB"

run_baseline() {
    echo "== BASELINE (no parcel factor): $EXP_PREFIX-$base_tag"
    python "$SRC_PATH/maldi/lgp_experiment_per_lipid.py" \
        --exp-name "$EXP_PREFIX-$base_tag" "${common_args[@]}"
}

run_parcel() {
    echo "== PARCEL: $EXP_PREFIX-$parcel_tag"
    local extra=(--parcel-field "$PARCEL_FIELD"
                 --parcel-rank "$PARCEL_RANK"
                 --parcel-init-scale "$PARCEL_INIT_SCALE")
    [ "$PARCEL_SHARED_B" = "1" ] && extra+=(--parcel-shared-B)
    python "$SRC_PATH/maldi/lgp_experiment_per_lipid.py" \
        --exp-name "$EXP_PREFIX-$parcel_tag" "${common_args[@]}" "${extra[@]}"
}

case "$MODE" in
    baseline) run_baseline ;;
    parcel)   run_parcel ;;
    both)     run_baseline; run_parcel ;;
    *) echo "MODE must be both|parcel|baseline, got '$MODE'" >&2; exit 2 ;;
esac

echo
echo "Results under $OUTPUT_DIR/"
echo "  baseline : $EXP_PREFIX-$base_tag"
echo "  parcel   : $EXP_PREFIX-$parcel_tag"
echo "Compare the metrics.csv / summary.json in each. In W&B, parcel/offdiag_mean"
echo "is the mean covariance multiplier between different parcels: it starts near"
echo "1 and falls only if the model actually learned to stop smoothing at borders."
