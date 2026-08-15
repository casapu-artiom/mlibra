#!/usr/bin/env sh
: "${NUM_INDUCING_POINTS:=1000}"
: "${INDUCING_SOURCE:=reference}"
: "${LATENT_DIM:=5}"
: "${BETA:=1.0}"        # KL weight; a float, or "elbo" for B/N
: "${BATCH_SIZE:=1000}"
: "${N_EPOCHS:=10}"
: "${LEARNING_RATE:=0.001}"
: "${SEED:=416465}"
: "${KERNEL:=matern}"
: "${MODE:=lgp}"
: "${NO_RSAMPLE:=true}"
: "${LEARN_INDUCING:=true}"   # true -> learn inducing-point locations (else fixed)
: "${ARD:=true}"              # true -> per-axis ARD lengthscales (else isotropic)
: "${DATA_PATH:=/home/casap/mlibra/mlibra_data}"
: "${OUTPUT_DIR:=/home/casap/mlibra/output/lgp}"
: "${MALDI_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal.parquet}"
: "${REFERENCE_FILE:=/home/casap/mlibra/mlibra_data/reference_image.npy}"
: "${ANNOTATION_FILE:=/home/casap/mlibra/mlibra_data/level_15annot.npy}"
: "${SLICES_DATASET_FILE:=/home/casap/mlibra_git/maldi/data/splits/fold_2.json}"
: "${AVAILABLE_LIPIDS_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal_available_lipids.npy}"
: "${SRC_PATH:=/home/casap/mlibra_git}"
: "${EXP_PREFIX:=FOLD-2}"

# Reconstruct only the voxels the composite render actually reads (slice planes +
# the 3D MIP's stride): ~5.5x fewer voxels, near-identical figure. Writes sparse
# volumes to volume_sparse/ instead of the dense volume/ that napari + the
# analysis scripts consume -- so set it to 0 if you need the full 3D volumes.
: "${RENDER_VOXELS_ONLY:=1}"
RENDER_ARGS=""
[ "$RENDER_VOXELS_ONLY" != "0" ] && RENDER_ARGS="--render-voxels-only"

if [ "$NO_RSAMPLE" = "true" ] || [ "$NO_RSAMPLE" = "1" ]; then
    SAMPLING_TAG="MEAN"         # Update this to whatever you want the name to be without rsample
    SAMPLING_FLAG="--no-rsample" # The flag to pass to your python script
else
    SAMPLING_TAG="RSAMPLE"
    SAMPLING_FLAG=""             # Leave empty so Python falls back to its default
fi

if [ "$LEARN_INDUCING" = "true" ] || [ "$LEARN_INDUCING" = "1" ]; then
    LEARN_INDUCING_FLAG="--learn-inducing"; LI_TAG="learnind"
else
    LEARN_INDUCING_FLAG="";               LI_TAG="fixind"
fi

if [ "$ARD" = "true" ] || [ "$ARD" = "1" ]; then
    ARD_FLAG="--ard"; ARD_TAG="ard"
else
    ARD_FLAG="";      ARD_TAG="iso"
fi

EXP_NAME="$EXP_PREFIX-LGPALL-$SAMPLING_TAG-$LATENT_DIM-$INDUCING_SOURCE-$NUM_INDUCING_POINTS-$BATCH_SIZE-$LI_TAG-$ARD_TAG"
# Only tag when beta departs from the historical 1.0, so existing run dirs
# (and every path in report_all) keep their names.
[ "$BETA" != "1.0" ] && EXP_NAME="$EXP_NAME-beta$BETA"

cd $SRC_PATH
#pip install -e .

# ---- reconstruction lipids from the curated subset file -------------------
# Reconstruct/render exactly the lipids listed in RECONSTRUCTION_LIPIDS_FILE.
# A tiny python parser (clearer than a shell loop) emits one lipid per line;
# with IFS=newline we then fold each line into a positional arg -- preserving
# the spaces in names -- so --reconstruction-lipids (nargs='+') picks them up.
: "${RECONSTRUCTION_LIPIDS_FILE:=/home/casap/mlibra_git/maldi/data/lipid_subset.txt}"
# Built-in fallback (mirror of lipid_subset.txt) used when the file above is
# missing/empty -- e.g. a container where the repo path differs -- so we still
# render the intended subset instead of every lipid.
RECON_LIPIDS_DEFAULT='PC 35:1 PE 38:1
PA 36:1
LPC 22:6
PE O-36:0 PE O-38:3
Hex2Cer 40:1;O2'
RECON_LIPIDS=$(python - "$RECONSTRUCTION_LIPIDS_FILE" <<'PY'
import sys
try:
    with open(sys.argv[1]) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                print(line)
except FileNotFoundError:
    pass
PY
)
if [ -n "$RECON_LIPIDS" ]; then
    echo "Reconstruction lipids from $RECONSTRUCTION_LIPIDS_FILE"
else
    echo "NOTE: $RECONSTRUCTION_LIPIDS_FILE missing/empty;" \
         "using built-in default lipid subset."
    RECON_LIPIDS="$RECON_LIPIDS_DEFAULT"
fi
IFS='
'
set -- "$@" --reconstruction-lipids $RECON_LIPIDS
unset IFS

python $SRC_PATH/maldi/lgp_experiment.py \
    --exp-name $EXP_NAME \
    --dataset-path $DATA_PATH \
    --maldi-file $MALDI_FILE \
    --template-name "reference" \
    --reference-file $REFERENCE_FILE \
    --annotations-file $ANNOTATION_FILE \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $N_EPOCHS \
    --learning-rate $LEARNING_RATE \
    --latent-dim $LATENT_DIM \
    --seed $SEED \
    --slices-dataset-file $SLICES_DATASET_FILE \
    --num-inducing $NUM_INDUCING_POINTS \
    --beta "$BETA" \
    --inducing-source "$INDUCING_SOURCE" \
    --kernel "$KERNEL" \
    --mode "$MODE" \
    --available-lipids-file $AVAILABLE_LIPIDS_FILE \
    --do-brain-reconstruction \
    $RENDER_ARGS \
    $LEARN_INDUCING_FLAG $ARD_FLAG \
    $SAMPLING_FLAG "$@"
