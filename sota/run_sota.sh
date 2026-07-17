#!/usr/bin/env sh
# SOTA 3D-reconstruction papers on the MALDI dataset, with the SAME
# reconstruction + render + diagnostics parity as run_manifold.sh / run_baseline.sh.
# MODEL selects the method:
#   ntf         -- Neural Transcriptomic Field: multiresolution hash-grid INR
#                  (bioRxiv 2026.05.28.726140). Heteroscedastic loss + TV
#                  smoothness + per-section bias.
#   spa3d       -- Spatial-pattern-enhanced GCN (Briefings in Bioinf. bbag060).
#                  SPE (Hilbert / ALFT) denoise + z-aware 3D GCN. Needs PyG.
#   deepspatial -- FAITHFUL DeepSpatial (bioRxiv 2026.04.28.721395): official
#                  GiT flow-matching + UOT + probability-flow ODE, within-specimen
#                  slice interpolation (its own DS_* knobs below). Not a harness
#                  regression model like ntf/spa3d -- it runs a separate driver
#                  (sota/deepspatial_transport/run_deepspatial_transport.py), but
#                  is launched from THIS single script.
#   gplfr       -- GP Latent Factor Regression (arXiv:2606.06576): latent GP +
#                  analytically-marginalized linear decoder. Base latent GP is
#                  BASE_GP={euclidean|riemann|spectral}; runs the MaldiExperiment
#                  harness (its own GPLFR_* / manifold knobs below).
: "${MODEL:=spa3d}"
: "${SEED:=416465}"
# Training defaults are model-specific (transport wants tiny batch / low LR / more
# epochs; GPLFR's collapsed linear decoder trains fast).
if [ "$MODEL" = "deepspatial" ]; then
    : "${N_EPOCHS:=100}" ; : "${BATCH_SIZE:=256}" ; : "${LEARNING_RATE:=0.0002}"
elif [ "$MODEL" = "gplfr" ]; then
    : "${N_EPOCHS:=2}" ; : "${BATCH_SIZE:=2000}" ; : "${LEARNING_RATE:=0.005}"
else
    : "${N_EPOCHS:=10}" ; : "${BATCH_SIZE:=4096}" ; : "${LEARNING_RATE:=0.001}"
fi
# ---- I/O (all default to LOCAL paths; the submit scripts just override these
#      env vars to point at the S3-mounted dirs) -----------------------------
: "${DATA_PATH:=/home/casap/mlibra/mlibra_data}"
: "${OUTPUT_DIR:=/home/casap/mlibra/output/sota}"
: "${MALDI_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal.parquet}"
: "${REFERENCE_FILE:=/home/casap/mlibra/mlibra_data/reference_image.npy}"
: "${ANNOTATION_FILE:=/home/casap/mlibra/mlibra_data/level_15annot.npy}"
: "${SLICES_DATASET_FILE:=/home/casap/mlibra_git/maldi/data/splits/fold_2.json}"
: "${AVAILABLE_LIPIDS_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal_available_lipids.npy}"
: "${RECONSTRUCTION_LIPIDS_FILE:=/home/casap/mlibra_git/maldi/data/lipid_subset.txt}"
: "${TEMPLATE_NAME:=reference}"
: "${SRC_PATH:=/home/casap/mlibra_git}"
: "${EXP_PREFIX:=FOLD-2}"
# whole_brain reconstruction => renders + per-lipid true-vs-pred scatterplots +
# value-distribution diagnostics (parity with run_manifold / run_baseline).
: "${RECONSTRUCT:=whole_brain}"
# Early stopping: best-checkpoint restore of the best epoch (never ship the last,
# which for a high-capacity field on the cross-mouse folds is usually the worst).
# EARLY_STOP_MONITOR=val (default) carves a val set from TRAIN (no leak) but only
# catches ordinary over-training; set =test to pick the best held-out-MOUSE epoch
# -- the fix for the rising-test-MSE case (selects the epoch on the test set).
: "${VAL_FRAC:=0.05}"
: "${EARLY_STOP_PATIENCE:=5}"
: "${EARLY_STOP_MONITOR:=val}"
# W&B: set WANDB=1 to enable (WANDB_PROJECT overrides the project name).
: "${WANDB:=1}"
: "${WANDB_PROJECT:=sota_maldi}"
WANDB_ARGS=""
[ "$WANDB" = "1" ] && WANDB_ARGS="--wandb --wandb-project $WANDB_PROJECT"

# --- NTF knobs (MODEL=ntf) --------------------------------------------------
: "${NTF_LEVELS:=12}"
: "${NTF_FEATURES:=8}"
: "${NTF_LOG2_HASHMAP:=19}"
: "${NTF_BASE_RES:=16}"
: "${NTF_MAX_RES:=1024}"
: "${NTF_HIDDEN:=256 256 128}"
: "${NTF_TV_WEIGHT:=0.05}"
: "${NTF_TV_EPS:=0.001}"
# L2 on the net incl. hash embeddings -- the strongest lever against the
# cross-mouse val/test gap (per-voxel memorization). Try 1e-4 .. 1e-2.
: "${NTF_WEIGHT_DECAY:=0.0001}"
# ported from the official NTF models.py: latent-z variance, slice-conditioned
# bias net on the low-freq hash levels, per-slice variance, and PSF averaging.
: "${NTF_FEATURES_Z:=16}"
: "${NTF_FEATURES_SLICE:=8}"
: "${NTF_LEVELS_BIAS:=4}"
: "${NTF_AUX_HIDDEN:=64}"
: "${NTF_BIAS_WEIGHT:=0.01}"
: "${NTF_PSF_SAMPLES:=4}"
: "${NTF_PSF_SIGMA:=0.01}"

# --- Spa3D knobs (MODEL=spa3d) ----------------------------------------------
: "${SPA3D_SPE:=alft}"          # none | hilbert | alft
: "${SPA3D_GRID:=128}"
: "${SPA3D_SECTIONS:=64}"
: "${SPA3D_ALFT_KEEP:=0.5}"
: "${SPA3D_Z_WEIGHT:=0.3}"
: "${SPA3D_KNN_K:=15}"
: "${SPA3D_GRAPH_NODES:=80000}"   # nodes in the single global Gaussian graph
: "${SPA3D_LENGTH_SCALE:=0.0}"    # Gaussian bandwidth l; 0 = median heuristic
: "${SPA3D_INTERP_K:=8}"          # nearest graph nodes blended at read-out
: "${SPA3D_HIDDEN:=512 512 256}"
: "${SPA3D_DROPOUT:=0.1}"

# --- DeepSpatial (faithful transport) knobs (MODEL=deepspatial) --------------
: "${DS_HIDDEN_SIZE:=256}"
: "${DS_DEPTH:=6}"
: "${DS_HEADS:=8}"
: "${DS_PATCH:=8}"
# Loss weights. Upstream is g=0.1,c=10 (100:1 toward CELL-TYPE, the paper's
# product). MALDI's product is the LIPIDS (region is only aux conditioning), so
# we FLIP it -- else the model learns per-lipid means but no spatial structure.
: "${DS_LAMBDA_G:=1.0}"
: "${DS_LAMBDA_C:=0.1}"
: "${DS_STEPS:=50}"
# Inter-plane spacing in xccf (mm) units; smaller => denser reconstruction.
: "${DS_THICKNESS:=0.02}"
: "${DS_ALPHA_SPATIAL:=0.3}"
: "${DS_UOT_REG:=0.3}"
: "${DS_UOT_TAU:=0.05}"
: "${DS_MAX_CELLS:=8000}"
# In-plane reconstruction density: every synthesized cell is a transported SOURCE
# cell, so coverage ~ this many points per section (measured sections are 60-90k
# voxels). 6000 => sparse/holey renders; raise toward the section size for a dense
# fill (0 = use ALL voxels). Total cells ~ this / DS_THICKNESS => memory/time.
: "${DS_MAX_CELLS_RECON:=0}"
# Source voxels per reconstruct() CALL. Each gap is chunked into batches of this
# size so one call never allocates batch*(gap/thickness)*n_lipids on the GPU --
# this is what lets DS_MAX_CELLS_RECON=0 (dense) run without OOM. Lower if you OOM.
: "${DS_RECON_BATCH:=8000}"
: "${DS_N_SAMPLES:=100000}"
# Section-pairing mode for training UOT trajectories:
#   within-mouse (default, faithful) = adjacent sections of the SAME mouse.
#   cross-mouse = pool all mice's sections into one AP-ordered stack and pair
#   adjacent across animals (denser sampling + cross-mouse fill; needs mice to
#   register well to the common CCF frame).
: "${DS_PAIRING:=cross-mouse}"
# Resume: if a checkpoint exists in the exp dir it is loaded and training is
# skipped. Set DS_FORCE_RETRAIN=1 to retrain from scratch instead.
: "${DS_FORCE_RETRAIN:=0}"
DS_FORCE_ARG=""
[ "$DS_FORCE_RETRAIN" = "1" ] && DS_FORCE_ARG="--force-retrain"

# --- GPLFR knobs (MODEL=gplfr) ----------------------------------------------
: "${BASE_GP:=riemann}"        # euclidean | riemann | spectral
: "${LATENT_DIM:=8}"
: "${NUM_INDUCING:=256}"
: "${INVERSE_TEMPERATURE:=0.1}"
: "${GPLFR_KERNEL:=matern}"
: "${GPLFR_NU:=2.5}"
# Manifold-base (riemann|spectral) knobs; ignored when BASE_GP=euclidean. The
# manifold bases need the eigenpair pipeline (EIGENVECTOR_DIR + graph/spectrum).
: "${EIGENVECTOR_DIR:=/home/casap/mlibra/output/eigenvectors}"
: "${NUM_MODES:=300}"
: "${STRIDE:=4}"
: "${THRESHOLD:=5}"
: "${KNN_K:=15}"
: "${LAPLACIAN_NORM:=randomwalk}"
: "${KNN_METHOD:=faiss_atlas_weighted}"
: "${CROSS_REGION_INFLATION:=10.0}"
: "${N_LIST:=sqrt}"
: "${N_PROBE:=8}"
: "${GRAPHBANDWIDTH_INIT:=0.1}"
: "${BUMP_SCALE:=1.0}"
: "${BUMP_DECAY:=0.01}"

cd $SRC_PATH
#pip install -e .

# Encode MODEL so a sweep over methods gets distinct output dirs.
MODEL_TAG=$(echo "$MODEL" | tr '[:lower:]' '[:upper:]')
if [ "$MODEL" = "deepspatial" ]; then
    DS_PAIRING_TAG=$(echo "$DS_PAIRING" | tr '[:lower:]' '[:upper:]')
    EXP_NAME="$EXP_PREFIX-DEEPSPATIAL-TRANSPORT-$DS_HIDDEN_SIZE-$DS_DEPTH-$N_EPOCHS-$DS_PAIRING_TAG"
elif [ "$MODEL" = "gplfr" ]; then
    EXP_NAME="$EXP_PREFIX-GPLFR-$BASE_GP-d$LATENT_DIM"
else
    EXP_NAME="$EXP_PREFIX-SOTA-$MODEL_TAG-$BATCH_SIZE"
fi

# ---- reconstruction lipids from the curated subset file (mirror run_manifold) ----
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
    echo "NOTE: $RECONSTRUCTION_LIPIDS_FILE missing/empty; using built-in default subset."
    RECON_LIPIDS="$RECON_LIPIDS_DEFAULT"
fi

IFS='
'
set -- --reconstruction-lipids $RECON_LIPIDS "$@"
unset IFS

if [ "$MODEL" = "deepspatial" ]; then
    # Faithful DeepSpatial: separate driver (within-specimen slice interpolation,
    # per-test-mouse volume reconstruction). Needs the deepspatial deps.
    python $SRC_PATH/sota/deepspatial_transport/run_deepspatial_transport.py \
        --mode lgp \
        --exp-name "$EXP_NAME" \
        --dataset-path $DATA_PATH \
        --maldi-file $MALDI_FILE \
        --output-dir $OUTPUT_DIR \
        --template-name "$TEMPLATE_NAME" \
        --reference-file $REFERENCE_FILE \
        --annotations-file $ANNOTATION_FILE \
        --slices-dataset-file $SLICES_DATASET_FILE \
        --available-lipids-file $AVAILABLE_LIPIDS_FILE \
        --seed $SEED \
        --epochs $N_EPOCHS \
        --batch-size $BATCH_SIZE \
        --learning-rate $LEARNING_RATE \
        --ds-hidden-size $DS_HIDDEN_SIZE \
        --ds-depth $DS_DEPTH \
        --ds-heads $DS_HEADS \
        --ds-patch $DS_PATCH \
        --ds-lambda-g $DS_LAMBDA_G \
        --ds-lambda-c $DS_LAMBDA_C \
        --ds-steps $DS_STEPS \
        --ds-thickness $DS_THICKNESS \
        --ds-alpha-spatial $DS_ALPHA_SPATIAL \
        --ds-uot-reg $DS_UOT_REG \
        --ds-uot-tau $DS_UOT_TAU \
        --ds-max-cells $DS_MAX_CELLS \
        --ds-max-cells-recon $DS_MAX_CELLS_RECON \
        --ds-recon-batch $DS_RECON_BATCH \
        --ds-n-samples $DS_N_SAMPLES \
        --ds-pairing $DS_PAIRING \
        $DS_FORCE_ARG \
        --reconstruct "$RECONSTRUCT" \
        $WANDB_ARGS \
        "$@"
elif [ "$MODEL" = "gplfr" ]; then
    # GPLFR: latent GP + collapsed linear decoder, on the MaldiExperiment harness.
    # Manifold flags are only forwarded for the manifold bases (euclidean ignores
    # them); they need the eigenpair pipeline (EIGENVECTOR_DIR + graph knobs).
    MANIFOLD_ARGS=""
    if [ "$BASE_GP" != "euclidean" ]; then
        MANIFOLD_ARGS="--eigenvector-dir $EIGENVECTOR_DIR \
            --num-modes $NUM_MODES --stride $STRIDE --threshold $THRESHOLD \
            --knn-k $KNN_K --laplacian-norm $LAPLACIAN_NORM --knn-method $KNN_METHOD \
            --annotations-file $ANNOTATION_FILE \
            --cross-region-inflation $CROSS_REGION_INFLATION \
            --n-list $N_LIST --n-probe $N_PROBE \
            --graphbandwidth-init $GRAPHBANDWIDTH_INIT \
            --bump-scale $BUMP_SCALE --bump-decay $BUMP_DECAY"
    fi
    python $SRC_PATH/sota/gplfr_experiment.py \
        --exp-name "$EXP_NAME" \
        --dataset-path $DATA_PATH \
        --maldi-file $MALDI_FILE \
        --output-dir $OUTPUT_DIR \
        --template-name "$TEMPLATE_NAME" \
        --reference-file $REFERENCE_FILE \
        --batch-size $BATCH_SIZE \
        --epochs $N_EPOCHS \
        --learning-rate $LEARNING_RATE \
        --latent-dim $LATENT_DIM \
        --seed $SEED \
        --slices-dataset-file $SLICES_DATASET_FILE \
        --num-inducing $NUM_INDUCING \
        --kernel "$GPLFR_KERNEL" \
        --nu $GPLFR_NU \
        --inverse-temperature $INVERSE_TEMPERATURE \
        --base-gp "$BASE_GP" \
        $MANIFOLD_ARGS \
        --mode "gplfr" \
        --available-lipids-file $AVAILABLE_LIPIDS_FILE \
        --do-brain-reconstruction \
        "$@"
else
    # NTF / Spa3D: coordinate regressors on the shared baselines harness.
    python $SRC_PATH/sota/run_sota.py \
        --exp-name "$EXP_NAME" \
        --dataset-path $DATA_PATH \
        --maldi-file $MALDI_FILE \
        --output-dir $OUTPUT_DIR \
        --template-name "$TEMPLATE_NAME" \
        --reference-file $REFERENCE_FILE \
        --annotations-file $ANNOTATION_FILE \
        --batch-size $BATCH_SIZE \
        --epochs $N_EPOCHS \
        --learning-rate $LEARNING_RATE \
        --latent-dim 5 \
        --seed $SEED \
        --slices-dataset-file $SLICES_DATASET_FILE \
        --num-inducing 200 \
        --kernel "matern" \
        --mode "lgp" \
        --available-lipids-file $AVAILABLE_LIPIDS_FILE \
        --model "$MODEL" \
        --ntf-levels $NTF_LEVELS \
        --ntf-features $NTF_FEATURES \
        --ntf-log2-hashmap $NTF_LOG2_HASHMAP \
        --ntf-base-res $NTF_BASE_RES \
        --ntf-max-res $NTF_MAX_RES \
        --ntf-hidden $NTF_HIDDEN \
        --ntf-tv-weight $NTF_TV_WEIGHT \
        --ntf-tv-eps $NTF_TV_EPS \
        --ntf-weight-decay $NTF_WEIGHT_DECAY \
        --ntf-features-z $NTF_FEATURES_Z \
        --ntf-features-slice $NTF_FEATURES_SLICE \
        --ntf-levels-bias $NTF_LEVELS_BIAS \
        --ntf-aux-hidden $NTF_AUX_HIDDEN \
        --ntf-bias-weight $NTF_BIAS_WEIGHT \
        --ntf-psf-samples $NTF_PSF_SAMPLES \
        --ntf-psf-sigma $NTF_PSF_SIGMA \
        --spa3d-spe $SPA3D_SPE \
        --spa3d-grid $SPA3D_GRID \
        --spa3d-sections $SPA3D_SECTIONS \
        --spa3d-alft-keep $SPA3D_ALFT_KEEP \
        --spa3d-z-weight $SPA3D_Z_WEIGHT \
        --spa3d-knn-k $SPA3D_KNN_K \
        --spa3d-graph-nodes $SPA3D_GRAPH_NODES \
        --spa3d-length-scale $SPA3D_LENGTH_SCALE \
        --spa3d-interp-k $SPA3D_INTERP_K \
        --spa3d-hidden $SPA3D_HIDDEN \
        --spa3d-dropout $SPA3D_DROPOUT \
        --reconstruct "$RECONSTRUCT" \
        --val-frac $VAL_FRAC \
        --early-stop-patience $EARLY_STOP_PATIENCE \
        --early-stop-monitor $EARLY_STOP_MONITOR \
        $WANDB_ARGS \
        "$@"
fi
