#!/usr/bin/env sh
# Launcher for the weight-space (spectral) Riemann manifold experiment, the
# spectral twin of run_manifold.sh. Differences from run_manifold.sh: the
# spectral GP has NO inducing points and does not support per-task lengthscales
# or the product-ARD ScaleKernel composition, so all of those knobs are gone
# here. Everything graph/Laplacian/eigenpair-related is identical (same caches).
: "${NUM_MODES:=1300}"
# Lanczos Krylov subspace floor. -1 = auto (max(1500, 3*NUM_MODES+20)).
# At stride=1 the 1500 floor blows up GPU memory; set e.g. NCV_MIN=100.
: "${NCV_MIN:=-1}"
: "${LATENT_DIM:=5}"
: "${STRIDE:=4}"
: "${BATCH_SIZE:=1000}"
: "${N_EPOCHS:=30}"
: "${LEARNING_RATE:=0.001}"
: "${GRAPHBANDWIDTH:=0.1}"
# Diffusion scale (manifold kernel): multiplicative scale on the frozen Laplacian
# spectrum (lambda_k -> DIFFUSION_SCALE_INIT * lambda_k in the Matern spectral
# density). No eigenpair recompute. LEARN_DIFFUSION_SCALE=1 trains it; otherwise
# pinned at the init (1.0 = identity). Adds -learndiff to EXP_NAME when learned.
: "${DIFFUSION_SCALE_INIT:=1.0}"
: "${LEARN_DIFFUSION_SCALE:=0}"
: "${NU:=2}"
: "${KNN_K:=15}"
: "${BUMP_SCALE:=1.0}"
: "${BUMP_DECAY:=0.01}"
: "${LENGTHSCALE_INIT:=1.0}"
: "${SEED:=416465}"
: "${KERNEL:=symmetric}"
: "${MODE:=lgp}"
: "${TEMPLATE_NAME:=reference}"
: "${DATA_PATH:=/home/casap/mlibra/mlibra_data}"
: "${EIGENVECTOR_DIR:=/home/casap/mlibra/output/eigenvectors}"
: "${OUTPUT_DIR:=/home/casap/mlibra/output}"
: "${MALDI_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal.parquet}"
: "${REFERENCE_FILE:=/home/casap/mlibra/mlibra_data/reference_image.npy}"
: "${ANNOTATION_FILE:=/home/casap/mlibra/mlibra_data/level_15annot.npy}"
: "${SLICES_DATASET_FILE:=/home/casap/mlibra_git/maldi/data/splits/fold_2.json}"
: "${AVAILABLE_LIPIDS_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal_available_lipids.npy}"
: "${KNN_METHOD:=faiss_atlas_weighted}"
: "${CROSS_REGION_INFLATION:=10.0}"
: "${SRC_PATH:=/home/casap/mlibra_git}"
: "${EXP_PREFIX:=FOLD-2-STATIC-INDP}"
: "${LAPLACIAN_NORM:=randomwalk}"
: "${THRESHOLD:=5}"

# ---- FAISS CPU-only flags (env -> CLI) ------------------------------------
# The submit script passes FAISS_CPU_* as env vars; here we translate them into
# the Python --faiss-cpu-* flags. Python reads ONLY the CLI, never the env.
: "${FAISS_CPU_GRAPH:=0}"
: "${FAISS_CPU_SEARCH:=0}"
: "${FAISS_CPU_RECON:=0}"
FAISS_CPU_ARGS=""
[ "$FAISS_CPU_GRAPH" = "1" ] && FAISS_CPU_ARGS="$FAISS_CPU_ARGS --faiss-cpu-graph"
[ "$FAISS_CPU_SEARCH" = "1" ] && FAISS_CPU_ARGS="$FAISS_CPU_ARGS --faiss-cpu-search"
[ "$FAISS_CPU_RECON" = "1" ] && FAISS_CPU_ARGS="$FAISS_CPU_ARGS --faiss-cpu-recon"

# Force a fresh KNN-graph build (bypass the cache) -- needed to actually time
# graph construction; otherwise the cached graph is just reloaded.
: "${FORCE_RECOMPUTE_GRAPH:=0}"
[ "$FORCE_RECOMPUTE_GRAPH" = "1" ] && FAISS_CPU_ARGS="$FAISS_CPU_ARGS --force-recompute-graph"

# FAISS IVF sizing. Pass an int or 'sqrt' (nlist=sqrt(N), nprobe=sqrt(nlist)) --
# 'sqrt' is what makes the CPU path fast at scale (see faiss_bench_report).
: "${N_LIST:=sqrt}"
: "${N_PROBE:=8}"

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

# NOTE: every swept hyperparameter MUST appear here, or two configs collapse to the
# same output dir and clobber each other's checkpoints/test/args.npy. K<modes> is
# included because the submit sweep varies NUM_MODES at a fixed stride. SPECTRAL
# tag distinguishes these dirs from the inducing-point (MANIFOLD) runs.
EXP_NAME="$EXP_PREFIX-SPECTRAL-$LATENT_DIM-$STRIDE-K$NUM_MODES-$TEMPLATE_NAME-$THRESHOLD-$BATCH_SIZE-$KNN_METHOD-$CROSS_REGION_INFLATION-$KNN_K-$LAPLACIAN_NORM-$NU-$BUMP_SCALE-$BUMP_DECAY-$GRAPHBANDWIDTH-$LENGTHSCALE_INIT"

# Diffusion scale: always pass the init (1.0 = identity); learn it only when asked.
# Encode the INIT VALUE in the tag (not just a boolean) so a sweep over inits gets
# distinct dirs. Frozen-at-1.0 (the default) adds no suffix => unchanged dir names.
DIFF_ARGS="--diffusion-scale-init $DIFFUSION_SCALE_INIT"
if [ "$LEARN_DIFFUSION_SCALE" = "1" ]; then
    DIFF_ARGS="$DIFF_ARGS --learn-diffusion-scale"
    EXP_NAME="$EXP_NAME-learndiff$DIFFUSION_SCALE_INIT"
elif [ "$DIFFUSION_SCALE_INIT" != "1.0" ]; then
    EXP_NAME="$EXP_NAME-diff$DIFFUSION_SCALE_INIT"
fi

python $SRC_PATH/maldi/spectral_lgp_manifold_experiment.py \
    --exp-name $EXP_NAME \
    --dataset-path $DATA_PATH \
    --maldi-file $MALDI_FILE \
    --output-dir $OUTPUT_DIR \
    --template-name $TEMPLATE_NAME \
    --reference-file $REFERENCE_FILE \
    --threshold $THRESHOLD \
    --annotations-file $ANNOTATION_FILE \
    --eigenvector-dir $EIGENVECTOR_DIR \
    --batch-size $BATCH_SIZE \
    --epochs $N_EPOCHS \
    --learning-rate $LEARNING_RATE \
    --latent-dim $LATENT_DIM \
    --seed $SEED \
    --slices-dataset-file $SLICES_DATASET_FILE \
    --num-modes $NUM_MODES \
    --ncv-min $NCV_MIN \
    --kernel "$KERNEL" \
    --laplacian-norm "$LAPLACIAN_NORM" \
    --stride $STRIDE \
    --mode "$MODE" \
    --nu $NU \
    --bump-scale $BUMP_SCALE \
    --bump-decay $BUMP_DECAY \
    --graphbandwidth-init $GRAPHBANDWIDTH \
    --lengthscale-init $LENGTHSCALE_INIT \
    $DIFF_ARGS \
    --knn-method $KNN_METHOD \
    --cross-region-inflation $CROSS_REGION_INFLATION \
    --knn-k $KNN_K \
    --n-list "$N_LIST" \
    --n-probe "$N_PROBE" \
    --available-lipids-file $AVAILABLE_LIPIDS_FILE \
    --do-brain-reconstruction \
    $FAISS_CPU_ARGS "$@"
