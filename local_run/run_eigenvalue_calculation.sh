#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "================================================="
echo " Starting Pre-computation Pipeline for Manifold "
echo "================================================="

# Force Python to run in unbuffered mode for real-time Run:ai logs
export PYTHONUNBUFFERED=1

REFERENCE_VOLUME="/home/casap/mlibra/mlibra_data/reference_image.npy"
ANNOTATION_VOLUME="/home/casap/mlibra/mlibra_data/level_15annot.npy"
OUTPUT_PATH="/home/casap/mlibra/output/eigenvectors"

# Eigensolver and Graph parameters
STRIDE=4
K_NEIGHBORS=15
NUM_MODES=1000
BANDWIDTH=1.0
# Lanczos Krylov subspace floor. -1 = auto (max(1500, 3*NUM_MODES+20)).
# At STRIDE=1 the 1500 floor blows up GPU memory; set e.g. NCV_MIN=100.
NCV_MIN=-1

# Execute the python runner
python manifold/compute_eigenvectors.py \
    --reference-volume $REFERENCE_VOLUME \
    --annotations-volume $ANNOTATION_VOLUME \
    --output-path $OUTPUT_PATH \
    --stride $STRIDE \
    --k $K_NEIGHBORS \
    --modes $NUM_MODES \
    --ncv-min $NCV_MIN \
    --bandwidth $BANDWIDTH \
    --project "riemann-eigensolver"

echo "================================================="
echo " Pipeline Finished Successfully "
echo "================================================="