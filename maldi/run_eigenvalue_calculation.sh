#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "================================================="
echo " Starting Pre-computation Pipeline for Manifold "
echo "================================================="

# Force Python to run in unbuffered mode for real-time Run:ai logs
export PYTHONUNBUFFERED=1

REFERENCE_VOLUME="/mydata/mlibra_data/reference_image.npy"
ANNOTATION_VOLUME="/mydata/mlibra_data/level_15annot.npy"
OUTPUT_PATH="/myhome/mlibra/output/eigenvectors"

# Eigensolver and Graph parameters
STRIDE=4
K_NEIGHBORS=15
NUM_MODES=1000
BANDWIDTH=1.0

# Execute the python runner
python maldi/compute_eigenvectors.py \
    --reference-volume $REFERENCE_VOLUME \
    --annotations-volume $ANNOTATION_VOLUME \
    --output-path $OUTPUT_PATH \
    --stride $STRIDE \
    --k $K_NEIGHBORS \
    --modes $NUM_MODES \
    --bandwidth $BANDWIDTH \
    --project "riemann-eigensolver"

echo "================================================="
echo " Pipeline Finished Successfully "
echo "================================================="