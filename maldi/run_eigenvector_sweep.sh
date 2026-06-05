#!/usr/bin/env sh
: "${OUTPUT_DIR:=/home/casap/mlibra/output}"
: "${REFERENCE_FILE:=/home/casap/mlibra/mlibra_data/reference_image.npy}"
: "${ANNOTATION_FILE:=/home/casap/mlibra/mlibra_data/level_15annot.npy}"
: "${EIGENVECTOR_DIR:=/home/casap/mlibra/output/eigenvectors}"
: "${SRC_PATH:=/home/casap/mlibra_git}"

cd $SRC_PATH
#pip install -e .

python $SRC_PATH/maldi/laplacian_psd_sweep.py \
    --template-name reference \
    --reference-file "$REFERENCE_FILE" \
    --annotations-file "$ANNOTATION_FILE" \
    --eigenvector-dir /home/casap/mlibra/output/eigenvectors \
    --stride 4 \
    --n-list 1 \
    --num-modes 1300 \
    --thresholds 5 20 40 150 \
    --knn-ks 15 60 120 180 \
    --graphbandwidths 0.05 0.1 0.5 1.0 \
    --cross-region-inflations 1 5 10 100 \
    --out $OUTPUT_DIR/psd_sweep.csv \
    --skip-on-error \
    --append \
    "$@"