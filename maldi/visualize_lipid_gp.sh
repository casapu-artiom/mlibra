#!/bin/bash
# Per-lipid spectral GP visualizer.
# Mirrors the relevant args from run_manifold.sh so the graph/eigendecomp
# match what your trained model used.

python maldi/visualize_lipid_gp.py \
    --template-name reference \
    --reference-file /home/casap/mlibra/mlibra_data/reference_image.npy \
    --annotations-file /home/casap/mlibra/mlibra_data/level_15annot.npy \
    --threshold 50 \
    --stride 8 \
    --eigenvector-dir /home/casap/mlibra/output/eigenvectors \
    --knn-method faiss --knn-k 120 \
    --num-modes 3000 \
    --laplacian-norm randomwalk \
    --graphbandwidth 1.0 \
    --nu 1 --lengthscale 1.0 \
    --bump-scale 20.0 --bump-decay 0.01 \
    --annotations-file /home/casap/mlibra/mlibra_data/level_15annot.npy \
    --maldi-file /home/casap/mlibra/mlibra_data/maindata_minimal.parquet \
    --slices-dataset-file /home/casap/mlibra_git/maldi/data/splits/difficult.json \
    --available-lipids-file /home/casap/mlibra/mlibra_data/maindata_minimal_available_lipids.npy \
    --initial-lipid-name "PA 36:1 PA 38:4" \
    --noise-sigma 0.3 \
    --render-stride 4 \
    --training-subsample 10000 \
    --eucl-subsample 3000