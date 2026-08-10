#!/usr/bin/env bash
# Full Laplacian explorer (heat sources, knn fabric, edge layers).
# For the reference / L·f / L_N·f comparison with graph + mode dropdowns, use
# manifold/viz/visualize_laplacian_simple.sh instead — the arg set below (n-sources,
# fabric-*, k-show) belongs to visualize_laplacian.py only.

python manifold/viz/visualize_laplacian.py \
    --template-name reference \
    --reference-file /home/casap/mlibra/mlibra_data/reference_image.npy \
    --annotations-file /home/casap/mlibra/mlibra_data/level_15annot.npy \
    --eigenvector-dir /home/casap/mlibra/output/eigenvectors \
    --knn-method faiss_atlas_weighted \
    --knn-k 15 \
    --n-probe 8 \
    --cross-region-inflation 50.0 \
    --num-modes 300 \
    --ncv-min ${NCV_MIN:--1} \
    --laplacian-norm randomwalk \
    --graphbandwidth 0.1 \
    --n-sources 100 --n-targets 60 --k-show 30 \
    --source-marker-size 6 \
    --fabric-edge-sample 200000 --laplacian-edge-sample 80000 \
    --threshold 5 \
    --stride 4