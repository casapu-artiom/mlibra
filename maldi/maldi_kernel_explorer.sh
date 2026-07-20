#!/usr/bin/env bash
# Interactive MALDI <-> kernel correlation explorer.
#   - pick a mouse (Sample) + lipid
#   - scroll MALDI slices, rendered flat via per-section PCA projection
#   - click a test point -> riemann / matern / diffusion kernels highlight
#     which nodes are "taken into account"
#   - live scatter of kernel weight vs lipid intensity (Pearson/Spearman)
#   - anatomical borders overlaid when knn-method is not faiss
#
# Add --self-test to run the numeric pipeline headless (no napari).
#
# To use TRAINED kernels (a fold) instead of the hand-set nu/lengthscale, and
# add two held-out-error layers, append (launch with the SAME graph params the
# runs were trained on so the eigvecs match — here stride4/knn15/infl50/K2300):
#   --riemann-run   /home/casap/mlibra/output/per_lipid/FOLD-2-learnls-manifold-nu2-K2300-...-infl50-learndiff10 \
#   --euclidean-run /home/casap/mlibra/output/per_lipid/FOLD-2-euclidean-ard-matern-nu1.5-ind2000-5-lr0.05-ep10-lbs1010
# The lipid dropdown then drives the trained hypers per lipid; toggle
# "use trained hypers" in the dock to compare against the sliders.

python maldi/maldi_kernel_explorer.py \
    --template-name reference \
    --reference-file /home/casap/mlibra/mlibra_data/reference_image.npy \
    --annotations-file /home/casap/mlibra/mlibra_data/level_15annot.npy \
    --eigenvector-dir /home/casap/mlibra/output/eigenvectors \
    --maldi-file /home/casap/mlibra/mlibra_data/maindata_minimal.parquet \
    --stride 4 --threshold 5 \
    --knn-method faiss_atlas_weighted \
    --cluster-k 64 --cluster-spatial-weight 1.0 \
    --cross-region-inflation 50 \
    --knn-k 15 \
    --n-list sqrt --n-probe 8 \
    --laplacian-norm randomwalk \
    --graphbandwidth 0.02 \
    --num-modes 1000 \
    --ncv-min ${NCV_MIN:--1} \
    --nu 2.0 --lengthscale 0.1 \
    --matern-nu 1.5 \
    --diffusion-time 1.0 --diffusion-sigma 1.0 \
    --point-size 1.3 --gamma 0.6 \
    --root-handling dissolve \
    --denoise-labels 3 --prune-cross-region 0.97 \
    "$@"
    #--riemann-run /home/casap/mlibra/output/per_lipid/FOLD-2-learnls-manifold-nu2-K2300-stride4-learnls-bs1.0-bd0.01-bw0.1-knn15-faiss_atlas_weighted-5-randomwalk-ind2000-lr0.05-ep10-lbs10-blend0.8-infl50-learndiff10 \
    #--euclidean-run /home/casap/mlibra/output/per_lipid/FOLD-2-euclidean-ard-matern-nu1.5-ind2000-5-lr0.05-ep10-lbs1010 \
