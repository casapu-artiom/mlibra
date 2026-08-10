python manifold/viz/visualize_kernels.py \
    --template-name reference \
    --reference-file /home/casap/mlibra/mlibra_data/reference_image.npy \
    --annotations-file /home/casap/mlibra/mlibra_data/level_5annot.npy \
    --eigenvector-dir /home/casap/mlibra/output/eigenvectors \
    --knn-k 15 \
    --knn-method faiss_atlas_weighted \
    --cross-region-inflation 50 \
    --num-modes 1000 \
    --stride 4 \
    --ncv-min ${NCV_MIN:--1} \
    --laplacian-norm randomwalk \
    --graphbandwidth 0.1 \
    --n-sources 100 --n-targets 60 --k-show 30 \
    --source-marker-size 6 \
    --n-list sqrt --n-probe 8 \
    --fabric-edge-sample 2000000 --laplacian-edge-sample 800000 \
    --root-handling dissolve \
    --skip-eigvecs

#--treat_zero_as_cross False \
#--denoise-labels 3 --prune-cross-region 0.5 \

# Soft anatomical prior instead of hard per-region topology: keep the plain
# faiss graph but inflate cross-region edge distances (needs --annotations-file):
#
#   --knn-method faiss_atlas_weighted --cross-region-inflation 10
#
# (inflation 1.0 = off, 10 = mild, 100 = strong; sweeps 10/50/100 are common).
