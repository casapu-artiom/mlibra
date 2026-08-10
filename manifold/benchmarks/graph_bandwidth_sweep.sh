# Output dir: override with OUT_DIR=/path ./graph_bandwidth_sweep.sh
#             (default <repo>/manifold/benchmarks/output/bw_sweep).
: "${OUT_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/output/bw_sweep}"
# Graph build device. This diagnostic only analyzes edge weights (no eigensolve),
# so CPU is the intended path; it also avoids the GPU faiss builder's device-side
# assert at high k (which aborts the whole run uncatchably). Override DEVICE=cuda.
: "${DEVICE:=cuda}"

python manifold/benchmarks/graph_bandwidth_sweep.py \
    --template-name reference \
    --reference-file /home/casap/mlibra/mlibra_data/reference_image.npy \
    --annotations-file /home/casap/mlibra/mlibra_data/level_15annot.npy \
    --eigenvector-dir /home/casap/mlibra/output/eigenvectors \
    --knn-methods faiss faiss_atlas_weighted \
    --knn-ks 15 \
    --inflations 50.0 \
    --root-handling cross dissolve \
    --denoise-labels 0 3 3 \
    --prune-cross-region 0.0 0.95 0.97 \
    --thresholds 5 \
    --strides 4 \
    --device "$DEVICE" \
    --out-dir "$OUT_DIR"