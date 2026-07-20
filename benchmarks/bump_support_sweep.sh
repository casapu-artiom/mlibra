#!/usr/bin/env bash
# ============================================================================
# bump_support_sweep.sh
# ---------------------------------------------------------------------------
# Sweep bump_scale x bump_decay across a set of GRAPH CONFIGURATIONS.
#
# Each graph config is ONE bump_support_report.py run: it calls the deployed
# visualize_laplacian.setup() once (builds/loads the graph), computes the
# MALDI->nearest-node distances once, then re-bins/re-weights them over the
# whole BUMP_SCALES x BUMP_DECAYS grid cheaply. Per-config output lands in
# OUT_DIR/<tag>/, and every summary.csv is concatenated into
# OUT_DIR/summary_all.csv (self-describing: it carries stride/threshold/
# knn_method/knn_k/inflation/bw/bump_scale/bump_decay columns).
#
# WHAT ACTUALLY MOVES THE ANSWER
#   Bump SUPPORT (within/beyond) depends only on the NODE SET = (stride,
#   threshold) and on alpha = bump_scale * bw. So:
#     * stride, threshold        -> change the node set  => real, EXPENSIVE axis
#     * bump_scale, bw           -> only scale alpha      => cheap
#     * bump_decay               -> only the in-support weight profile => cheap
#     * knn_method, knn_k, infl  -> change EDGES, not nodes => DO NOT change
#                                   coverage (swept only if you want them logged)
#   Hence the defaults sweep stride x threshold and keep knn/bw single. Add more
#   axes by exporting the arrays below if you want them recorded per config.
#
# Override anything via env, e.g.:
#   OUT_DIR=/tmp/bs STRIDES="4 8" THRESHOLDS="5 40 50" \
#   BUMP_SCALES="1 2 3 5 10 20" BUMP_DECAYS="0.01 0.05 0.1" ./bump_support_sweep.sh
# ============================================================================
set -u

: "${SRC_PATH:=/home/casap/mlibra_git}"
: "${DATA:=/home/casap/mlibra/mlibra_data}"
: "${OUT_DIR:=./bump_sweep}"
: "${DEVICE:=cuda}"

: "${REFERENCE_FILE:=${DATA}/reference_image.npy}"
: "${ANNOTATION_FILE:=${DATA}/level_15annot.npy}"
: "${MALDI_FILE:=${DATA}/maindata_minimal.parquet}"
: "${EIGENVECTOR_DIR:=/home/casap/mlibra/output/eigenvectors}"
: "${TEMPLATE_NAME:=reference}"
: "${LAPLACIAN_NORM:=randomwalk}"
: "${NUM_MODES:=300}"   # eigvecs are unused by the report; kept modest for setup()

# ---- expensive axis: node set ----------------------------------------------
: "${STRIDES:=4}"
: "${THRESHOLDS:=5}"
# ---- edge axes (do NOT change coverage; single by default) -----------------
: "${KNN_METHODS:=faiss faiss_atlas_weighted}"
: "${KNN_KS:=15}"
: "${INFLATION:=50.0}"
: "${GRAPHBANDWIDTHS:=0.1}"
# ---- cheap inner grid: one run re-bins the whole thing ----------------------
: "${BUMP_SCALES:=1}"
: "${BUMP_DECAYS:=0.01}"

# argparse requires a --bump-scale / --bump-decay anchor; use the first of each
# (it is deduped into the swept grid, so it adds no extra row).
BASE_SCALE=$(printf '%s\n' $BUMP_SCALES | head -1)
BASE_DECAY=$(printf '%s\n' $BUMP_DECAYS | head -1)

mkdir -p "$OUT_DIR"

for stride in $STRIDES; do
  for thresh in $THRESHOLDS; do
    for method in $KNN_METHODS; do
      for k in $KNN_KS; do
        for infl in $INFLATION; do
          for bw in $GRAPHBANDWIDTHS; do
            tag="m${method}_k${k}_s${stride}_t${thresh}_i${infl}_bw${bw}"
            echo "=================================================================="
            echo "=== $tag ==="
            python "$SRC_PATH/benchmarks/bump_support_report.py" \
                --template-name "$TEMPLATE_NAME" \
                --reference-file "$REFERENCE_FILE" \
                --annotations-file "$ANNOTATION_FILE" \
                --eigenvector-dir "$EIGENVECTOR_DIR" \
                --knn-method "$method" \
                --knn-k "$k" \
                --cross-region-inflation "$infl" \
                --num-modes "$NUM_MODES" \
                --laplacian-norm "$LAPLACIAN_NORM" \
                --graphbandwidth "$bw" \
                --threshold "$thresh" \
                --stride "$stride" \
                --bump-scale "$BASE_SCALE" --bump-decay "$BASE_DECAY" \
                --bump-scale-sweep $BUMP_SCALES \
                --bump-decay-sweep $BUMP_DECAYS \
                --maldi "$MALDI_FILE" \
                --device "$DEVICE" \
                --out-dir "$OUT_DIR/$tag" \
              || echo "  [skip] $tag failed (see above)"
          done
        done
      done
    done
  done
done

# ---- aggregate every per-config summary into one table ---------------------
python - "$OUT_DIR" <<'PY'
import sys, glob, os
import pandas as pd
root = sys.argv[1]
frames = []
for f in sorted(glob.glob(os.path.join(root, "*", "summary.csv"))):
    df = pd.read_csv(f)
    df.insert(0, "config_dir", os.path.basename(os.path.dirname(f)))
    frames.append(df)
if frames:
    out = os.path.join(root, "summary_all.csv")
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    print(f"\n[aggregate] {len(frames)} config summaries -> {out}")
else:
    print("\n[aggregate] no per-config summary.csv found (all runs failed?)")
PY
