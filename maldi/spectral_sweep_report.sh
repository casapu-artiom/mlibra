#!/usr/bin/env bash
# Aggregate + rank the per-config rows written by spectral_distance_sweep.sh.
# Reads OUT_DIR/rows/*.csv and writes OUT_DIR/{per_lipid,per_config,ranked}.csv.
#
# Run after the sweep jobs finish. Override anything by exporting it, e.g.
#     OBJECTIVE=spectral_gain RANK_STAT=decay_strength_quantile ./spectral_sweep_report.sh
# Extra flags pass straight through:  ./spectral_sweep_report.sh --top 30
set -eu

: "${OUTPUT_DIR:=/home/casap/mlibra/output}"
: "${SRC_PATH:=/home/casap/mlibra_git}"
: "${OUT_DIR:=${OUTPUT_DIR}/spectral_sweep}"

: "${OBJECTIVE:=spectral}"     # euclidean|geodesic|spectral|spectral_low<N>|spectral_gain|lowmode_gain
: "${RANK_STAT:=decay_strength_quantile}"
: "${TOP:=20}"

cd "$SRC_PATH"

python "$SRC_PATH/maldi/spectral_sweep_report.py" \
    --out-dir "$OUT_DIR" \
    --objective "$OBJECTIVE" \
    --rank-stat "$RANK_STAT" \
    --top "$TOP" \
    "$@"
