#!/usr/bin/env bash
# Measure how cleanly the MALDI lipid data covariance decays along euclidean /
# geodesic / spectral / spectral_low distance, per lipid, for the graph config(s)
# given by the env vars below. Writes OUT_DIR/rows/<slug>.csv per config; it does
# NOT aggregate or rank — that is spectral_sweep_report.{py,sh}.
#
# With singleton axes this runs ONE config (the batch submitter
# submit/run_spectral_distance_sweep.sh fans the grid out across one job per
# config, all writing into a shared OUT_DIR/rows/). Multi-value axes still run a
# local grid in-process.
#
# Every value has a default; override any by exporting it before calling, e.g.
#     KNN_KS="60 120" BANDWIDTHS="0.05 0.1 0.2" ./spectral_distance_sweep.sh
# Extra flags pass straight through:  ./spectral_distance_sweep.sh --force
#
# Resumable: each config writes OUT_DIR/rows/<slug>.csv and is skipped on a
# re-run, so a preempted job picks up where it stopped.

set -eu

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
: "${OUTPUT_DIR:=/home/casap/mlibra/output}"
: "${REFERENCE_FILE:=/home/casap/mlibra/mlibra_data/reference_image.npy}"
: "${ANNOTATION_FILE:=/home/casap/mlibra/mlibra_data/level_15annot.npy}"
: "${EIGENVECTOR_DIR:=/home/casap/mlibra/output/eigenvectors}"
: "${SRC_PATH:=/home/casap/mlibra_git}"
: "${TEMPLATE_NAME:=reference}"
: "${MALDI_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal.parquet}"
: "${AVAILABLE_LIPIDS_FILE:=/home/casap/mlibra/mlibra_data/maindata_minimal_available_lipids.npy}"
: "${LIPIDS_FILE:=/home/casap/mlibra_git/maldi/data/lipid_subset.txt}"
: "${OUT_DIR:=${OUTPUT_DIR}/spectral_sweep}"

# No fold: use ALL measured points, dropping whole samples (disease/pregnancy).
# Set EXCLUDE_SAMPLES="" to keep everything.
: "${EXCLUDE_SAMPLES:=GBA1 Pregnant1 Pregnant2 Pregnant4}"
: "${SAMPLE_COLUMN:=Sample}"

# -------------------------------------------------------------------------
# Sweep grid (space-separated lists)
# -------------------------------------------------------------------------
: "${KNN_METHODS:=faiss_atlas_weighted}"
: "${NORMALIZATIONS:=randomwalk}"
: "${THRESHOLDS:=5 40 50}"
: "${KNN_KS:=15 60}"
: "${BANDWIDTHS:=0.1 0.2}"
: "${INFLATIONS:=10 50}"
: "${NUS:=2}"
: "${LENGTHSCALES:=1.0 2.0}"
# STRIDE:MAX_MODES pairs — resolution coupled to its mode budget. A fine stride
# with a large mode budget (e.g. 4:6000) needs a huge Lanczos basis and can OOM;
# either shrink the budget (4:3000) or cap NCV_MIN below.
: "${STRIDE_MODES:=4:1300 8:6000}"
# Krylov subspace floor for the eigensolve. -1 = auto = max(1500, 3*modes+20),
# which at fine strides + many modes is what OOMs. Cap it to e.g. modes+a few
# hundred (must exceed the largest num_modes) to cut eigensolve memory.
: "${NCV_MIN:=-1}"

# Feature mode: 'snap' (graph nodes, bump-free) or 'nystrom' (real MALDI points at
# true coords via the deployed bump-weighted OOS map). BUMP_SCALES/BUMP_DECAYS are
# swept only in nystrom mode (inert in snap).
: "${FEATURE_MODE:=nystrom}"
: "${BUMP_SCALES:=1.0 2.0}"
: "${BUMP_DECAYS:=0.01}"

# -------------------------------------------------------------------------
# Low-mode spectral variant + ranking
# -------------------------------------------------------------------------
# LOW_MODES: space-separated absolute mode counts -> one spectral_low<N> metric
# each (e.g. "64 256 1000"). Empty => single spectral_low at LOW_MODE_FRAC.
# (Ranking lives in spectral_sweep_report.py, so OBJECTIVE/RANK_STAT are no longer
#  passed here — they belong to spectral_sweep_report.sh.)
: "${LOW_MODES:=20 50 100 200}"
: "${LOW_MODE_FRAC:=0.1}"      # used only when LOW_MODES is empty
: "${VARIO_ANCHORS:=1000}"
: "${VARIO_MAX_TARGETS:=4000}"
: "${N_BINS:=24}"
: "${DEVICE:=cuda}"

cd "$SRC_PATH"

python "$SRC_PATH/maldi/spectral_distance_sweep.py" \
    --template-name "$TEMPLATE_NAME" \
    --reference-file "$REFERENCE_FILE" \
    --annotations-file "$ANNOTATION_FILE" \
    --eigenvector-dir "$EIGENVECTOR_DIR" \
    --maldi-file "$MALDI_FILE" \
    --available-lipids-file "$AVAILABLE_LIPIDS_FILE" \
    --lipids-file "$LIPIDS_FILE" \
    --exclude-samples $EXCLUDE_SAMPLES \
    --sample-column "$SAMPLE_COLUMN" \
    --knn-methods $KNN_METHODS \
    --normalizations $NORMALIZATIONS \
    --thresholds $THRESHOLDS \
    --knn-ks $KNN_KS \
    --bandwidths $BANDWIDTHS \
    --inflations $INFLATIONS \
    --nus $NUS \
    --lengthscales $LENGTHSCALES \
    --stride-modes $STRIDE_MODES \
    --ncv-min "$NCV_MIN" \
    --feature-mode "$FEATURE_MODE" \
    --bump-scales $BUMP_SCALES \
    --bump-decays $BUMP_DECAYS \
    ${LOW_MODES:+--low-modes $LOW_MODES} \
    --low-mode-frac "$LOW_MODE_FRAC" \
    --vario-anchors "$VARIO_ANCHORS" \
    --vario-max-targets "$VARIO_MAX_TARGETS" \
    --n-bins "$N_BINS" \
    --device "$DEVICE" \
    --out-dir "$OUT_DIR" \
    "$@"
