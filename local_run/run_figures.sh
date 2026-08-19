#!/usr/bin/env sh
# Regenerate the paper figures under reports/figures/. CPU only, no GPU, no
# cluster -- but the inputs can live on an S3 mount, which is why every path is
# an env var.
#
# Two figures, two scripts, because they need different things:
#
#   combined_panels.py     comparison grids: one row per model, so it needs a
#                          run dir per model. Point it at the staged winners
#                          tree (FIG_ROOT) or let it derive the winners from a
#                          report CSV (FROM_REPORT) and read them off S3.
#   lipid_volume_panel.py  full-page volume render of ONE run, so all it needs
#                          is that run's dense volume/ dir (VOLUME_DIR).
#
# Usage:
#   ./local_run/run_figures.sh                 # both, from local defaults
#   WHICH=combined ./local_run/run_figures.sh  # just the comparison grids
#   FIG_GROUPS="gps figure" ./local_run/run_figures.sh
#   WHICH=volume   ./local_run/run_figures.sh  # just the full-page render
#   DRY_RUN=1 ./local_run/run_figures.sh       # print the commands only
#
# On the cluster:
#   FROM_REPORT=reports/output/report_all/per_run.csv \
#   VOLUME_DIR=/s3/mlibra/mlibra-data/artiom/lgb_experiment_cv/FOLD-2-LGPALL-MEAN-5-data-2000-1000-learnind-ard10/volume \
#   OUT_DIR=/myhome/mlibra/figures ./local_run/run_figures.sh
: "${SRC_PATH:=/home/casap/mlibra_git}"
: "${WHICH:=all}"                 # all | combined | volume
: "${OUT_DIR:=$SRC_PATH/reports/figures}"
: "${DRY_RUN:=0}"

# --- combined_panels.py -----------------------------------------------------
# FROM_REPORT wins when set: run dirs become $S3_ROOT/<source>/FOLD-$FOLD-<config>,
# with the winner per method derived from the CSV. Otherwise FIG_ROOT is used as
# a staged tree with one subdirectory per model.
: "${FIG_ROOT:=/home/casap/mlibra/output/winners}"
: "${FROM_REPORT:=}"
: "${S3_ROOT:=/s3/mlibra/mlibra-data/artiom}"
: "${FOLD:=2}"
: "${FIG_GROUPS:=}"               # empty = the three defaults; e.g. "gps figure"
# NOT named GROUPS: that is a bash built-in array (the caller's group ids), so
# `GROUPS="gps" ./run_figures.sh` is silently dropped and never reaches here.
# The shared colour scale needs every model, so a cold run reads them all. Cache
# it and later runs with GROUPS load only what they draw.
: "${SCALE_JSON:=}"

# --- lipid_volume_panel.py --------------------------------------------------
# One run's DENSE volume/ dir. A run made with RENDER_VOXELS_ONLY=1 only has
# volume_sparse/ and cannot be volume-rendered; the script says so.
: "${VOLUME_DIR:=/mnt/e/mlibra-backup/lgb_experiment_cv/FOLD-2-LGPALL-MEAN-5-data-2000-1000-learnind-ard10/volume}"
: "${PAGE:=letter}"
: "${DOWNSAMPLE:=2}"              # 1 = full resolution, ~8x the work
: "${DPI:=400}"

cd "$SRC_PATH" || exit 1

run() {
    echo ">>> $*"
    [ "$DRY_RUN" = "1" ] || "$@" || exit 1
}

if [ "$WHICH" = "all" ] || [ "$WHICH" = "combined" ]; then
    set -- python reports/figures/combined_panels.py --out-dir "$OUT_DIR"
    if [ -n "$FROM_REPORT" ]; then
        set -- "$@" --from-report "$FROM_REPORT" --s3-root "$S3_ROOT" --fold "$FOLD"
    else
        set -- "$@" --root "$FIG_ROOT"
    fi
    [ -n "$FIG_GROUPS" ] && set -- "$@" --groups $FIG_GROUPS
    [ -n "$SCALE_JSON" ] && set -- "$@" --scale-json "$SCALE_JSON"
    run "$@"
fi

if [ "$WHICH" = "all" ] || [ "$WHICH" = "volume" ]; then
    run python reports/figures/lipid_volume_panel.py \
        --volume-dir "$VOLUME_DIR" \
        --out-dir "$OUT_DIR" \
        --page "$PAGE" \
        --downsample "$DOWNSAMPLE" \
        --dpi "$DPI"
fi

echo
echo "figures in $OUT_DIR"
