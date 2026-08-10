#!/usr/bin/env sh
# GPLFR (GP latent factor regression, arXiv:2606.06576) on MALDI -- thin wrapper
# pinning MODEL=gplfr. run_sota.sh handles it inline: the base latent GP is
# BASE_GP={euclidean|riemann|spectral} and the GPLFR_* / manifold knobs
# (LATENT_DIM, NUM_INDUCING, INVERSE_TEMPERATURE, NUM_MODES, ...) apply. Runs the
# MaldiExperiment harness with whole-brain reconstruction (--do-brain-reconstruction).
# The manifold bases (riemann|spectral) need the eigenpair pipeline
# (EIGENVECTOR_DIR + graph/spectrum knobs).
#
#   BASE_GP=spectral LATENT_DIM=8 ./local_run/run_gplfr.sh
#   BASE_GP=euclidean ./local_run/run_gplfr.sh      # no eigenpair pipeline needed
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export MODEL=gplfr
exec "$SCRIPT_DIR/run_sota.sh" "$@"
