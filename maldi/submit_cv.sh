#!/usr/bin/env sh5
DEPLOY_SCRIPT="./deploy.sh"
zsh $DEPLOY_SCRIPT
MEM=50G
CPU=15
GPU=0.5
INDUCINGS=(500 1500)
LATENT_DIMS=(5 20)
BATCH_SIZES=(1000 2000)
FOLDS=(1 2 3 4 5)
for batch_size in  $BATCH_SIZES; do
    for INDUCING in $INDUCINGS; do
      for LATENT_DIM in $LATENT_DIMS; do
        for FOLD in $FOLDS; do
          JOB_NAME="gp-postpred-cv-${FOLD}-${INDUCING}-${LATENT_DIM}-${batch_size}"
          runai workspace submit  "$JOB_NAME" \
            --preemptible \
            --gpu-request-type portion \
            --cpu-core-request "$CPU" \
            --cpu-core-limit "$CPU" \
            --gpu-portion-request "$GPU" \
            -i registry.renkulab.io/daniel.trejobanos1/mlibra \
            --cpu-memory-limit "$MEM" \
            --cpu-memory-request "$MEM" \
            -p mlibra-daniel \
            --command -- \
            bash /myhome/mlibra/maldi/run_all.sh "$INDUCING" "$LATENT_DIM" "$FOLD" "$batch_size"
        done
      done
    done
done
