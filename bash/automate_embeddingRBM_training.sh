#!/bin/bash

DATASETS=("RR" "SH3" "Globin" "CM")
NUM_UPDATES=10000

for dataset in "${DATASETS[@]}"; do
    if [ "$dataset" == "RR" ]; then
        mkdir -p "./experiments/models/$dataset/rbm"
        PATH_DATA="./experiments/datasets/$dataset/RR.fasta"
        OUTPUT="./experiments/models/$dataset/rbm/embeddingRBM.h5"
    elif [ "$dataset" == "SH3" ]; then
        mkdir -p "./experiments/models/$dataset/rbm"
        PATH_DATA="./experiments/datasets/$dataset/SH3.fasta"
        OUTPUT="./experiments/models/$dataset/rbm/embeddingRBM.h5"
    elif [ "$dataset" == "Globin" ]; then
        mkdir -p "./experiments/models/$dataset/rbm"
        PATH_DATA="./experiments/datasets/$dataset/Globin_morkos.fasta"
        OUTPUT="./experiments/models/$dataset/rbm/embeddingRBM.h5"
    elif [ "$dataset" == "CM" ]; then
        mkdir -p "./experiments/models/$dataset/rbm"
        PATH_DATA="./experiments/datasets/$dataset/cm_russ_natural.fasta"
        OUTPUT="./experiments/models/$dataset/rbm/embeddingRBM.h5"
    else
        echo "Unknown dataset: $dataset"
        exit 1
    fi

    rbms train \
        --dataset "$PATH_DATA" \
        --train_size 1.0 \
        --num_hiddens 1024 \
        --gibbs_steps 10 \
        --learning_rate 0.01 \
        --num_updates "$NUM_UPDATES" \
        --filename "$OUTPUT"
done