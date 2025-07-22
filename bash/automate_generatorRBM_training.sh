#!/bin/bash

# Define your parameter options
datasets=("RR")
BASE_DIR="./experiments"
label_column_names=("label_true" "label_msa" "label_foundation" "label_train" "label_foundation_0.8")

for dataset in "${datasets[@]}"; do
  if [ "$dataset" == "RR" ]; then
    t1s=(0.4 0.7)
    NUM_TRAIN_SEQS=(100)
  elif [ "$dataset" == "SH3" ]; then
    t1s=(0.7)
    NUM_TRAIN_SEQS=(100)
  elif [ "$dataset" == "Globin" ]; then
    t1s=(0.7)
    NUM_TRAIN_SEQS=(1393)
  elif [ "$dataset" == "CM" ]; then
    t1s=(0.7)
    NUM_TRAIN_SEQS=(529)
  else
    echo "Unknown dataset: $dataset"
    exit 1
  fi

  for t1 in "${t1s[@]}"; do
    INPUT_DATASET="$BASE_DIR/datasets/$dataset/input-label_rbm/predictions_t1$t1.csv"
    OUTPUT_DIR="$BASE_DIR/models/$dataset/rbm"
    for num_train_seqs in "${NUM_TRAIN_SEQS[@]}"; do
        for label_column in "${label_column_names[@]}"; do
            FLAG="labelRBM-$label_column-t1$t1-n$num_train_seqs"
            CMD="annadca train \
                -d $INPUT_DATASET \
                -o $OUTPUT_DIR \
                --column_names header \
                --column_sequences sequence_align \
                --column_label $label_column \
                -H 100 \
                -l $FLAG \
                --lr 0.01 \
                --gibbs_steps 10 \
                --nchains 5000 \
                --nepochs 30000"

            # Print and run the command
            echo "Running: $CMD"
            eval $CMD
        done
    done
  done
done