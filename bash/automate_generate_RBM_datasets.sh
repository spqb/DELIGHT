#!/bin/bash

# Define your parameter options
datasets=("RR")
BASE_DIR="./experiments"
SCRIPT="./src/generate_rbm_datasets.py"

for dataset in "${datasets[@]}"; do
  if [ "$dataset" == "RR" ]; then
    t1s=(0.4 0.7)
    NUM_TRAIN_SEQS=(100 500 1000 2000)
  elif [ "$dataset" == "SH3" ]; then
    t1s=(0.7)
    NUM_TRAIN_SEQS=(100 500 1000)
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
    for num_train_seqs in "${NUM_TRAIN_SEQS[@]}"; do

      CMD="python $SCRIPT \
        --dataset_name $dataset \
        --t1 $t1 \
        --num_training_samples $num_train_seqs"

      # Print and run the command
      echo "Running: $CMD"
      eval $CMD
    done
  done
done