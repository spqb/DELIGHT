#!/bin/bash

# Define your parameter options
datasets=("RR" "SH3" "Globin" "CM")
MODES=("contrastive" "foundation")
BASE_DIR="./experiments"

for dataset in "${datasets[@]}"; do
  if [ "$dataset" == "RR" ]; then
    folder_ids=("t10.4_t20.7" "t10.7_t20.7")
    NUM_TRAIN_SEQS=(100 500 1000 2000)
    epochs=50
    patience=5
  elif [ "$dataset" == "SH3" ]; then
    folder_ids=("t10.7_t20.7")
    NUM_TRAIN_SEQS=(100 500 1000)
    epochs=20
    patience=5
  elif [ "$dataset" == "Globin" ]; then
    folder_ids=("t10.7_t20.7")
    NUM_TRAIN_SEQS=(1393)
    epochs=20
    patience=8
  elif [ "$dataset" == "CM" ]; then
    folder_ids=("t10.7_t20.7")
    NUM_TRAIN_SEQS=(529)
    epochs=20
    patience=5
  else
    echo "Unknown dataset: $dataset"
    exit 1
  fi

  for folder_id in "${folder_ids[@]}"; do
    # Base paths
    DATASETS_DIR="$BASE_DIR/datasets/$dataset"
    SCRIPT="src/pLM_encoding.py"

    # Loop over all combinations
    for num_train_seqs in "${NUM_TRAIN_SEQS[@]}"; do
      for mode in "${MODES[@]}"; do
        
        # Construct file paths and flags
        OUTPUT_DIR="$BASE_DIR/models/$dataset/prot_bert/$folder_id/$mode/$num_train_seqs"
        # create output directory if it doesn't exist
        if [ "$mode" == "contrastive" ]; then
          mkdir -p "$OUTPUT_DIR"
        fi
        
        TRAIN_FILE="$DATASETS_DIR/$folder_id/train_${num_train_seqs}.csv"
        QUERY_FILE="$DATASETS_DIR/$folder_id/test.csv"
        FLAG="embedding_${mode}_${num_train_seqs}"

        # Build command
        CMD="python $SCRIPT \
          --train $TRAIN_FILE \
          --query $QUERY_FILE \
          --output $OUTPUT_DIR \
          --flag $FLAG \
          --epochs $epochs \
          --patience $patience \
          --bf16 \
          --save_steps 20"

        # Add zero-shot flag if mode is foundation
        if [ "$mode" == "foundation" ]; then
          CMD="$CMD --zero-shot"
        fi

        # Print and run the command
        echo "Running: $CMD"
        eval $CMD

      done
    done
  done
done