echo "Starting repeated analysis for RR"

SEEDS=(1 2 3 4 5 6 7 8 9 10)
BASE_DIR="./experiments"
SOURCE_FILE="./experiments/datasets/RR/RR.csv"
NUM_TRAIN_SEQS=(100 500 1000 2000)
T1=(0.4 0.7)
DATASET="RR"  # Add missing DATASET variable
MODES=("contrastive" "foundation")  # Define MODES array

for t1 in "${T1[@]}"; do
    folder_id="t1${t1}_t20.7_repeated"
    OUTPUT_DIR="./experiments/datasets/$DATASET/$folder_id"
    
    for SEED in "${SEEDS[@]}";
    do
      echo "Running analysis for SEED=$SEED"
      # Train/Test split using cobalt
      echo "Splitting dataset with t1=$t1, t2=1.0, t3=0.7"
      python ./src/cobalt_split_csv.py \
        --input_csv "$SOURCE_FILE" \
        --output_prefix "$OUTPUT_DIR/full_seed_${SEED}" \
        --seed $SEED \
        --t1 $t1 \
        --t2 1.0 \
        --t3 0.7 \
        --min_train 30

      # Subsample training sets
      echo "Subsampling training sets for various sizes"
      python ./src/subsample_csv.py \
        --input_train_csv "$OUTPUT_DIR/full_seed_${SEED}.train.csv" \
        --input_test_csv "$OUTPUT_DIR/full_seed_${SEED}.test.csv" \
        --output_prefix "$OUTPUT_DIR/seed_${SEED}" \
        --num_samples "${NUM_TRAIN_SEQS[@]}" \
        --seed $SEED

      # pLM embeddings
        for num_train_seqs in "${NUM_TRAIN_SEQS[@]}"; do
            for mode in "${MODES[@]}"; do  # Fix: use array syntax instead of parentheses
                OUTPUT_MODEL_DIR="$BASE_DIR/models/$DATASET/prot_bert/$folder_id/seed_$SEED/$mode/$num_train_seqs"
                if [ "$mode" == "contrastive" ]; then
                  mkdir -p "$OUTPUT_MODEL_DIR"
                fi
                TRAIN_FILE="$OUTPUT_DIR/seed_${SEED}.train_${num_train_seqs}.csv"  # Fix: use OUTPUT_DIR instead of undefined DATASETS_DIR
                QUERY_FILE="$OUTPUT_DIR/seed_${SEED}.test.csv"  # Fix: use OUTPUT_DIR instead of undefined DATASETS_DIR
                FLAG="embedding_${mode}_${num_train_seqs}"
        
                CMD="python src/pLM_encoding.py \
                  --train $TRAIN_FILE \
                  --query $QUERY_FILE \
                  --output $OUTPUT_MODEL_DIR \
                  --flag $FLAG \
                  --epochs 50 \
                  --patience 5 \
                  --bf16 \
                  --save_steps 20"
        
                if [ "$mode" == "foundation" ]; then
                  CMD="$CMD --zero-shot"
                fi
        
                echo "Running: $CMD"
                eval $CMD
            done  # Close mode loop
        done  # Close num_train_seqs loop
    done  # Close SEED loop
done  # Close t1 loop
python ./src/one_hot_and_RBM_encoding_repeated.py