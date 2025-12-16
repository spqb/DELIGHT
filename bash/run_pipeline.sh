#!/bin/bash

# Pipeline script for protein sequence embedding and classification
# Usage: bash run_pipeline.sh <csv_pool_file> <output_folder> <rbm_path>

set -e  # Exit on error

# Disable progress bars for tqdm and other libraries
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1

# Check if correct number of arguments provided
if [ "$#" -ne 3 ]; then
    echo "Usage: bash run_pipeline.sh <csv_pool_file> <output_folder> <rbm_path>"
    exit 1
fi

# Input arguments
CSV_POOL_FILE="$1"
OUTPUT_FOLDER="$2"
RBM_PATH="$3"

# Validate inputs
if [ ! -f "$CSV_POOL_FILE" ]; then
    echo "Error: CSV pool file '$CSV_POOL_FILE' not found!"
    exit 1
fi

if [ ! -f "$RBM_PATH" ]; then
    echo "Error: RBM model file '$RBM_PATH' not found!"
    exit 1
fi

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Define log file
LOG_FILE="$OUTPUT_FOLDER/pipeline.log"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Start logging
log "=========================================="
log "Starting pipeline"
log "=========================================="
log "CSV Pool File: $CSV_POOL_FILE"
log "Output Folder: $OUTPUT_FOLDER"
log "RBM Path: $RBM_PATH"
log ""

# Define seeds
#SEEDS=(0)
SEEDS=($(seq 152 200))

# Main pipeline loop
for SEED in "${SEEDS[@]}"; do
    log "=========================================="
    log "Processing SEED: $SEED"
    log "=========================================="
    
    # Define file prefixes
    POOL_PREFIX="${OUTPUT_FOLDER}/pool_seed_${SEED}"
    SEED_PREFIX="${OUTPUT_FOLDER}/seed_${SEED}"
    
    # Step 1: Run cobalt_split_csv.py
    log "Step 1: Running cobalt_split_csv.py..."
    python src/cobalt_split_csv.py \
        --input_csv "$CSV_POOL_FILE" \
        --output_prefix "$POOL_PREFIX" \
        --t1 0.4 \
        --t2 1.0 \
        --t3 0.7 \
        --seed "$SEED" >> "$LOG_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        log "ERROR: cobalt_split_csv.py failed for seed $SEED"
        continue
    fi
    log "cobalt_split_csv.py completed successfully"
    log ""
    
    # Step 2: Run subsample_csv.py
    log "Step 2: Running subsample_csv.py..."
    python src/subsample_csv.py \
        --train_csv "${POOL_PREFIX}.train.csv" \
        --test_csv "${POOL_PREFIX}.test.csv" \
        --output_prefix "$SEED_PREFIX" \
        --num_samples_list 100 \
        --seed "$SEED" >> "$LOG_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        log "ERROR: subsample_csv.py failed for seed $SEED"
        continue
    fi
    log "subsample_csv.py completed successfully"
    log ""
    
    # Define train and test files
    TRAIN_FILE="${SEED_PREFIX}.train_100.csv"
    TEST_FILE="${SEED_PREFIX}.test.csv"
    
    # Verify files were created
    if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$TEST_FILE" ]; then
        log "ERROR: Expected output files not found for seed $SEED"
        log "Missing: $TRAIN_FILE or $TEST_FILE"
        continue
    fi
    
    log "Generated files:"
    log "  Train: $TRAIN_FILE"
    log "  Test: $TEST_FILE"
    log ""
    
    # Step 3: Run embedders
    log "Step 3: Running embedders..."
    log ""
    
    # 3a. One-hot encoding
    log "  3a. Running onehot_encoding.py on train set..."
    python src/embedders/onehot_encoding.py \
        --input "$TRAIN_FILE" >> "$LOG_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        log "ERROR: onehot_encoding.py failed for train set (seed $SEED)"
    else
        log "onehot_encoding.py completed for train set"
    fi
    log ""
    
    log "  3a. Running onehot_encoding.py on test set..."
    python src/embedders/onehot_encoding.py \
        --input "$TEST_FILE" >> "$LOG_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        log "ERROR: onehot_encoding.py failed for test set (seed $SEED)"
    else
        log "onehot_encoding.py completed for test set"
    fi
    log ""
    
    # 3b. RBM encoding
    log "  3b. Running rbm_encoding.py on train set..."
    python src/embedders/rbm_encoding.py \
        --input "$TRAIN_FILE" \
        --rbm_model_path "$RBM_PATH" >> "$LOG_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        log "ERROR: rbm_encoding.py failed for train set (seed $SEED)"
    else
        log "rbm_encoding.py completed for train set"
    fi
    log ""
    
    log "  3b. Running rbm_encoding.py on test set..."
    python src/embedders/rbm_encoding.py \
        --input "$TEST_FILE" \
        --rbm_model_path "$RBM_PATH" >> "$LOG_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        log "ERROR: rbm_encoding.py failed for test set (seed $SEED)"
    else
        log "rbm_encoding.py completed for test set"
    fi
    log ""
    
    # 3c. PLM encoding (zero-shot with foundation flag)
    log "  3c. Running plm_encoding.py (zero-shot)..."
    python src/embedders/plm_encoding.py \
        --train "$TRAIN_FILE" \
        --query "$TEST_FILE" \
        --flag "foundation" \
        --zero-shot >> "$LOG_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        log "ERROR: plm_encoding.py failed (seed $SEED)"
    else
        log "plm_encoding.py completed"
    fi
    log ""
    
    # 3d. MSA Pairformer encoding
    log "  3d. Running msa_pairformer_encoding.py on train set..."
    python src/embedders/msa_pairformer_encoding.py \
        --input "$TRAIN_FILE" >> "$LOG_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        log "ERROR: msa_pairformer_encoding.py failed for train set (seed $SEED)"
    else
        log "msa_pairformer_encoding.py completed for train set"
    fi
    log ""
    
    log "  3d. Running msa_pairformer_encoding.py on test set..."
    python src/embedders/msa_pairformer_encoding.py \
        --input "$TEST_FILE" >> "$LOG_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        log "ERROR: msa_pairformer_encoding.py failed for test set (seed $SEED)"
    else
        log "msa_pairformer_encoding.py completed for test set"
    fi
    log ""
    
    # Step 4: Run predictions
    log "Step 4: Running predictions with predict_from_embeddings.py..."
    log ""
    
    # Define embedding file prefixes
    TRAIN_BASE="${SEED_PREFIX}.train_100"
    TEST_BASE="${SEED_PREFIX}.test"
    
    # 4a. One-hot predictions
    if [ -f "${TRAIN_BASE}.onehot.npz" ] && [ -f "${TEST_BASE}.onehot.npz" ]; then
        log "  4a. Running predictions on one-hot embeddings..."
        python src/predict_from_embeddings.py \
            --train_npz "${TRAIN_BASE}.onehot.npz" \
            --test_npz "${TEST_BASE}.onehot.npz" >> "$LOG_FILE" 2>&1
        
        if [ $? -ne 0 ]; then
            log "ERROR: predict_from_embeddings.py failed for one-hot (seed $SEED)"
        else
            log "Predictions completed for one-hot embeddings"
        fi
        log ""
    else
        log "WARNING: One-hot embedding files not found, skipping predictions"
        log ""
    fi
    
    # 4b. RBM predictions
    if [ -f "${TRAIN_BASE}.rbm.npz" ] && [ -f "${TEST_BASE}.rbm.npz" ]; then
        log "  4b. Running predictions on RBM embeddings..."
        python src/predict_from_embeddings.py \
            --train_npz "${TRAIN_BASE}.rbm.npz" \
            --test_npz "${TEST_BASE}.rbm.npz" >> "$LOG_FILE" 2>&1
        
        if [ $? -ne 0 ]; then
            log "ERROR: predict_from_embeddings.py failed for RBM (seed $SEED)"
        else
            log "Predictions completed for RBM embeddings"
        fi
        log ""
    else
        log "WARNING: RBM embedding files not found, skipping predictions"
        log ""
    fi
    
    # 4c. PLM (foundation) predictions
    if [ -f "${TRAIN_BASE}.foundation.npz" ] && [ -f "${TEST_BASE}.foundation.npz" ]; then
        log "  4c. Running predictions on foundation (PLM) embeddings..."
        python src/predict_from_embeddings.py \
            --train_npz "${TRAIN_BASE}.foundation.npz" \
            --test_npz "${TEST_BASE}.foundation.npz" >> "$LOG_FILE" 2>&1
        
        if [ $? -ne 0 ]; then
            log "ERROR: predict_from_embeddings.py failed for foundation (seed $SEED)"
        else
            log "Predictions completed for foundation embeddings"
        fi
        log ""
    else
        log "WARNING: Foundation embedding files not found, skipping predictions"
        log ""
    fi
    
    # 4d. MSA Pairformer predictions
    if [ -f "${TRAIN_BASE}.msa_pairformer.npz" ] && [ -f "${TEST_BASE}.msa_pairformer.npz" ]; then
        log "  4d. Running predictions on MSA Pairformer embeddings..."
        python src/predict_from_embeddings.py \
            --train_npz "${TRAIN_BASE}.msa_pairformer.npz" \
            --test_npz "${TEST_BASE}.msa_pairformer.npz" >> "$LOG_FILE" 2>&1
        
        if [ $? -ne 0 ]; then
            log "ERROR: predict_from_embeddings.py failed for MSA Pairformer (seed $SEED)"
        else
            log "Predictions completed for MSA Pairformer embeddings"
        fi
        log ""
    else
        log "WARNING: MSA Pairformer embedding files not found, skipping predictions"
        log ""
    fi
    
    # Step 5: Clean up intermediate files (keep only .predictions.h5 files)
    log "Step 5: Cleaning up intermediate files..."
    
    # Remove CSV files
    rm -f "${POOL_PREFIX}.train.csv" "${POOL_PREFIX}.test.csv" 2>/dev/null || true
    rm -f "${SEED_PREFIX}.train_100.csv" "${SEED_PREFIX}.test.csv" 2>/dev/null || true
    
    # Remove embedding files (.npz)
    rm -f "${TRAIN_BASE}.onehot.npz" "${TEST_BASE}.onehot.npz" 2>/dev/null || true
    rm -f "${TRAIN_BASE}.rbm.npz" "${TEST_BASE}.rbm.npz" 2>/dev/null || true
    rm -f "${TRAIN_BASE}.foundation.npz" "${TEST_BASE}.foundation.npz" 2>/dev/null || true
    rm -f "${TRAIN_BASE}.msa_pairformer.npz" "${TEST_BASE}.msa_pairformer.npz" 2>/dev/null || true
    
    log "Cleanup completed - kept only .predictions.h5 files"
    log ""
    
    log "=========================================="
    log "Completed processing for SEED: $SEED"
    log "=========================================="
    log ""
done

log "=========================================="
log "Pipeline completed for all seeds"
log "=========================================="
log "Results saved in: $OUTPUT_FOLDER"
log "Log file: $LOG_FILE"
log ""

# Summary
log "Generated files summary:"
for SEED in "${SEEDS[@]}"; do
    log "Seed $SEED:"
    ls -lh "${OUTPUT_FOLDER}/seed_${SEED}"* >> "$LOG_FILE" 2>&1 || true
    log ""
done

log "Pipeline finished successfully!"
