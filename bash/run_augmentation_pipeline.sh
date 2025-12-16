#!/bin/bash

# Script to run the full augmentation pipeline:
# 1. Generate augmented sequences using DCA
# 2. Encode sequences using RBM
# 3. Predict labels using classifiers

# Parameters for augmentation
NUM_SEQUENCES=1000
NUM_STEPS=50
COLUMN_SEQUENCE="sequence_align"

# Set up paths
BASE_DIR="/home/lorenzo/Documents/DELIGHT_dev"
DATA_DIR="${BASE_DIR}/experiments/datasets/RR/data_augmentation"
OUTPUT_DIR="${DATA_DIR}/x${NUM_SEQUENCES}-${NUM_STEPS}_mut"
DCA_MODEL="${BASE_DIR}/experiments/models/RR/augmenters/bmDCA_params.dat"
RBM_MODEL="${BASE_DIR}/experiments/models/RR/rbm/rbm_ptt_last_model.h5"

# Python scripts
AUGMENTER_SCRIPT="${BASE_DIR}/src/augmenter.py"
RBM_ENCODING_SCRIPT="${BASE_DIR}/src/embedders/rbm_encoding.py"
PREDICT_SCRIPT="${BASE_DIR}/src/predict_from_embeddings.py"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Starting augmentation pipeline"
echo "=========================================="
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "DCA model: ${DCA_MODEL}"
echo "RBM model: ${RBM_MODEL}"
echo "=========================================="

# Loop through seeds 1-10
for i in {1..10}; do
    SEED=$i
    TRAIN_FILE="${DATA_DIR}/seed_${SEED}.train_100.csv"
    TEST_FILE="${DATA_DIR}/seed_${SEED}.test.csv"
    
    echo ""
    echo "=========================================="
    echo "Processing seed ${SEED}"
    echo "=========================================="
    
    # Check if train file exists
    if [ ! -f "${TRAIN_FILE}" ]; then
        echo "WARNING: Training file ${TRAIN_FILE} not found, skipping..."
        continue
    fi
    
    # Check if test file exists
    if [ ! -f "${TEST_FILE}" ]; then
        echo "WARNING: Test file ${TEST_FILE} not found, skipping..."
        continue
    fi
    
    # ==========================================
    # BASELINE: Process original training data (no augmentation)
    # ==========================================
    echo ""
    echo "BASELINE: Processing original training data (seed ${SEED})..."
    
    # Copy original training file to output directory
    BASELINE_TRAIN="${OUTPUT_DIR}/seed_${SEED}.train_100_baseline.csv"
    cp "${TRAIN_FILE}" "${BASELINE_TRAIN}"
    
    # RBM encoding for baseline training set
    echo "BASELINE: RBM encoding for baseline training set (seed ${SEED})..."
    python "${RBM_ENCODING_SCRIPT}" \
        --input "${BASELINE_TRAIN}" \
        --rbm_model_path "${RBM_MODEL}" \
        --column_sequences "${COLUMN_SEQUENCE}" \
        --column_labels "label" \
        --column_headers "header"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: RBM encoding failed for baseline training set (seed ${SEED})"
        continue
    fi
    
    echo "BASELINE: RBM encoding completed for baseline training set (seed ${SEED})"
    
    # Copy test file to output directory
    TEST_FILE_COPY="${OUTPUT_DIR}/seed_${SEED}.test.csv"
    cp "${TEST_FILE}" "${TEST_FILE_COPY}"
    
    # RBM encoding for test set (used for both baseline and augmented)
    echo "BASELINE: RBM encoding for test set (seed ${SEED})..."
    python "${RBM_ENCODING_SCRIPT}" \
        --input "${TEST_FILE_COPY}" \
        --rbm_model_path "${RBM_MODEL}" \
        --column_sequences "${COLUMN_SEQUENCE}" \
        --column_labels "label" \
        --column_headers "header"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: RBM encoding failed for test set (seed ${SEED})"
        continue
    fi
    
    echo "BASELINE: RBM encoding completed for test set (seed ${SEED})"
    
    # Predict from baseline embeddings
    echo "BASELINE: Predicting labels for baseline (seed ${SEED})..."
    BASELINE_TRAIN_NPZ="${OUTPUT_DIR}/seed_${SEED}.train_100_baseline.rbm.npz"
    TEST_NPZ="${OUTPUT_DIR}/seed_${SEED}.test.rbm.npz"
    
    python "${PREDICT_SCRIPT}" \
        --train_npz "${BASELINE_TRAIN_NPZ}" \
        --test_npz "${TEST_NPZ}" \
        --flag "baseline_seed_${SEED}"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Baseline prediction failed for seed ${SEED}"
        continue
    fi
    
    echo "BASELINE: Prediction completed for seed ${SEED}"
    echo "=========================================="
    
    # ==========================================
    # AUGMENTED: Process with data augmentation
    # ==========================================
    
    # Step 1: Generate augmented sequences
    echo ""
    echo "Step 1: Generating augmented sequences for seed ${SEED}..."
    AUGMENTED_TRAIN="${OUTPUT_DIR}/seed_${SEED}.train_100_augmented.csv"
    
    python "${AUGMENTER_SCRIPT}" \
        --input_csv "${TRAIN_FILE}" \
        --output_csv "${AUGMENTED_TRAIN}" \
        --DCA_params "${DCA_MODEL}" \
        --num_sequences ${NUM_SEQUENCES} \
        --num_steps ${NUM_STEPS} \
        --column_sequence "${COLUMN_SEQUENCE}" \
        --column_name "header" \
        --column_label "label"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Augmentation failed for seed ${SEED}"
        continue
    fi
    
    echo "Augmentation completed for seed ${SEED}"
    
    # Step 2a: RBM encoding for augmented training set
    echo ""
    echo "Step 2a: RBM encoding for augmented training set (seed ${SEED})..."
    
    python "${RBM_ENCODING_SCRIPT}" \
        --input "${AUGMENTED_TRAIN}" \
        --rbm_model_path "${RBM_MODEL}" \
        --column_sequences "${COLUMN_SEQUENCE}" \
        --column_labels "label" \
        --column_headers "header"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: RBM encoding failed for augmented training set (seed ${SEED})"
        continue
    fi
    
    echo "RBM encoding completed for augmented training set (seed ${SEED})"
    
    # Step 3: Predict from augmented embeddings (test set already encoded in baseline step)
    echo ""
    echo "Step 3: Predicting labels for augmented data (seed ${SEED})..."
    
    TRAIN_NPZ="${OUTPUT_DIR}/seed_${SEED}.train_100_augmented.rbm.npz"
    TEST_NPZ="${OUTPUT_DIR}/seed_${SEED}.test.rbm.npz"
    
    python "${PREDICT_SCRIPT}" \
        --train_npz "${TRAIN_NPZ}" \
        --test_npz "${TEST_NPZ}" \
        --flag "augmented_seed_${SEED}"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Prediction failed for augmented data (seed ${SEED})"
        continue
    fi
    
    echo "Augmented prediction completed for seed ${SEED}"
    echo "=========================================="
    echo "Seed ${SEED} processing complete (baseline + augmented)!"
    echo "=========================================="
    
done

echo ""
echo "=========================================="
echo "Pipeline completed for all seeds!"
echo "=========================================="
echo "Results saved in: ${OUTPUT_DIR}"
echo ""
