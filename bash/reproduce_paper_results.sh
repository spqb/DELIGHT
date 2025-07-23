#!/bin/bash

echo "Starting the embedding automation script..."
chmod +x ./bash/automate_pLM_embeddings.sh
./bash/automate_pLM_embeddings.sh

echo "Starting the encoding RBM training script..."
chmod +x ./bash/automate_embeddingRBM_training.sh
./bash/automate_embeddingRBM_training.sh

echo "Starting the one-hot and RBM encoding script..."
python3 ./src/one_hot_and_RBM_encoding.py

echo "Starting the label-RBM input dataset generation script..."
chmod +x ./bash/automate_generate_RBM_datasets.sh
./bash/automate_generate_RBM_datasets.sh

echo "Starting the label-RBM training script..."
chmod +x ./bash/automate_generatorRBM_training.sh
./bash/automate_generatorRBM_training.sh
