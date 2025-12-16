"""
MSA Transformer Encoding Script

Encodes aligned protein sequences using Facebook's ESM MSA Transformer models.
This script processes sequences from CSV files and generates embeddings that can be
used for downstream tasks like classification.

Usage:
    python msa_transformer_encoding.py --input data.csv --column_sequences sequence_align

Requirements:
    - fair-esm: pip install fair-esm
    - torch, numpy, pandas, biopython
"""

# !pip install biopython biotite
# !pip install git+https://github.com/facebookresearch/esm.git
# !apt-get install aria2

import torch
import numpy as np
import argparse
import pandas as pd
import os
import time
from typing import List, Tuple, Optional
import esm

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encodes aligned sequences using MSA Transformer and saves them as npz files.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input dataset in .csv format.")
    parser.add_argument("--column_sequences", type=str, default="sequence_align", help="Column name in the input .csv file containing the sequences.")
    parser.add_argument("--column_labels", type=str, default="label", help="Column name in the input .csv file containing the labels.")
    parser.add_argument("--column_headers", type=str, default="header", help="Column name in the input .csv file containing the sequence identifiers.")
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum sequence length for the MSA Transformer.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing sequences.")
    parser.add_argument("--model_name", type=str, default="esm_msa1b_t12_100M_UR50S", 
                        choices=["esm_msa1_t12_100M_UR50S", "esm_msa1b_t12_100M_UR50S"],
                        help="MSA Transformer model to use.")
    parser.add_argument("--pool_method", type=str, default="mean", 
                        choices=["mean", "cls"],
                        help="Pooling method: 'mean' for mean pooling over sequence length, 'cls' for CLS token.")
    parser.add_argument("--split_by_label", action="store_true", help="Whether to split the MSA by label.")
    return parser


def csv_to_sequences(csv_file: str, sequence_column: str, header_column: str, label_column: str) -> Tuple[List[str], List[str], Optional[List]]:
    """Load CSV file and extract sequences, headers, and labels."""
    df = pd.read_csv(csv_file)
    headers = df[header_column].to_list()
    sequences = df[sequence_column].to_list()
    labels = df[label_column].to_list() if label_column in df.columns else None
    return headers, sequences, labels


def prepare_msa_batch(sequences: List[str], headers: List[str]) -> List[Tuple[str, str]]:
    """
    Prepare MSA batch in the format expected by ESM MSA Transformer.
    Each MSA is a list of (description, sequence) tuples.
    For single sequences, we create an MSA with depth 1.
    """
    # For individual sequences, create single-sequence "MSAs"
    msa_batch = []
    for header, sequence in zip(headers, sequences):
        # Remove gaps for individual sequences if needed, or keep them for aligned sequences
        # ESM MSA Transformer expects aligned sequences with gaps
        msa_batch.append([(header, sequence)])
    return msa_batch


def batch_converter_wrapper(batch_converter, msa_batch: List[List[Tuple[str, str]]]):
    """Wrapper for batch converter to handle MSA format."""
    return batch_converter(msa_batch)


def main(config):
    assert os.path.exists(config["input"]), f"Input file {config['input']} does not exist."
    
    start_time_total = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading input dataset from {config['input']}...")
    headers, sequences, labels = csv_to_sequences(
        config["input"], 
        config["column_sequences"], 
        config["column_headers"], 
        config["column_labels"]
    )
    
    print(f"Loaded {len(sequences)} sequences")
    
    # Load MSA Transformer model
    print(f"Loading MSA Transformer model: {config['model_name']}...")
    start_time_model = time.time()
    
    # Load model using imported functions
    if config["model_name"] == "esm_msa1_t12_100M_UR50S":
        model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    elif config["model_name"] == "esm_msa1b_t12_100M_UR50S":
        model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    else:
        raise ValueError(f"Unknown model: {config['model_name']}")
    
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    
    time_model_load = time.time() - start_time_model
    print(f"Model loaded in {time_model_load:.2f}s")
    
    # Prepare MSA data
    print("Preparing MSA data...")
    start_time_prep = time.time()
    
    # Create single-sequence MSAs for each input sequence
    msa_batch = prepare_msa_batch(sequences, headers)
    
    time_prep = time.time() - start_time_prep
    print(f"MSA preparation completed in {time_prep:.2f}s")
    
    # Generate embeddings
    print("Generating embeddings...")
    start_time_embed = time.time()
    
    all_embeddings = []
    batch_size = config["batch_size"]
    num_sequences = len(msa_batch)
    
    with torch.no_grad():
        if not config["split_by_label"]:
            # Process all sequences together
            for start_idx in range(0, num_sequences, batch_size):
                end_idx = min(start_idx + batch_size, num_sequences)
                print(f"Processing sequences {start_idx} to {end_idx} / {num_sequences}")
                
                batch = msa_batch[start_idx:end_idx]
                
                # Convert batch to model input format
                msa_labels, msa_strs, msa_tokens = batch_converter(batch)
                msa_tokens = msa_tokens.to(device)
                
                # Forward pass
                results = model(msa_tokens, repr_layers=[12], return_contacts=False)
                
                # Extract representations from the last layer
                # Shape: (batch_size, num_alignments, seq_len, hidden_dim)
                # For single-sequence MSAs, num_alignments = 1
                representations = results["representations"][12]
                
                # Remove the alignment dimension (since we have single-sequence MSAs)
                # Shape: (batch_size, seq_len, hidden_dim)
                representations = representations[:, 0, :, :]
                
                # Apply pooling
                if config["pool_method"] == "mean":
                    # Mean pooling over sequence length (excluding special tokens)
                    # Tokens: <cls> seq <eos> <pad>...
                    # We'll compute mean over positions 1:-1 (excluding <cls> and <eos>)
                    seq_lens = (msa_tokens[:, 0, :] != alphabet.padding_idx).sum(1)
                    batch_embeddings = []
                    for i, seq_len in enumerate(seq_lens):
                        # Mean pool from position 1 to seq_len-1 (excluding <cls> at 0 and <eos> at seq_len-1)
                        embedding = representations[i, 1:seq_len-1, :].mean(0)
                        batch_embeddings.append(embedding.cpu().numpy())
                    batch_embeddings = np.stack(batch_embeddings)
                elif config["pool_method"] == "cls":
                    # Use the <cls> token representation (position 0)
                    batch_embeddings = representations[:, 0, :].cpu().numpy()
                else:
                    raise ValueError(f"Unknown pooling method: {config['pool_method']}")
                
                all_embeddings.append(batch_embeddings)
        else:
            # Process each label separately
            if labels is None:
                raise ValueError("Cannot split by label when labels are not provided in the dataset")
            
            unique_labels = list(set(labels))
            print(f"Processing {len(unique_labels)} unique labels separately")
            
            # Create mapping of label to indices
            label_to_indices = {}
            for idx, label in enumerate(labels):
                if label not in label_to_indices:
                    label_to_indices[label] = []
                label_to_indices[label].append(idx)
            
            # Store embeddings with original indices to reorder later
            embeddings_dict = {}
            
            for label in unique_labels:
                print(f"Processing sequences with label {label}")
                label_indices = label_to_indices[label]
                label_batch = [msa_batch[i] for i in label_indices]
                
                label_embeddings = []
                for start_idx in range(0, len(label_batch), batch_size):
                    end_idx = min(start_idx + batch_size, len(label_batch))
                    print(f"  Processing sequences {start_idx} to {end_idx} / {len(label_batch)}")
                    
                    batch = label_batch[start_idx:end_idx]
                    
                    # Convert batch to model input format
                    msa_labels, msa_strs, msa_tokens = batch_converter(batch)
                    msa_tokens = msa_tokens.to(device)
                    
                    # Forward pass
                    results = model(msa_tokens, repr_layers=[12], return_contacts=False)
                    
                    # Extract representations from the last layer
                    representations = results["representations"][12]
                    
                    # Remove the alignment dimension
                    representations = representations[:, 0, :, :]
                    
                    # Apply pooling
                    if config["pool_method"] == "mean":
                        seq_lens = (msa_tokens[:, 0, :] != alphabet.padding_idx).sum(1)
                        batch_embeddings = []
                        for i, seq_len in enumerate(seq_lens):
                            embedding = representations[i, 1:seq_len-1, :].mean(0)
                            batch_embeddings.append(embedding.cpu().numpy())
                        batch_embeddings = np.stack(batch_embeddings)
                    elif config["pool_method"] == "cls":
                        batch_embeddings = representations[:, 0, :].cpu().numpy()
                    else:
                        raise ValueError(f"Unknown pooling method: {config['pool_method']}")
                    
                    label_embeddings.append(batch_embeddings)
                
                # Store embeddings for this label with their original indices
                label_embeddings_concat = np.concatenate(label_embeddings, axis=0)
                for i, orig_idx in enumerate(label_indices):
                    embeddings_dict[orig_idx] = label_embeddings_concat[i]
            
            # Reorder embeddings to match original sequence order
            embeddings_list = [embeddings_dict[i] for i in range(len(sequences))]
            all_embeddings = [np.stack(embeddings_list)]
    
    # Concatenate all embeddings
    embeddings = np.concatenate(all_embeddings, axis=0)
    
    time_embed = time.time() - start_time_embed
    print(f"Embedding generation completed in {time_embed:.2f}s")
    print(f"Final embeddings shape: {embeddings.shape}")
    
    # Prepare output filename
    output_prefix = os.path.splitext(config["input"])[0]
    output_path = f"{output_prefix}.msa_transformer.npz"
    
    # Save to npz file
    print(f"Saving MSA Transformer encoding to {output_path}...")
    if labels is not None:
        np.savez_compressed(
            output_path,
            embeddings=embeddings,
            labels=np.array(labels),
            headers=np.array(headers)
        )
    else:
        np.savez_compressed(
            output_path,
            embeddings=embeddings,
            headers=np.array(headers)
        )
    
    print(f"Successfully saved MSA Transformer encoding to {output_path}")
    
    time_total = time.time() - start_time_total
    
    print("\n" + "="*60)
    print("PROCESSING TIME SUMMARY:")
    print("="*60)
    print(f"Model loading:        {time_model_load:.2f}s")
    print(f"MSA preparation:      {time_prep:.2f}s")
    print(f"Embedding generation: {time_embed:.2f}s")
    print(f"Total time:           {time_total:.2f}s")
    print("="*60)
    print("Done!")

    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = vars(args)
    main(config)
