import argparse
import numpy as np
import pandas as pd
from adabmDCA.fasta import get_tokens, encode_sequence
import os


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encodes aligned sequences using one-hot encoding and saves them as npz files.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input dataset in .csv format.")
    parser.add_argument("--column_sequences", type=str, default="sequence_align", help="Column name in the input .csv file containing the sequences.")
    parser.add_argument("--column_labels", type=str, default="label", help="Column name in the input .csv file containing the labels.")
    parser.add_argument("--column_headers", type=str, default="header", help="Column name in the input .csv file containing the sequence identifiers.")
    
    return parser


def one_hot_encode_sequences(sequences, tokens):
    """
    One-hot encode a list of aligned sequences.
    
    Args:
        sequences: List of aligned protein sequences (all same length)
        tokens: Token list from adabmDCA
    
    Returns:
        One-hot encoded array of shape (n_sequences, length * n_tokens)
    """
    # Encode sequences to integer representation
    encoded = encode_sequence(sequences, tokens)
    
    # One-hot encode: convert to (n_sequences, length, n_tokens) then flatten
    one_hot = np.eye(len(tokens))[encoded]
    # Reshape to (n_sequences, length * n_tokens)
    one_hot = one_hot.reshape(len(sequences), -1)
    
    return one_hot


def main(config):
    assert os.path.exists(config["input"]), f"Input file {config['input']} does not exist."
    
    # Get protein tokens
    tokens = get_tokens("protein")
    print(f"Using {len(tokens)} tokens for one-hot encoding: {tokens}")
    
    print(f"Loading input dataset from {config['input']}...")
    
    # Load CSV file
    df = pd.read_csv(config["input"])
    
    # Check required columns
    assert config["column_sequences"] in df.columns, f"Column '{config['column_sequences']}' not found in CSV file"
    assert config["column_headers"] in df.columns, f"Column '{config['column_headers']}' not found in CSV file"
    
    sequences = df[config["column_sequences"]].values
    headers = df[config["column_headers"]].values
    labels = df[config["column_labels"]].values if config["column_labels"] in df.columns else None
    
    print(f"Loaded {len(sequences)} sequences from CSV file")
    if labels is not None:
        print(f"Found labels with {len(np.unique(labels))} unique values")
    
    # One-hot encode sequences
    print("One-hot encoding aligned sequences...")
    one_hot = one_hot_encode_sequences(sequences, tokens)
    print(f"One-hot encoded shape: {one_hot.shape}")
    
    # Prepare output filename
    output_prefix = os.path.splitext(config["input"])[0]
    output_path = f"{output_prefix}.onehot.npz"
    
    # Save to npz file
    print(f"Saving one-hot encoding to {output_path}...")
    if labels is not None:
        np.savez_compressed(
            output_path,
            embeddings=one_hot,
            labels=labels,
            headers=headers
        )
    else:
        np.savez_compressed(
            output_path,
            embeddings=one_hot,
            headers=headers
        )
    
    print(f"Successfully saved one-hot encoding to {output_path}")
    print("Done!")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = vars(args)
    main(config)
