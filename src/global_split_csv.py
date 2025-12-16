#!/usr/bin/env python3

import os
import argparse
import numpy as np
import torch
from adabmDCA.cobalt import prune_redundant_sequences
from adabmDCA import get_tokens, encode_sequence
from sklearn.model_selection import train_test_split
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser(description="Split MSA data into training and test sets.")
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file containing the MSA.")
    parser.add_argument("--output_prefix", type=str, default=".", help="Prefix for the output files.")
    parser.add_argument("--test_fraction", type=float, default=0.2, help="Fraction of data to use for testing.")
    parser.add_argument("--redundancy_threshold", type=float, default=0.8, help="Redundancy threshold for pruning sequences.")
    parser.add_argument("--alphabet", type=str, default="protein", help="Alphabet type: 'protein', 'rna', 'dna' or a user-defined alphabet.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dtype = torch.float32
    rnd_gen = torch.Generator(device=device).manual_seed(args.seed)
    
    tokens = get_tokens(args.alphabet)
    df_source = pd.read_csv(args.input_csv)
    sequences_all = df_source['sequence_align'].to_numpy()
    headers_all = df_source['header'].to_numpy()
    sequences_all = encode_sequence(sequences_all, tokens)
    sequences_all = torch.tensor(np.array(sequences_all), device=device, dtype=torch.long)
    print(f"Total sequences loaded: {len(sequences_all)}")
    headers_pruned, sequences_pruned = prune_redundant_sequences(
        headers_all,
        sequences_all,
        seqid_th=args.redundancy_threshold,
        rnd_gen=rnd_gen,
    )
    print(f"Sequences after pruning: {len(sequences_pruned)}")
    headers_train, headers_test, seqs_train, seqs_test = train_test_split(
        headers_pruned,
        sequences_pruned.cpu().numpy(),
        test_size=args.test_fraction,
        random_state=args.seed,
    )
    print(f"Training set size: {len(seqs_train)}")
    print(f"Test set size: {len(seqs_test)}")
    output_train_csv = args.output_prefix + ".train.csv"
    output_test_csv = args.output_prefix + ".test.csv"
    df_train = df_source[df_source['header'].isin(headers_train)]
    df_test = df_source[df_source['header'].isin(headers_test)]
    df_train.to_csv(output_train_csv, index=False)
    df_test.to_csv(output_test_csv, index=False)
    print(f"Training set written to: {output_train_csv}")
    print(f"Test set written to: {output_test_csv}")
    print("Data splitting completed successfully.")
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)