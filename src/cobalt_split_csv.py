#!/usr/bin/env python3

# takes a csv file in input and splits it using Cobalt into training and test sets for each label
# saves the resulting csv files into a specified directory

import pandas as pd
import os
import torch
import argparse
import time
from adabmDCA.fasta import get_tokens, encode_sequence
from adabmDCA.cobalt import run_cobalt

def get_parser():
    parser = argparse.ArgumentParser(description="Split CSV dataset using Cobalt")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_prefix", type=str, required=True, help="Prefix for the output CSV files. The files will be saved as <output_prefix>.train.csv and <output_prefix>.test.csv")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for splitting")
    parser.add_argument("--t1", type=float, default=0.4, help="Cobalt T1 threshold")
    parser.add_argument("--t2", type=float, default=1.0, help="Cobalt T2 threshold")
    parser.add_argument("--t3", type=float, default=0.7, help="Cobalt T3 threshold")
    parser.add_argument("--min_train", type=int, default=None, help="Maximum number of training samples per label")
    return parser

def main(args):
    start_time_total = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dtype = torch.float32
    tokens = get_tokens("protein")

    df_source = pd.read_csv(args.input_csv)
    # check that the dataframe contains the required columns
    required_columns = ['header', 'sequence', 'sequence_align', 'label']
    for col in required_columns:
        if col not in df_source.columns:
            raise ValueError(f"Input CSV is missing required column: {col}")
        
    # create output directory if it does not exist
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    rng = torch.Generator(device=device).manual_seed(args.seed)
    train_pool = pd.DataFrame()
    test_pool = pd.DataFrame()
    
    time_cobalt_total = 0.0

    for label in df_source['label'].unique():
        print(f"--> Processing label: {label}")
        start_time_label = time.time()
        df_label = df_source[df_source['label'] == label].reset_index(drop=True)
        headers = df_label['header'].values
        sequences = df_label['sequence_align'].values
        sequences_enc = encode_sequence(sequences, tokens)
        sequences_enc = torch.tensor(sequences_enc, device=device, dtype=dtype)
        keep_going = True
        num_retries = 0
        while keep_going:
            start_time_cobalt = time.time()
            # NOTE: Cobalt returns the biggest set as training set. For our purpose, we invert train and test sets and apply the filtering on the test set using T3 instead of T2.
            test_headers, _, train_headers,_ = run_cobalt(
                headers=headers,
                X=sequences_enc,
                t1=args.t1,
                t2=args.t2,
                t3=args.t3,
                max_train=None,
                max_test=None,
                rnd_gen=rng,
            )
            time_cobalt_run = time.time() - start_time_cobalt
            time_cobalt_total += time_cobalt_run
            
            if args.min_train is not None and len(train_headers) < args.min_train:
                num_retries += 1
                print(f"----> Not enough training samples ({len(train_headers)}), retrying Cobalt split... (attempt {num_retries})")
                continue
            else:
                keep_going = False
        
        time_label = time.time() - start_time_label
        print(f"----> Train samples: {len(train_headers)}, Test samples: {len(test_headers)}")
        print(f"----> Time for label {label}: {time_label:.2f}s (Cobalt: {time_cobalt_run:.2f}s)")
        train_df = df_label[df_label['header'].isin(train_headers)].reset_index(drop=True)
        test_df = df_label[df_label['header'].isin(test_headers)].reset_index(drop=True)
        test_pool = pd.concat([test_pool, test_df], ignore_index=True)
        train_pool = pd.concat([train_pool, train_df], ignore_index=True)

    train_pool.to_csv(f"{args.output_prefix}.train.csv", index=False)
    test_pool.to_csv(f"{args.output_prefix}.test.csv", index=False)
    
    time_total = time.time() - start_time_total
    
    print("\n" + "="*60)
    print("PROCESSING TIME SUMMARY:")
    print("="*60)
    print(f"Total Cobalt time:    {time_cobalt_total:.2f}s")
    print(f"Total processing time: {time_total:.2f}s")
    print("="*60)
    print(f"\nSaved train set to: {args.output_prefix}.train.csv")
    print(f"Saved test set to: {args.output_prefix}.test.csv")
    print("Done!")
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)