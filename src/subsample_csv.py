#!/usr/bin/env python3

# Takes a csv file in input and subsamples it creating several smaller scv files
# The subsampled files are saved as <output_prefix>_<num_samples>.csv
# Each label is represented equally in the subsampled files. If not possible, a warning is printed.
# Test sequences are selected so to have the same number of samples per label, which is dictated by the smallest class.

import pandas as pd
import os
import random
import argparse
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description="Subsample CSV dataset")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to the input train CSV file")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to the input test CSV file")
    parser.add_argument("--output_prefix", type=str, required=True, help="Prefix for the output CSV files. The files will be saved as <output_prefix>_<num_samples>.csv")
    parser.add_argument("--num_samples_list", type=int, nargs='+', required=True, help="List of numbers of samples for subsampling")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subsampling")
    return parser

def main(args):
    df_train = pd.read_csv(args.train_csv)
    df_test = pd.read_csv(args.test_csv)
    
    # create output directory if it does not exist
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    labels = np.unique(df_train['label'])
    # set random seed
    random.seed(args.seed)
    for num_samples in args.num_samples_list:
        random_state = random.randint(0, int(1e6))
        samples_per_label = num_samples // len(labels)
        df_subsampled_list = []
        for label in labels:
            df_label = df_train[df_train['label'] == label]
            num_samples_label = len(df_label)
            if num_samples_label < samples_per_label:
                print(f"Warning: Not enough samples for label {label}. Requested {samples_per_label}, but only {num_samples_label} available.")
                num_extract = num_samples_label
            else:
                num_extract = samples_per_label
            df_label_subsampled = df_label.sample(n=num_extract, random_state=random_state, replace=False).reset_index(drop=True)
            df_subsampled_list.append(df_label_subsampled)
        df_subsampled = pd.concat(df_subsampled_list).reset_index(drop=True)
        output_path = f"{args.output_prefix}.train_{num_samples}.csv"
        df_subsampled.to_csv(output_path, index=False)
        print(f"Saved subsampled CSV with {num_samples} samples to: {output_path}")
        
    # Now subsample the test set to have the same number of samples per label
    labels, counts = np.unique(df_test['label'], return_counts=True)
    min_count_test = min(counts)
    print(f"Subsampling test set to have {min_count_test} samples per label.")
    df_test_subsampled_list = []
    for label in labels:
        df_label = df_test[df_test['label'] == label]
        df_label_subsampled = df_label.sample(n=min_count_test, random_state=random_state, replace=False).reset_index(drop=True)
        df_test_subsampled_list.append(df_label_subsampled)
    df_test_subsampled = pd.concat(df_test_subsampled_list).reset_index(drop=True)
    output_test_path = f"{args.output_prefix}.test.csv"
    df_test_subsampled.to_csv(output_test_path, index=False)
    print(f"Saved subsampled test CSV to: {output_test_path}")
        
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    