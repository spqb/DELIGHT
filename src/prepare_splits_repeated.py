#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import torch
from adabmDCA.fasta import get_tokens, encode_sequence
from adabmDCA.cobalt import run_cobalt

NUM_SEEDS = 1
T1 = 0.4
T2 = 1.0
T3 = 0.7
NUMS_EXTRACTION_SAMPLES = [2000, 1000, 500, 100]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
tokens = get_tokens("protein")

fname_source_csv = "/home/lorenzo/Documents/DELIGHT/experiments/datasets/RR/RR.csv"
df_source = pd.read_csv(fname_source_csv)

dirname = f"/home/lorenzo/Documents/DELIGHT/experiments/datasets/RR/t1{T1}_t2{T3}_repeated"
os.makedirs(dirname, exist_ok=True)

labels = df_source['label'].unique().tolist()

for seed in range(0, NUM_SEEDS):
    print(f"\nProcessing seed: {seed}")
    rng = torch.Generator(device=device).manual_seed(seed)
    train_pool = pd.DataFrame()
    test_pool = pd.DataFrame()
    for label in labels:
        print(f"--> Processing label: {label}")
        df_label = df_source[df_source['label'] == label].reset_index(drop=True)
        headers = df_label['header'].values
        sequences = df_label['sequence'].values
        sequences_enc = encode_sequence(sequences, tokens)
        sequences_enc = torch.tensor(sequences_enc, device=device, dtype=dtype)
        
        # NOTE: Cobalt returns the biggest set as training set. For our purpose, we invert train and test sets and apply the filtering on the test set using T3 instead of T2.
        test_headers, _, train_headers,_ = run_cobalt(
            headers=headers,
            X=sequences_enc,
            t1=T1,
            t2=T2,
            t3=T3,
            max_train=None,
            max_test=None,
            rnd_gen=rng,
        )
        print(f"----> Train samples: {len(train_headers)}, Test samples: {len(test_headers)}")
        
        train_df = df_label[df_label['header'].isin(train_headers)].reset_index(drop=True)
        test_df = df_label[df_label['header'].isin(test_headers)].reset_index(drop=True)
        test_pool = pd.concat([test_pool, test_df], ignore_index=True)
        train_pool = pd.concat([train_pool, train_df], ignore_index=True)
        
    # extract training subsets
    for num_samples_tot in NUMS_EXTRACTION_SAMPLES:
        train_pool_subset = pd.DataFrame()
        for label in labels:
            num_samples_label = num_samples_tot // len(labels)
            train_pool_label = train_pool[train_pool['label'] == label].reset_index(drop=True)
            if len(train_pool_label) <= num_samples_label:
                train_pool_subset = pd.concat([train_pool_subset, train_pool_label], ignore_index=True)
            else:
                train_pool_label_sampled = train_pool_label.sample(n=num_samples_label, random_state=seed, replace=False).reset_index(drop=True)
                train_pool_subset = pd.concat([train_pool_subset, train_pool_label_sampled], ignore_index=True)
        
        train_pool_subset.to_csv(os.path.join(dirname, f"seed_{seed}.train_{num_samples_tot}.csv"), index=False)
    test_pool.to_csv(os.path.join(dirname, f"seed_{seed}.test.csv"), index=False)
