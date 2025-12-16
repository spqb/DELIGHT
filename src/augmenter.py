import torch
import numpy as np
import pandas as pd
import adabmDCA as abm
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Data Augmentation using bmDCA")
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the input CSV file containing sequences for augmentation.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to the output CSV file to save augmented sequences.",
    )
    parser.add_argument(
        "--DCA_params",
        type=str,
        required=True,
        help="Path to DCA model parameters file.",
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=100,
        help="Number of augmented sequences to generate for each wild type sequence.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Number of MCMC steps for sequence generation.",
    )
    parser.add_argument(
        "--column_name",
        type=str,
        default="header",
        help="Column name in the CSV file that contains the sequence header.",
    )
    parser.add_argument(
        "--column_sequence",
        type=str,
        default="sequence",
        help="Column name in the CSV file that contains the sequences.",
    )
    parser.add_argument(
        "--column_label",
        type=str,
        default="label",
        help="Column name in the CSV file that contains the sequence labels.",
    )
    parser.add_argument(
        "--alphabet",
        type=str,
        default="protein",
        help="Alphabet type for the sequences (e.g., 'protein' 'dna' or 'rna').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the DCA model on (e.g., 'cpu' or 'cuda').",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Data type for model computations (e.g., 'float32' or 'float64').",
    )
    

    return parser


def main(args):
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    print(f"Using device: {device}, dtype: {dtype}")
    
    print( "Loading input CSV file..." )
    df = pd.read_csv(args.input_csv)
    assert args.column_name in df.columns and args.column_sequence in df.columns and args.column_label in df.columns, f"The input .csv file must contain '{args.column_name}', '{args.column_sequence}' and '{args.column_label}' columns."
    
    tokens = abm.get_tokens(args.alphabet)
    print(f"Loading DCA model parameters from {args.DCA_params}...")
    params = abm.load_params(args.DCA_params, tokens=tokens, dtype=dtype, device=device)
    
    print("Generating augmented sequences...")
    df_pool = pd.DataFrame(columns=[args.column_name, args.column_sequence, args.column_label])
    for idx, row in df.iterrows():
        print(f"Processing sequence {idx+1}/{len(df)}: {row[args.column_name]}")
        sequence_enc = abm.encode_sequence(row[args.column_sequence], tokens)
        sequence_torch = torch.tensor(sequence_enc, dtype=torch.long, device=device).unsqueeze(0)
        sequence_torch_oh = torch.nn.functional.one_hot(sequence_torch, num_classes=len(tokens)).float()
        pool = sequence_torch_oh.repeat(args.num_sequences, 1, 1)
        for step in range(args.num_steps):
            pool = abm.gibbs_step_independent_sites(pool, params)
        pool_dec = abm.decode_sequence(pool, tokens)
        pool_headers = [f"{row[args.column_name]}_aug_{i+1}" for i in range(args.num_sequences)]
        pool_labels = [row[args.column_label]] * args.num_sequences
        # Include the original sequence
        pool_dec = np.append(pool_dec, [row[args.column_sequence]], axis=0)
        pool_headers.append(row[args.column_name])
        pool_labels.append(row[args.column_label])
        df_temp = pd.DataFrame({
            args.column_name: pool_headers,
            args.column_sequence: pool_dec,
            args.column_label: pool_labels
        })
        df_pool = pd.concat([df_pool, df_temp], ignore_index=True)
    print(f"Saving augmented sequences to {args.output_csv}...")
    df_pool.to_csv(args.output_csv, index=False)
    print("Data augmentation completed.")
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)