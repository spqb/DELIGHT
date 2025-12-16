import torch
from huggingface_hub import login
from MSA_Pairformer.model import MSAPairformer
from MSA_Pairformer.dataset import MSA, prepare_msa_masks, aa2tok_d
from adabmDCA import write_fasta
import numpy as np
import argparse
import pandas as pd
import os
import time


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encodes aligned sequences using MSA Pairformer and saves them as npz files.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input dataset in .csv format.")
    parser.add_argument("--column_sequences", type=str, default="sequence_align", help="Column name in the input .csv file containing the sequences.")
    parser.add_argument("--column_labels", type=str, default="label", help="Column name in the input .csv file containing the labels.")
    parser.add_argument("--column_headers", type=str, default="header", help="Column name in the input .csv file containing the sequence identifiers.")
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum sequence length for the MSA Pairformer.")
    parser.add_argument("--split_by_label", action="store_true", help="Whether to split the MSA by label.")
    return parser

def csv_to_fasta(csv_file, sequence_column, header_column, label_column):
    """wrapper function that loads a CSV file and converts it to a fasta file"""
    df = pd.read_csv(csv_file)
    headers = df[header_column].to_list()
    sequences = df[sequence_column].to_list()
    labels = df[label_column].to_list() if label_column in df.columns else None
    out_name = csv_file + ".fasta"
    write_fasta(out_name, headers, sequences)
    return out_name, labels

def main(config):
    assert os.path.exists(config["input"]), f"Input file {config['input']} does not exist."
    
    start_time_total = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading input dataset from {config['input']}...")
    
    # This function will allow you to login to huggingface via an API key
    #login()

    # Download model weights and load model
    # As long as the cache doesn't get cleared, you won't need to re-download the weights whenever you re-run this
    print("Loading MSA Pairformer model...")
    start_time_model = time.time()
    model = MSAPairformer.from_pretrained(device=device)

    # You can also save the downloaded weights to a specified directory in your filesystem.
    # Saving the model weights like so will allow you to load the model without re-downloading if your cache gets cleared.
    # Once you run this code once, you can re-run and it will automatically load the weights
    save_model_dir = "../MSA_pairformer_weights"
    model = MSAPairformer.from_pretrained(weights_dir=save_model_dir, device=device)
    time_model_load = time.time() - start_time_model
    print(f"Model loaded in {time_model_load:.2f}s")
    
    msa_file, labels = csv_to_fasta(config["input"], config["column_sequences"], config["column_headers"], config["column_labels"])
    max_msa_depth = 10000
    max_length = 128
    chain_break_idx = 265
    np.random.seed(42)
    
    print("Preparing MSA data...")
    start_time_prep = time.time()
    msa_obj = MSA(
        msa_file_path=msa_file,
        max_seqs=max_msa_depth,
        max_length=max_length,
        max_tokens=np.inf,
        diverse_select_method="none",
        hhfilter_kwargs={"binary": "none"},
    )
    # remove temporary fasta file
    os.remove(msa_file)
    # Prepare MSA and mask tensors
    msa_tokenized_t = msa_obj.diverse_tokenized_msa
    msa_onehot_t = torch.nn.functional.one_hot(msa_tokenized_t, num_classes=len(aa2tok_d)).unsqueeze(0).float().to(device)
    mask, msa_mask, full_mask, pairwise_mask = prepare_msa_masks(msa_obj.diverse_tokenized_msa.unsqueeze(0))
    mask, msa_mask, full_mask, pairwise_mask = mask.to(device), msa_mask.to(device), full_mask.to(device), pairwise_mask.to(device)
    time_prep = time.time() - start_time_prep
    print(f"MSA preparation completed in {time_prep:.2f}s")

    # Run MSA Pairformer to generate embeddings and predict contacts
    print("Generating embeddings...")
    start_time_embed = time.time()
    with torch.no_grad():
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            # batch the input if too large for memory
            batch_size = 1024
            num_sequences = msa_onehot_t.shape[1]
            if num_sequences > batch_size:
                all_final_msa_repr = []
                for start_idx in range(0, num_sequences, batch_size):
                    print(f"Processing sequences {start_idx} to {min(start_idx + batch_size, num_sequences)} / {num_sequences}")
                    end_idx = min(start_idx + batch_size, num_sequences)
                    msa_batch = msa_onehot_t[:, start_idx:end_idx, :, :]
                    msa_mask_batch = msa_mask[:, start_idx:end_idx]
                    full_mask_batch = full_mask[:, start_idx:end_idx, :]
                    res_batch = model(
                        msa=msa_batch.to(torch.bfloat16),
                        mask=mask,
                        msa_mask=msa_mask_batch,
                        full_mask=full_mask_batch,
                        pairwise_mask=pairwise_mask,
                        complex_chain_break_indices=[[chain_break_idx]],
                        return_seq_weights=True,
                        return_pairwise_repr_layer_idx=None,
                        return_msa_repr_layer_idx=None,
                        query_only=False,
                    )
                    all_final_msa_repr.append(res_batch['final_msa_repr'].squeeze(0).float().cpu().numpy())
                embeddings = np.concatenate(all_final_msa_repr, axis=0)  # Shape: (num_sequences, sequence_length, embedding_dim)
                # mean pooling over sequence length
                embeddings = np.mean(embeddings, axis=1)  # Shape: (num_sequences, embedding_dim)
            else:
                if not config["split_by_label"]:
                    print(f"Processing all {num_sequences} sequences at once")
                    res = model(
                        msa=msa_onehot_t.to(torch.bfloat16),
                        mask=mask,
                        msa_mask=msa_mask,
                        full_mask=full_mask,
                        pairwise_mask=pairwise_mask,
                        complex_chain_break_indices=[[chain_break_idx]],
                        return_seq_weights=True,
                        return_pairwise_repr_layer_idx=None,
                        return_msa_repr_layer_idx=True,
                        query_only=False,
                    )
                    # res is a dictionary with the following keys: final_msa_repr, final_pairwise_repr, msa_repr_d, pairwise_repr_d, seq_weights_list_d, predicted_cb_contacts, predicted_confind_contacts
                    embeddings = res['final_msa_repr'].squeeze(0).float().cpu().numpy()  # Shape: (num_sequences, sequence_length, embedding_dim)
                    # mean pooling over sequence length
                    embeddings = np.mean(embeddings, axis=1)  # Shape: (num_sequences, embedding_dim)
                else:
                    # process each label separately
                    all_final_msa_repr = []
                    unique_labels = list(set(labels))
                    for label in unique_labels:
                        print(f"Processing sequences with label {label}")
                        label_indices = [i for i, l in enumerate(labels) if l == label]
                        msa_batch = msa_onehot_t[:, label_indices, :, :]
                        msa_mask_batch = msa_mask[:, label_indices]
                        full_mask_batch = full_mask[:, label_indices, :]
                        res = model(
                            msa=msa_batch.to(torch.bfloat16),
                            mask=mask,
                            msa_mask=msa_mask_batch,
                            full_mask=full_mask_batch,
                            pairwise_mask=pairwise_mask,
                            complex_chain_break_indices=[[chain_break_idx]],
                            return_seq_weights=True,
                            return_pairwise_repr_layer_idx=None,
                            return_msa_repr_layer_idx=True,
                            query_only=False,
                        )
                        all_final_msa_repr.append(res['final_msa_repr'].squeeze(0).float().cpu().numpy())
                    embeddings = np.concatenate(all_final_msa_repr, axis=0)  # Shape: (num_sequences, sequence_length, embedding_dim)
                    # mean pooling over sequence length
                    embeddings = np.mean(embeddings, axis=1)  # Shape: (num_sequences, embedding_dim)
    
    time_embed = time.time() - start_time_embed
    print(f"Embedding generation completed in {time_embed:.2f}s")
    
    # Prepare output filename
    output_prefix = os.path.splitext(config["input"])[0]
    output_path = f"{output_prefix}.msa_pairformer.npz"
    
    # Save to npz file
    print(f"Saving MSA Pairformer encoding to {output_path}...")
    if labels is not None:
        np.savez_compressed(
            output_path,
            embeddings=embeddings,
            labels=np.array(labels),
            headers=msa_obj.ids_l
        )
    else:
        np.savez_compressed(
            output_path,
            embeddings=embeddings,
            headers=msa_obj.ids_l
        )
    
    print(f"Successfully saved MSA Pairformer encoding to {output_path}")
    
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