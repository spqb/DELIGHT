import numpy as np
import pandas as pd
import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Generate datasets to inform the label-aware RBM.")
    parser.add_argument("--num_training_samples", type=int, help="Number of training samples in the dataset.")
    parser.add_argument("--t1", type=float, help="Parameter t1 for the input data")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to be used")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    output_dir = f"./experiments/datasets/{args.dataset_name}/input-label_rbm"
    input_dir = f"./experiments/datasets/{args.dataset_name}/t1{args.t1}_t20.7"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df_train = pd.read_csv(os.path.join(input_dir, f"train_{args.num_training_samples}.csv"))
    df_test = pd.read_csv(os.path.join(input_dir, "test.csv"))
    predictions_msa_npz = np.load(os.path.join(input_dir, f"test.msa_{args.num_training_samples}.npz"),  allow_pickle=True)
    predictions_contrastive_npz = np.load(os.path.join(input_dir, f"test.embedding_contrastive_{args.num_training_samples}.npz"), allow_pickle=True)
    predictions_foundation_npz = np.load(os.path.join(input_dir, f"test.embedding_foundation_{args.num_training_samples}.npz"), allow_pickle=True)
    df_predictions_msa = pd.DataFrame.from_dict({
        "header": predictions_msa_npz["headers"].tolist(),
        "label": predictions_msa_npz["labels"].tolist(),
    })
    df_predictions_contrastive = pd.DataFrame.from_dict({
        "header": predictions_contrastive_npz["headers"].tolist(),
        "label": predictions_contrastive_npz["labels"].tolist(),
    })
    df_predictions_foundation = pd.DataFrame.from_dict({
        "header": predictions_foundation_npz["headers"].tolist(),
        "label": predictions_foundation_npz["labels"].tolist(),
    })
    
    # ground truth
    df_rbm = pd.concat([df_train, df_test], ignore_index=True)
    df_predictions_msa = pd.concat([df_predictions_msa, df_train[["header", "label"]]], ignore_index=True)
    df_predictions_msa.rename(columns={"label": "label_msa"}, inplace=True)
    df_predictions_contrastive = pd.concat([df_predictions_contrastive, df_train[["header", "label"]]], ignore_index=True)
    df_predictions_contrastive.rename(columns={"label": "label_contrastive"}, inplace=True)
    df_predictions_foundation = pd.concat([df_predictions_foundation, df_train[["header", "label"]]], ignore_index=True)
    df_predictions_foundation.rename(columns={"label": "label_foundation"}, inplace=True)
    df_rbm = df_rbm.merge(df_predictions_msa, on="header", how="left")
    df_rbm = df_rbm.merge(df_predictions_contrastive, on="header", how="left")
    df_rbm = df_rbm.merge(df_predictions_foundation, on="header", how="left")
    df_rbm.rename(columns={"label": "label_true"}, inplace=True)
    # sort columns
    df_rbm = df_rbm[["header", "sequence", "sequence_align", "label_true", "label_msa", "label_foundation", "label_contrastive"]]

    # prepare column having only the training labels
    header_to_all_label = {h: "" for h in df_rbm["header"].tolist()}
    header_to_train_label = {h: l for h, l in zip(df_train["header"].tolist(), df_train["label"].tolist())}
    for h in df_train["header"].tolist():
        header_to_all_label[h] = header_to_train_label[h]
    # add column "label_train" to df
    df_rbm["label_train"] = df_rbm["header"].map(header_to_all_label)
    
    # prepare column with high-probability label predictions
    threshold = 0.8

    predictions_msa_npz = np.load(os.path.join(input_dir, f"test.msa_{args.num_training_samples}.npz"),  allow_pickle=True)
    predictions_contrastive_npz = np.load(os.path.join(input_dir, f"test.embedding_contrastive_{args.num_training_samples}.npz"), allow_pickle=True)
    predictions_foundation_npz = np.load(os.path.join(input_dir, f"test.embedding_foundation_{args.num_training_samples}.npz"), allow_pickle=True)
    
    predictions_msa_mask = np.any(predictions_msa_npz["probs"] >= threshold, axis=1)
    predictions_contrastive_mask = np.any(predictions_contrastive_npz["probs"] >= threshold, axis=1)
    predictions_foundation_mask = np.any(predictions_foundation_npz["probs"] >= threshold, axis=1)
    
    header_to_all_label_msa = {h: "" for h in df_rbm["header"].tolist()}
    header_to_all_label_contrastive = {h: "" for h in df_rbm["header"].tolist()}
    header_to_all_label_foundation = {h: "" for h in df_rbm["header"].tolist()}
    
    header_to_msa_prediction = {h: l for h, l in zip(df_rbm["header"].tolist(), df_rbm["label_msa"].tolist())}
    header_to_contrastive_prediction = {h: l for h, l in zip(df_rbm["header"].tolist(), df_rbm["label_contrastive"].tolist())}
    header_to_foundation_prediction = {h: l for h, l in zip(df_rbm["header"].tolist(), df_rbm["label_foundation"].tolist())}
    
    headers_th_msa = predictions_msa_npz["headers"][predictions_msa_mask]
    header_th_contrastive = predictions_contrastive_npz["headers"][predictions_contrastive_mask]
    headers_th_foundation = predictions_foundation_npz["headers"][predictions_foundation_mask]
    
    for h in headers_th_msa:
        header_to_all_label_msa[h] = header_to_msa_prediction[h]
    for h in header_th_contrastive:
        header_to_all_label_contrastive[h] = header_to_contrastive_prediction[h]
    for h in headers_th_foundation:
        header_to_all_label_foundation[h] = header_to_foundation_prediction[h]
    
    df_rbm["label_msa_0.8"] = df_rbm["header"].map(header_to_all_label_msa)
    df_rbm["label_contrastive_0.8"] = df_rbm["header"].map(header_to_all_label_contrastive)
    df_rbm["label_foundation_0.8"] = df_rbm["header"].map(header_to_all_label_foundation)
    
    df_rbm.to_csv(os.path.join(output_dir, f"predictions_t1{args.t1}.csv"), index=False, na_rep="")
    print(f"Dataset saved to {os.path.join(output_dir, f'predictions_t1{args.t1}.csv')}")
    
if __name__ == "__main__":
    main()