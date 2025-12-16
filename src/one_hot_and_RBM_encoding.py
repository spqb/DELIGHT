import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
import argparse

from adabmDCA.fasta import get_tokens, encode_sequence
from rbms.utils import get_saved_updates
from rbms.io import load_model


def get_parser():
    parser = argparse.ArgumentParser(description="One-hot and RBM encoding with Logistic Regression Classifier")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to the training CSV file")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to the test CSV file")
    parser.add_argument("--output_prefix", type=str, required=True, help="Prefix for the output files")
    parser.add_argument("--rbm_model_path", type=str, required=True, help="Path to the RBM model file")
    return parser

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    tokens = get_tokens("protein")
    
    # Logistic Regression Classifier on one-hot encoded sequences
    print("Running Logistic Regression Classifier on one-hot encoded sequences...")
    test_df = pd.read_csv(args.test_csv)
    X_test = test_df["sequence_align"].to_numpy()
    # one-hot encode sequences
    X_test = encode_sequence(X_test, tokens)
    X_test = np.eye(len(tokens))[X_test].reshape(len(X_test), -1)
    
    train_df = pd.read_csv(args.train_csv)
    X_train = train_df["sequence_align"].to_numpy()
    y_train = train_df["label"].to_numpy()
    # one-hot encode sequences
    X_train = encode_sequence(X_train, tokens)
    X_train = np.eye(len(tokens))[X_train].reshape(len(X_train), -1)
            
    # standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # fit logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # predict on test data
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # save results
    np.savez_compressed(
        f"{args.output_prefix}.msa.npz",
        labels=y_pred,
        probs=y_pred_proba,
        headers=test_df["header"].to_numpy(),
        embeddings=X_test,
    )
    print(f"Saved one-hot encoding results to: {args.output_prefix}.msa.npz")
            
    # Logistic Regression Classifier on RBM encoded sequences
    print("Running Logistic Regression Classifier on RBM encoded sequences...")
    saved_updates = get_saved_updates(filename=args.rbm_model_path)
    params, *_ = load_model(filename=args.rbm_model_path, index=saved_updates[-1], device=device, dtype=dtype)
    
    X_test = test_df["sequence_align"].to_numpy()
    X_test = encode_sequence(X_test, tokens)
    X_test = torch.tensor(X_test).to(dtype=dtype, device=device)
    # encode test data
    X_test_input = {"visible": X_test}
    H_test = params.sample_hiddens(X_test_input)["hidden_mag"].cpu().numpy()

    # import train data
    train_df = pd.read_csv(args.train_csv)
    X_train = train_df["sequence_align"].to_numpy()
    y_train = train_df["label"].to_numpy()
    X_train = encode_sequence(X_train, tokens)
    X_train = torch.tensor(X_train).to(dtype=dtype, device=device)
    # encode train data
    X_train_input = {"visible": X_train}
    H_train = params.sample_hiddens(X_train_input)["hidden_mag"].cpu().numpy()
    
    # standardize the data
    scaler = StandardScaler()
    H_train = scaler.fit_transform(H_train)
    H_test_scaled = scaler.transform(H_test)
    
    # fit logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(H_train, y_train)
    
    # predict on test data
    y_pred = model.predict(H_test_scaled)
    y_pred_proba = model.predict_proba(H_test_scaled)
    
    # save results
    np.savez_compressed(
        f"{args.output_prefix}.rbm.npz",
        labels=y_pred,
        probs=y_pred_proba,
        headers=test_df["header"].to_numpy(),
        embeddings=H_test,
    )
    print(f"Saved RBM encoding results to: {args.output_prefix}.rbm.npz")
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
