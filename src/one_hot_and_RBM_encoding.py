import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import torch

from adabmDCA.fasta import get_tokens, encode_sequence
from rbms.utils import get_saved_updates
from rbms.io import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Define the datasets and parameters
tokens = get_tokens("protein")
datasets_list = ["RR", "SH3", "Globin", "CM"]
num_train_samples_dict = {
    "RR": [100, 500, 1000, 2000],
    "SH3": [100, 500, 1000,],
    "Globin": [1393,],
    "CM": [529,],
}

# Logistic Regression Classifier on one-hot encoded sequences
print("Running Logistic Regression Classifier on one-hot encoded sequences...")
for dataset in datasets_list:
    print(f"Processing dataset: {dataset}")
    t1_values = [0.4, 0.7] if dataset == "RR" else [0.7,]
    for t1 in t1_values:
        print(f"--> Using t1: {t1}")
        dirname = f"./experiments/datasets/{dataset}/t1{t1}_t20.7"
        
        # import test data
        test_df = pd.read_csv(os.path.join(dirname, "test.csv"))
        X_test = test_df["sequence_align"].values
        y_test = test_df["label"].values
        # one-hot encode sequences
        X_test = encode_sequence(X_test, tokens)
        X_test = np.eye(len(tokens))[X_test].reshape(len(X_test), -1)
        
        for num_train_samples in num_train_samples_dict[dataset]:
            print(f"---->Training with {num_train_samples} samples...")
            # import train data
            train_df = pd.read_csv(os.path.join(dirname, f"train_{num_train_samples}.csv"))
            X_train = train_df["sequence_align"].values
            y_train = train_df["label"].values
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
                os.path.join(dirname, f"test.msa_{num_train_samples}.npz"),
                labels=y_pred,
                probs=y_pred_proba,
                headers=test_df["header"].values,
                embeddings=X_test,
            )
            
# Logistic Regression Classifier on RBM encoded sequences
print("\n--- Logistic Regression Classifier on RBM encoded sequences ---\n")
for dataset in datasets_list:
    print(f"Processing dataset: {dataset}")
    t1_values = [0.4, 0.7] if dataset == "RR" else [0.7,]
    for t1 in t1_values:
        print(f"--> Using t1: {t1}")
        dirname = f"./experiments/datasets/{dataset}/t1{t1}_t20.7"
        
        # import model
        file_model = f"./experiments/models/{dataset}/rbm/embeddingRBM.h5"
        saved_updates = get_saved_updates(filename=file_model)
        params, *_ = load_model(filename=file_model, index=saved_updates[-1], device=device, dtype=dtype)
        
        # import test data
        test_df = pd.read_csv(os.path.join(dirname, "test.csv"))
        X_test = test_df["sequence_align"].values
        y_test = test_df["label"].values
        X_test = encode_sequence(X_test, tokens)
        X_test = torch.tensor(X_test).to(dtype=dtype, device=device)
        # encode test data
        X_test_input = {"visible": X_test}
        H_test = params.sample_hiddens(X_test_input)["hidden_mag"].cpu().numpy()
        
        for num_train_samples in num_train_samples_dict[dataset]:
            print(f"---->Training with {num_train_samples} samples...")
            # import train data
            train_df = pd.read_csv(os.path.join(dirname, f"train_{num_train_samples}.csv"))
            X_train = train_df["sequence_align"].values
            y_train = train_df["label"].values
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
                os.path.join(dirname, f"test.rbm_{num_train_samples}.npz"),
                labels=y_pred,
                probs=y_pred_proba,
                headers=test_df["header"].values,
                embeddings=H_test,
            )
