import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import argparse
import os
import time
import h5py


def get_parser():
    parser = argparse.ArgumentParser(description="Predict from embeddings using multiple classifiers (Logistic Regression, SVM, Random Forest)")
    parser.add_argument("--train_npz", type=str, required=True, help="Path to the training NPZ file")
    parser.add_argument("--test_npz", type=str, required=True, help="Path to the test NPZ file")
    parser.add_argument("--flag", type=str, default="", help="Optional flag")
    return parser

def main(args):
    print("Loading data...")
    train = np.load(args.train_npz, allow_pickle=True)
    test = np.load(args.test_npz, allow_pickle=True)
    X_train = train["embeddings"]
    y_train = train["labels"]
    X_test = test["embeddings"]
    h_test = test["headers"]
    y_test = test["labels"] if "labels" in test else None
    
    print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, Test headers shape: {h_test.shape}")
    
    # standardize the data
    print("Standardizing data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Dictionary to store predictions
    predictions = {}
    
    # 1. Logistic Regression
    print("\nTraining Logistic Regression...")
    start_time = time.time()
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred_logreg = logreg.predict(X_test_scaled)
    y_proba_logreg = logreg.predict_proba(X_test_scaled)
    time_logreg = time.time() - start_time
    predictions['logreg'] = {
        'labels_pred': y_pred_logreg,
        'labels_probs': y_proba_logreg
    }
    if y_test is not None:
        acc_logreg = accuracy_score(y_test, y_pred_logreg)
        print(f"Logistic Regression completed - Accuracy: {acc_logreg:.4f} - Time: {time_logreg:.2f}s")
    else:
        print(f"Logistic Regression completed - Time: {time_logreg:.2f}s")
    
    # # 2. SVM
    # print("\nTraining SVM...")
    # start_time = time.time()
    # svm = SVC(kernel='linear', probability=True, random_state=42)
    # svm.fit(X_train, y_train)
    # y_pred_svm = svm.predict(X_test_scaled)
    # y_proba_svm = svm.predict_proba(X_test_scaled)
    # time_svm = time.time() - start_time
    # predictions['SVM'] = {
    #     'labels_pred': y_pred_svm,
    #     'labels_probs': y_proba_svm
    # }
    # if y_test is not None:
    #     acc_svm = accuracy_score(y_test, y_pred_svm)
    #     print(f"SVM completed - Accuracy: {acc_svm:.4f} - Time: {time_svm:.2f}s")
    # else:
    #     print(f"SVM completed - Time: {time_svm:.2f}s")
    
    # # 3. Random Forest
    # print("\nTraining Random Forest...")
    # start_time = time.time()
    # rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf.fit(X_train, y_train)
    # y_pred_rf = rf.predict(X_test_scaled)
    # y_proba_rf = rf.predict_proba(X_test_scaled)
    # time_rf = time.time() - start_time
    # predictions['random_forest'] = {
    #     'labels_pred': y_pred_rf,
    #     'labels_probs': y_proba_rf
    # }
    # if y_test is not None:
    #     acc_rf = accuracy_score(y_test, y_pred_rf)
    #     print(f"Random Forest completed - Accuracy: {acc_rf:.4f} - Time: {time_rf:.2f}s")
    # else:
    #     print(f"Random Forest completed - Time: {time_rf:.2f}s")
    
    # Prepare output filename
    output_prefix = os.path.splitext(args.test_npz)[0]
    if args.flag != "":
        output_path = f"{output_prefix}.{args.flag}.predictions.h5"
    else:
        output_path = f"{output_prefix}.predictions.h5"
    
    # Save results to HDF5
    print(f"\nSaving predictions to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        # Save top-level arrays
        f.create_dataset('headers', data=h_test.astype('S'))  # Convert strings to bytes for HDF5
        f.create_dataset('embeddings', data=X_test)
        if y_test is not None:
            f.create_dataset('labels_true', data=y_test.astype('S') if y_test.dtype.kind in ('U', 'O') else y_test)
        
        # Create groups for each classifier
        for classifier_name, preds in predictions.items():
            grp = f.create_group(f'predictions/{classifier_name}')
            labels_pred = preds['labels_pred']
            # Convert string labels to bytes if necessary
            if labels_pred.dtype.kind in ('U', 'O'):
                labels_pred = labels_pred.astype('S')
            grp.create_dataset('labels_pred', data=labels_pred)
            grp.create_dataset('labels_probs', data=preds['labels_probs'])
    
    print(f"Successfully saved predictions to {output_path}")
    print("\nResults structure (HDF5):")
    print("  - headers: test sequence headers (stored as bytes)")
    print("  - embeddings: test embeddings")
    print("  - labels_true: true labels (if available)")
    print("  - predictions/")
    print("      - logreg/labels_pred, logreg/labels_probs")
    print("      - SVM/labels_pred, SVM/labels_probs")
    print("      - random_forest/labels_pred, random_forest/labels_probs")
    
    if y_test is not None:
        print("\n" + "="*60)
        print("SUMMARY:")
        print("="*60)
        print(f"{'Classifier':<20} {'Accuracy':<12} {'Time (s)':<12}")
        print("-"*60)
        print(f"{'Logistic Regression':<20} {acc_logreg:<12.4f} {time_logreg:<12.2f}")
        # print(f"{'SVM':<20} {acc_svm:<12.4f} {time_svm:<12.2f}")
        # print(f"{'Random Forest':<20} {acc_rf:<12.4f} {time_rf:<12.2f}")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("PROCESSING TIME SUMMARY:")
        print("="*60)
        print(f"{'Classifier':<20} {'Time (s)':<12}")
        print("-"*60)
        print(f"{'Logistic Regression':<20} {time_logreg:<12.2f}")
        # print(f"{'SVM':<20} {time_svm:<12.2f}")
        # print(f"{'Random Forest':<20} {time_rf:<12.2f}")
        print("="*60)
    
    print("Done!")
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)