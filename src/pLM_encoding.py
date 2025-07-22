import argparse
import torch
import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from train.dataloaders.dataloader import PairDataset, PairwiseInputCollator
from train.models.contrastive import ContrastiveLM
from train.trainer import Trainer
from train.utils.contrastive_utils import compute_embeddings
import warnings
from Bio import SeqIO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encodes sequences using a pre-trained language model and optionally fine-tunes it.")
    parser.add_argument("--model", type=str, default="Rostlab/prot_bert", help="Model name.", choices=["Rostlab/prot_bert", "facebook/esm2_t12_35M_UR50D", "facebook/esm2_t6_8M_UR50D"])
    parser.add_argument("--train", type=str, required=True, help="Train dataset path.")
    parser.add_argument("--query", type=str, default=None, help="Dataset with the sequences to be annotated. Can be a fasta file or a .csv file.")
    parser.add_argument("--output", type=str, default=None, help="Output directory where to save the model's parameters.")
    parser.add_argument("--column_sequences", type=str, default="sequence", help="Column name in the input .csv file containing the sequences.")
    parser.add_argument("--column_labels", type=str, default="label", help="Column name in the input .csv file containing the labels.")
    parser.add_argument("--column_headers", type=str, default="header", help="Column name in the input .csv file containing the sequence identifiers.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load the model from.")
    parser.add_argument("--flag", type=str, default="embedding", help="Flag to add to the embedding file name.")
    parser.add_argument("--zero-shot", action="store_true", help="Embed query file without fine-tuning the model.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=1, help="Maxium number of epochs.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save the model every N steps.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--feat_dim", type=int, default=128, help="Feature dimension for the contrastive heads.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank of the decomposition.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Scaling factor.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability.")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 precision.")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging.")
    
    return parser


def main(config):
    
    if config["zero_shot"] and config["query"] is None:
        raise ValueError("If --zero-shot is set, --query must be provided. Nothing to embed.")
    
    if config["output"] is None and config["checkpoint"] is None and not config["zero_shot"]:
        raise ValueError("When --zero-shot is not set, one between --output or --checkpoint must be provided.")
    
    if config["checkpoint"] is not None and config["output"] is not None:
        warnings.warn("Both --checkpoint and --output are set. The model will be loaded from the checkpoint and the output directory will be ignored.")
    
    if config["output"] is not None:
        if not os.path.exists(config["output"]):
            os.makedirs(config["output"])
    
    if config["checkpoint"] is not None:
        assert os.path.exists(config["checkpoint"]), f"Checkpoint {config['checkpoint']} does not exist."
        
    assert os.path.exists(config["train"]), f"Training dataset {config['train']} does not exist."
        
    device = torch.device("cuda" if torch.cuda.is_available() else ValueError("No GPU available"))
    print("Loading dataset...")
    train_dataset = PairDataset(config["train"], column_sequences=config["column_sequences"], column_labels=config["column_labels"])
    print(f"Constructed {len(train_dataset)} positive pairs from the input dataset")
    tokenizer = AutoTokenizer.from_pretrained(config["model"], do_lower_case=False)
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=int(config["lora_rank"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        bias="none",
        target_modules=["query", "value"],
    )
    
    print("Loading model...")
    model = ContrastiveLM(feat_dim=config["feat_dim"], backbone=config["model"])
    model = model.to(device)
    spaced_tokens = True if "prot_bert" in config["model"] else False
    
    if not config["zero_shot"]:
        model.backbone = get_peft_model(model.backbone, lora_config)
        model.backbone.print_trainable_parameters()
        model.train()
        collator_fn = PairwiseInputCollator(tokenizer, max_length=int(config["max_length"]), insert_whitespace=spaced_tokens)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            output_dir=config["output"],
            optimizer=optimizer,
            collator_fn=collator_fn,
            num_train_epochs=int(config["epochs"]),
            batch_size=int(config["batch_size"]),
            learning_rate=float(config["lr"]),
            patience=int(config["patience"]),
            save_steps=int(config["save_steps"]),
            gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
            bf16=bool(config["bf16"]),
            wandb=bool(config["wandb"]),
        )
        trainer.train()
        model = trainer.model
        
    
    if config["query"] is not None:
        assert os.path.exists(config["query"]), f"Query dataset {config['query']} does not exist."
        model.eval()
        
        if config["train"] is not None:
            assert os.path.exists(config["train"]), f"Training dataset {config['train']} does not exist."
            print("Embedding the training dataset...")
            df = pd.read_csv(config["train"])
            assert config["column_sequences"] in df.columns and config["column_labels"] in df.columns and config["column_headers"] in df.columns, "The input .csv file must contain {} and {} columns.".format(config["column_sequences"], config["column_labels"])
            seq_train = df[config["column_sequences"]].values
            y_train = df[config["column_labels"]].values
            headers_train = df[config["column_headers"]].values
            if spaced_tokens:
                seq_train = list(map(lambda x: " ".join(x), seq_train))
            X_train = compute_embeddings(
                model,
                seq_train,
                tokenizer,
                batch_size=config["batch_size"],
                max_length=config["max_length"],
            ).numpy()
            fname_train = os.path.splitext(config["train"])[0] + ".{}.npz".format(config["flag"])
            np.savez_compressed(fname_train, embeddings=X_train, labels=y_train, headers=headers_train)
            print(f"Training dataset embedding saved to {fname_train}")
        
        print("Loading the query dataset...")
        # check if the query dataset is a fasta file or a .csv file
        with open(config["query"], "r") as f:
            first_line = f.readline()
        if first_line.startswith(">"):
            # fasta file
            records = list(SeqIO.parse(config["query"], "fasta"))
            seq_query = [str(record.seq) for record in records]
            headers_query = [record.id for record in records]
        else: # assuming it's a .csv file
            try:
                df = pd.read_csv(config["query"])
            except ValueError:
                raise ValueError("The query dataset must be a .csv file with {}, {} and {} columns.".format(config["column_labels"], config["column_sequences"], config["column_headers"]))
            assert config["column_labels"] in df.columns and config["column_sequences"] in df.columns and config["column_headers"] in df.columns, "The input .csv file must contain {} and {} columns.".format(config["column_labels"], config["column_sequences"])
            seq_query = df[config["column_sequences"]].values
            headers_query = df[config["column_headers"]].values
        if spaced_tokens:
            seq_query = list(map(lambda x: " ".join(x), seq_query))
            
        print("Embedding the query dataset...")
        X_test = compute_embeddings(
            model,
            seq_query,
            tokenizer,
            batch_size=config["batch_size"],
            max_length=config["max_length"],
        ).numpy()
        
        fname_query = os.path.splitext(config["query"])[0] + ".{}.npz".format(config["flag"])
        if config["train"] is not None:
            print("Training a logistc regression model on the training dataset's embedding...")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)
            print("Predicting the query dataset's embedding...")
            X_test_scaled = scaler.transform(X_test)
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)
            
            np.savez_compressed(
                fname_query,
                embeddings=X_test,
                labels=y_pred,
                headers=headers_query,
                probs=y_prob,
            )
            # create a DataFrame with the predicted labels and save it
            df_query = pd.DataFrame({
                "header": headers_query,
                "sequence": seq_query,
                "predicted_label": y_pred,
            })
            df_query.to_csv(os.path.splitext(fname_query)[0] + "_predicted_labels.csv", index=False)
            print("Query dataset's embedding and predicted labels saved to {}".format(fname_query))
            print(f"Query dataset's embedding and predicted labels saved to {fname_query}")
        else:
            print("Saving the query dataset's embedding...")
            np.savez_compressed(
                fname_query,
                embeddings=X_test,
                headers=headers_query,
            )
            print(f"Query dataset's embedding saved to {fname_query}")
    print("Done!")
            
          
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = vars(args)
    main(config)