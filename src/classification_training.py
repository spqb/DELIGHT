import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Contrastive training of a model on a dataset")
    parser.add_argument("--model", type=str, default="Rostlab/prot_bert", help="Model name.", choices=["Rostlab/prot_bert", "facebook/esm2_t12_35M_UR50D", "facebook/esm2_t6_8M_UR50D"])
    parser.add_argument("--train", type=str, required=True, help="Train dataset path.")
    parser.add_argument("--query", type=str, default=None, help="Dataset with the sequences to be annotated. Can be a fasta file or a .csv file.")
    parser.add_argument("--output", type=str, required=True, help="Output directory where to save the model's parameters.")
    parser.add_argument("--flag", type=str, default="embedding", help="Flag to add to the embedding file name.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=1, help="Maxium number of epochs.")
    parser.add_argument("--save_steps", type=int, default=20, help="Save the model every N steps.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay.")
    #parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank of the decomposition.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Scaling factor.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability.")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 precision.")
    
    return parser


def main(config):
    
    if not os.path.exists(config["output"]):
        os.makedirs(config["output"])
        
    print("Loading dataset...")
    train_dataset = load_dataset("csv", data_files=config["train"], split="train")
    tokenizer = AutoTokenizer.from_pretrained(config["model"], do_lower_case=False)
    spaced_tokens = True if "prot_bert" in config["model"] else False

    # Import train dataset
    if spaced_tokens:
        def tokenize_function(entry):
            sequence = " ".join(entry["sequence"])
            return tokenizer(sequence, padding="max_length", truncation=True, max_length=config["max_length"])
    else:
        def tokenize_function(entry):
            return tokenizer(entry["sequence"], padding="max_length", truncation=True, max_length=config["max_length"])
    train_dataset = train_dataset.map(tokenize_function)
        
    # Set labels to be integers
    unique_labels = train_dataset.unique("label")
    label_map = {label : i for i, label in enumerate(unique_labels)}
    def set_labels(entry):
        entry["label"] = label_map[entry["label"]]
        return entry
    train_dataset = train_dataset.map(set_labels)
    
    # Import model
    model = AutoModelForSequenceClassification.from_pretrained(config["model"], num_labels=len(unique_labels))
    # Import Lora adapters
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=config["output"],
        num_train_epochs=int(config["epochs"]),
        per_device_train_batch_size=int(config["batch_size"]),
        eval_strategy="no",
        save_strategy="steps",
        logging_dir=config["output"],
        logging_steps=int(config["save_steps"]),
        label_names=["label"],
        learning_rate=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        bf16=config["bf16"],
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        compute_metrics=None,
    )
    trainer.train()
    
    # predict on train dataset
    print("Predicting on train dataset...")
    trainer.model.eval()
    train_preds = trainer.predict(train_dataset)
    train_probs = train_preds.predictions[1]
    train_labels = np.argmax(train_probs, axis=1)
    
    # Predict on query dataset
    if config["query"] is not None:
        assert os.path.exists(config["query"]), f"Query dataset {config['query']} does not exist."
        print("Embedding the query dataset...")
        query_dataset = load_dataset("csv", data_files=config["query"], split="train")
        query_dataset = query_dataset.map(tokenize_function)
        query_dataset = query_dataset.map(set_labels)
        
        # Predict
        trainer.model.eval()
        query_preds = trainer.predict(query_dataset)
        query_logits = query_preds.predictions[1]
        query_labels_num = np.argmax(query_logits, axis=1)
        # converto to string
        query_labels = [unique_labels[i] for i in query_labels_num]       
        query_probs = np.exp(query_logits) / np.sum(np.exp(query_logits), axis=1, keepdims=True)
        
        # Save embeddings
        fname_query = os.path.splitext(config["query"])[0] + ".{}.npz".format(config["flag"])
        headers_query = query_dataset["header"] if "header" in query_dataset.column_names else None
        sequences_query = query_dataset["sequence"] if "sequence" in query_dataset.column_names else None
        np.savez_compressed(fname_query, embeddings=sequences_query, labels=query_labels, headers=headers_query, probs=query_probs)
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = vars(args)
    
    main(config)
        
    
        