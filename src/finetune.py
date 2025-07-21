import argparse
from pathlib import Path
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finetune a model on a dataset")
    parser.add_argument("--model", type=str, required=True, help="Model name.", choices=["Rostlab/prot_bert", "facebook/esm2_t12_35M_UR50D", "facebook/esm2_t6_8M_UR50D"])
    parser.add_argument("--label", type=str, default=None, help="Label for the model.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path.")
    parser.add_argument("--output", type=str, required=True, help="Output directory.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--spaced_tokens", action="store_true", help="Add spaces between tokens.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay.")
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank of the decomposition.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Scaling factor.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability.")
    return parser

def main(args):
    model_label = args.model.split("/")[-1] if args.label is None else args.label
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import and preprocess the dataset
    print("Loading dataset...")
    dataset = load_dataset("csv", data_files=args.dataset)
    
    # protBERT requires the amino acids to be separated by spaces
    tokenizer = AutoTokenizer.from_pretrained(args.model, do_lower_case=False)
    
    if args.spaced_tokens:
        def add_spaces(x):
            return {"sequence" : " ".join(list(x["sequence"]))}
        dataset = dataset.map(add_spaces)
        
    # Tokenize the sequences
    def tokenize_function(examples):
        return tokenizer(examples["sequence"], padding="max_length", truncation=True, max_length=args.max_length)
    
    print("Tokenizing sequences...")
    dataset_tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["sequence", "header", "label"])
    
    # Load the model and set the LoRA configuration
    print("Loading model...")
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    lora_config = LoraConfig(
        r=args.lora_rank,  # Rank of the decomposition
        lora_alpha=args.lora_alpha,  # Scaling factor
        lora_dropout=args.lora_dropout,  # Dropout probability
        bias="none",
        target_modules=["query", "value"]  # Apply LoRA to attention layers
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # prepare the trainer
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    training_args = TrainingArguments(
        output_dir=output_dir / model_label,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        logging_steps=1000,
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        save_steps=1000,
        eval_strategy="no",
        bf16=True,
    )
            
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_tokenized["train"],
    )
    
    trainer.train()
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)