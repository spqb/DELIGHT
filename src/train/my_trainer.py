import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_scheduler
from torch.utils.data import DataLoader
from train.dataloaders.dataloader import PairwiseInputCollator
from train.utils.contrastive_utils import contrastive_loss
from tqdm.autonotebook import tqdm
try:
    import wandb
    wandb_available = True
except:
    wandb_available = False

# Early stopping callback
class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")

    def step(self, loss: float):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        output_dir: str,
        optimizer: optim.Optimizer,
        collator_fn: PairwiseInputCollator,
        num_train_epochs: int,
        batch_size: int,
        learning_rate: float,
        patience: int,
        save_steps: int,
        gradient_accumulation_steps: int = 1,
        bf16: bool = True,
        wandb: bool = False,
    ):
        self.model = model
        self.collator_fn = collator_fn
        self.optimizer = optimizer
        self.device = next(model.parameters()).device
        self.train_dataset = train_dataset
        self.collator_fn = collator_fn
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.save_steps = save_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.bf16 = bf16
        self.early_stopping_counter = 0
        self.wandb = wandb if wandb_available else False
        
    def compute_loss(self, inputs):
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        feat_1, feat2, _, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = contrastive_loss(feat_1, feat2)
        return loss
    
    def save_model(self, steps):
        # remove old checkpoints directories
        for filename in os.listdir(self.output_dir):
            if filename.startswith("checkpoint-") and os.path.isdir(os.path.join(self.output_dir, filename)):
                # remove the directory content
                for file in os.listdir(os.path.join(self.output_dir, filename)):
                    file_path = os.path.join(self.output_dir, filename, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(os.path.join(self.output_dir, filename))
        # save the model
        self.model.save(os.path.join(self.output_dir, f"checkpoint-{steps}"))
        
    def train(self):
        # setup wandb if available
        if self.wandb:
            wandb.init(
                project="contrastive-training",
                config={
                    "learning_rate": self.learning_rate,
                    "epochs": self.num_train_epochs,
                    "batch_size": self.batch_size,
                    "patience": self.patience,
                    },
                dir=self.output_dir,
                name=os.path.basename(self.output_dir),
            )
            wandb.watch(self.model, log="all")
        self.model.train()
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collator_fn, shuffle=True)
        early_stopping = EarlyStopping(patience=self.patience)
        time_start = time.time()
        scaler = torch.cuda.amp.GradScaler(enabled=self.bf16)  # Enable mixed precision if bf16 is True
        num_training_steps = len(train_dataloader) * self.num_train_epochs
        scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        pbar = tqdm(total=num_training_steps, desc="Fine-tuning the model", leave=False)
        pbar.set_postfix({"Epoch": 0.00, "Loss": float("inf")})
        tot_steps = 0
        num_gradient_updates = 0
        loss_accumulated = 0.0
        best_loss = float("inf")
        for epoch in range(self.num_train_epochs):
            for batch in train_dataloader:
                tot_steps += 1
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=self.bf16):  # Mixed precision context
                    loss = self.compute_loss(inputs)
                loss = loss / self.gradient_accumulation_steps
                loss_accumulated += loss.item()
                if (tot_steps % self.gradient_accumulation_steps == 0):
                    scaler.scale(loss).backward()  # Scale the loss for mixed precision
                    scaler.step(self.optimizer)
                    scaler.update()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    num_gradient_updates += 1
                
                pbar.update(1)
                pbar.set_postfix({"Epoch": (tot_steps) / num_training_steps, "Loss": loss.item()})
                if num_gradient_updates % self.save_steps == 0:
                    loss_avg = loss_accumulated / self.save_steps
                    loss_accumulated = 0.0
                    if loss_avg < best_loss:
                        best_loss = loss_avg
                        self.save_model(tot_steps)
                        pbar.write(f"Saved checkpoint at step {tot_steps}, training loss: {loss_avg:.4f}")
                    else:
                        pbar.write(f"Step {tot_steps}, training loss: {loss_avg:.4f}")
                    if self.wandb:
                        wandb.log(
                            {
                            "epoch": (tot_steps) / len(train_dataloader),
                            "step": tot_steps,
                            "loss": loss_avg,
                            "time": time.time() - time_start,
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                            }
                        )
                    # Check for early stopping
                    if early_stopping.step(loss_avg):
                        pbar.write("Early stopping triggered.")
                        return
    
        pbar.write("Training complete.")
        self.save_model()
        pbar.write(f"Model saved in {self.output_dir}")
        pbar.close()
        if self.wandb:
            wandb.finish()
        return