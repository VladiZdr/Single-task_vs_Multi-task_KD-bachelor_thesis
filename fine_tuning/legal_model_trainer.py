import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import logging
import os
from math import inf
from typing import Dict, Any
from tqdm import tqdm
from configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


class LegalModelTrainer:
    def __init__(self, model: nn.Module, config: ModelConfig):
        # Stores the model and config, chooses the device, and moves the model there.
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.criterion = config.get_loss_criterion()

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor | None]:
        # Moves the batch to the appropriate device and ensures labels are in the correct format 
        labels = batch["labels"].to(self.device)
        if self.config.problem_type == "multi_label":
            labels = labels.float()
        else:
            labels = labels.long()

        token_type_ids = batch.get("token_type_ids")
        return {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
            "token_type_ids": token_type_ids.to(self.device) if token_type_ids is not None else None,
            "labels": labels,
        }

    def train_epoch(self, dataloader: DataLoader, optimizer: AdamW, scheduler: Any) -> float:
        if len(dataloader) == 0:
            raise ValueError("Cannot train with an empty dataloader")

        # Turns on training behavior, such as dropout
        self.model.train()
        total_loss = 0.0
        current_batch_count = 0

        for batch in tqdm(dataloader, desc="Training Iteration"):
            # If num_of_batches is set to a positive integer, we limit the number of batches processed per epoch.
            if self.config.num_of_batches > 0 and current_batch_count >= self.config.num_of_batches:
                break
            elif self.config.num_of_batches > 0:
                current_batch_count += 1

            # Clears old gradients before computing new ones.
            optimizer.zero_grad(set_to_none=True)

            # Moves the batch to the appropriate device (GPU or CPU)
            prepared = self._prepare_batch(batch)
            labels = prepared["labels"]

            # Forward pass to compute logits and loss
            logits = self.model( prepared["input_ids"], prepared["attention_mask"], prepared["token_type_ids"])
            assert logits.shape[0] == labels.shape[0],        "Batch size mismatch"      # type: ignore
            assert logits.shape[1] == self.config.num_labels, "Logit dimension mismatch"
            loss = self.criterion(logits, labels)

            # Backward pass and optimization step
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            optimizer.step()

            # Updates the learning rate schedule after each batch
            scheduler.step()

            # Track epoch loss
            total_loss += float(loss.item())

        # Mean training loss for the epoch.
        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        if len(dataloader) == 0:
            raise ValueError("Cannot evaluate with an empty dataloader")

        self.model.eval()
        total_loss = 0.0

        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader, desc="Evaluation Iteration"):
            prepared = self._prepare_batch(batch)
            labels = prepared["labels"]
            
            logits = self.model( prepared["input_ids"], prepared["attention_mask"], prepared["token_type_ids"])
            loss = self.criterion(logits, labels)

            total_loss += float(loss.item())

            # Convert logits to predictions
            if self.config.problem_type == "multi_label":
                preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
            else:
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

            # Collect predictions and labels (later passed to F1 scoring.)
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())                                 # type: ignore

        avg_loss = total_loss / len(dataloader)
        macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
        micro_f1 = float(f1_score(all_labels, all_preds, average="micro", zero_division=0))

        return { "loss": float(avg_loss), "macro_f1": macro_f1, "micro_f1": micro_f1 }
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> str | None:
        if self.config.epochs == 0:
            logger.info("Configured epochs=0; skipping training and validation.")
            return None

        # Separate parameters for weight decay (L2 regularization: as small and close to zero as possible) 
        # and no weight decay (biases and LayerNorm weights)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
        
        # Calculate total training steps and warmup steps for the learning rate scheduler
        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        
        # Model is saved whenever validation macro-F1 improves, including the first epoch.
        best_macro_f1 = -inf
        best_checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
        
        # Loop over epochs, training and evaluating the model -> save the best checkpoint based on validation macro-F1 score
        for epoch in range(self.config.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            metrics = self.evaluate(val_loader)
            
            logger.info(
                f"Train Loss: {train_loss:.4f} | Val Loss: {metrics['loss']:.4f} | "
                f"Val Macro-F1: {metrics['macro_f1']:.4f} | Val Micro-F1: {metrics['micro_f1']:.4f}"
            )
            
            # Save the model if macro-F1 improves
            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                torch.save(self.model.state_dict(), best_checkpoint_path)
                logger.info(f"Saved best checkpoint to {best_checkpoint_path} with Macro-F1: {best_macro_f1:.4f}")

        return best_checkpoint_path
