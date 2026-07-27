import os
import logging
from math import inf
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm
from tf_idf_baseline.tf_idf_model import TfidfModel

logger = logging.getLogger(__name__)

class TfidfTrainer:
    def __init__(self, model: TfidfModel):
        self.model = model
        self.config = model.config
        self.device = torch.device(model.config.device)
        self.model.to(self.device)
        self.criterion = model.config.get_loss_criterion()

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        labels = batch["labels"].to(self.device)
        if self.config.problem_type == "multi_label":
            labels = labels.float()
        else:
            labels = labels.long()

        return {
            "input_ids": batch["input_ids"].to(self.device).float(),
            "labels": labels,
        }

    def train_epoch(self, dataloader: DataLoader, optimizer: AdamW, scheduler: Any) -> float:
        if len(dataloader) == 0:
            raise ValueError("Cannot train with an empty dataloader")

        self.model.train()
        total_loss = 0.0
        processed_batches = 0

        for batch in tqdm(dataloader, desc="TF-IDF Training Iteration"):
            optimizer.zero_grad(set_to_none=True)
            prepared = self._prepare_batch(batch)
            labels = prepared["labels"]

            logits = self.model(prepared["input_ids"])
            loss = self.criterion(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())
            processed_batches += 1

        return total_loss / processed_batches

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        if len(dataloader) == 0:
            raise ValueError("Cannot evaluate with an empty dataloader")

        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        processed_batches = 0

        for batch in tqdm(dataloader, desc="TF-IDF Evaluation Iteration"):
            prepared = self._prepare_batch(batch)
            labels = prepared["labels"]

            logits = self.model(prepared["input_ids"])
            loss = self.criterion(logits, labels)

            total_loss += float(loss.item())

            if self.config.problem_type == "multi_label":
                preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
            else:
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())
            processed_batches += 1

        avg_loss = total_loss / processed_batches
        macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
        micro_f1 = float(f1_score(all_labels, all_preds, average="micro", zero_division=0))

        return {"loss": float(avg_loss), "macro_f1": macro_f1, "micro_f1": micro_f1}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> str | None:
        if self.config.epochs == 0:
            logger.info("Configured epochs=0; skipping training.")
            return None

        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        best_macro_f1 = -inf
        best_checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            metrics = self.evaluate(val_loader)

            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {metrics['loss']:.4f} | Val Macro-F1: {metrics['macro_f1']:.4f}"
            )

            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                torch.save(self.model.state_dict(), best_checkpoint_path)
                logger.info(f"Saved best checkpoint to {best_checkpoint_path} with Macro-F1: {best_macro_f1:.4f}")

        return best_checkpoint_path