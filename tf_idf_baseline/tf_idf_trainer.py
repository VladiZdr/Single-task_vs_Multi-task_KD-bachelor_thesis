from __future__ import annotations

import logging
import os
from math import inf
from typing import Any, Dict

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from tf_idf_baseline.tf_idf_model import TfidfModel

logger = logging.getLogger(__name__)


class TfidfTrainer:
    """Trainer for TF-IDF baseline models with AMP and optimized throughput."""

    scaler: GradScaler

    def __init__(self, model: TfidfModel):
        self.model = model
        self.config = model.config
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.criterion = self.config.get_loss_criterion()

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        labels = batch["labels"].to(self.device, non_blocking=True)
        if self.config.problem_type == "multi_label":
            labels = labels.float()
        else:
            labels = labels.long()

        return {
            "input_ids": batch["input_ids"].to(self.device, non_blocking=True).float(),
            "labels": labels,
        }

    def train_epoch(self, dataloader: DataLoader, optimizer: AdamW, scheduler: Any) -> float:
        if len(dataloader) == 0:
            raise ValueError("Cannot train with an empty dataloader")

        self.model.train()
        total_loss = torch.zeros((), device=self.device)
        processed_batches = 0

        use_amp = self.device.type == "cuda"
        if not hasattr(self, "scaler"):
            self.scaler = GradScaler(enabled=use_amp)

        for batch in tqdm(dataloader, desc="TF-IDF Training Iteration"):
            optimizer.zero_grad(set_to_none=True)
            prepared = self._prepare_batch(batch)
            
            input_ids = prepared["input_ids"]
            labels = prepared["labels"]

            assert isinstance(input_ids, torch.Tensor), "input_ids must be a Tensor"
            assert isinstance(labels, torch.Tensor), "labels must be a Tensor"

            # AMP autocast forward pass
            with autocast(enabled=use_amp):
                logits = self.model(input_ids)
                loss = self.criterion(logits, labels)

            # Scaled backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(optimizer)
            self.scaler.update()

            scheduler.step()

            # GPU-side loss tracking
            total_loss += loss.detach()
            processed_batches += 1

        if processed_batches == 0:
            raise ValueError("No training batches were processed")

        return (total_loss / processed_batches).item()

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        if len(dataloader) == 0:
            raise ValueError("Cannot evaluate with an empty dataloader")

        self.model.eval()
        total_loss = torch.zeros((), device=self.device)
        all_preds = []
        all_labels = []
        processed_batches = 0

        use_amp = self.device.type == "cuda"

        for batch in tqdm(dataloader, desc="TF-IDF Evaluation Iteration"):
            prepared = self._prepare_batch(batch)
            input_ids = prepared["input_ids"]
            labels = prepared["labels"]

            assert isinstance(input_ids, torch.Tensor), "input_ids must be a Tensor"
            assert isinstance(labels, torch.Tensor), "labels must be a Tensor"

            with autocast(enabled=use_amp):
                logits = self.model(input_ids)
                loss = self.criterion(logits, labels)

            total_loss += loss.detach()

            if self.config.problem_type == "multi_label":
                preds = (torch.sigmoid(logits) >= 0.5).int()
            else:
                preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            processed_batches += 1

        if processed_batches == 0:
            raise ValueError("No evaluation batches were processed")

        full_preds = torch.cat(all_preds, dim=0).numpy()
        full_labels = torch.cat(all_labels, dim=0).numpy()

        avg_loss = (total_loss / processed_batches).item()
        macro_f1 = float(f1_score(full_labels, full_preds, average="macro", zero_division=0))
        micro_f1 = float(f1_score(full_labels, full_preds, average="micro", zero_division=0))

        return {"loss": avg_loss, "macro_f1": macro_f1, "micro_f1": micro_f1}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> str | None:
        if self.config.epochs == 0:
            logger.info("Configured epochs=0; skipping training.")
            return None

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
        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )

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