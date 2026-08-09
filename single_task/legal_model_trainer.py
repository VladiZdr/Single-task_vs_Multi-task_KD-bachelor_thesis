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
from datasets_manipulation.export_teacher_outputs import SoftTargetExporter
from single_task.legal_model import LegalModel
from torch.cuda.amp import GradScaler, autocast

logger = logging.getLogger(__name__)


class LegalModelTrainer:
    def __init__(self, model: LegalModel):
        self.model = model
        self.config = model.config
        self.device = torch.device(model.config.device)
        self.model.to(self.device)

        self.criterion = model.config.get_loss_criterion()
        self._sync_teacher_weight(epoch_index=0)

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)

    def _sync_teacher_weight(self, epoch_index: int) -> None:
        if hasattr(self.criterion, "set_teacher_weight"):
            teacher_weight = self.config.get_kd_teacher_weight(epoch_index, max(self.config.epochs, 1))
            self.criterion.set_teacher_weight(teacher_weight)  # type: ignore
            logger.info(
                f"KD teacher weight set to {teacher_weight:.4f} for epoch {epoch_index + 1}/{max(self.config.epochs, 1)}"
            )
    
    def _remove_teacher_weight_for_evaluation(self) -> None:
        if hasattr(self.criterion, "set_teacher_weight"):
            self.criterion.set_teacher_weight(0.0)  # type: ignore
            logger.info(
                "KD teacher weight set to 0 for evaluation to ensure student performance is measured against ground-truth labels."
            )                                                                                    

    # Moves the batch to the appropriate device with non_blocking DMA transfers
    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor | None]:
        labels = batch["labels"].to(self.device, non_blocking=True)
        if self.config.problem_type == "multi_label":
            labels = labels.float()
        else:
            labels = labels.long()

        token_type_ids = batch.get("token_type_ids")
        
        prepared = {
            "input_ids": batch["input_ids"].to(self.device, non_blocking=True),
            "attention_mask": batch["attention_mask"].to(self.device, non_blocking=True),
            "token_type_ids": (
                token_type_ids.to(self.device, non_blocking=True) 
                if token_type_ids is not None 
                else None
            ),
            "labels": labels,
        }
        
        if "logits" in batch:
            prepared["logits"] = batch["logits"].to(self.device, non_blocking=True)
            
        return prepared

    def train_epoch(self, dataloader: DataLoader, optimizer: AdamW, scheduler: Any) -> float:
        if len(dataloader) == 0:
            raise ValueError("Cannot train with an empty dataloader")

        self.model.train()
        total_loss = torch.zeros((), device=self.device)
        processed_batches = 0

        use_amp = self.device.type == "cuda"
        if not hasattr(self, "scaler"):
            self.scaler = GradScaler(enabled=use_amp)

        for batch in tqdm(dataloader, desc="Training Iteration", leave=False):
            optimizer.zero_grad(set_to_none=True)

            prepared = self._prepare_batch(batch)
            labels = prepared["labels"]
            assert labels is not None, "Labels tensor cannot be None"

            with autocast(enabled=use_amp):
                logits = self.model(
                    prepared["input_ids"], 
                    prepared["attention_mask"], 
                    prepared["token_type_ids"]
                )
                
                if self.config.loss_type == 'kldiv':
                    teacher_logits = prepared["logits"]
                    loss = self.criterion(logits, teacher_logits, labels)
                else:
                    loss = self.criterion(logits, labels)

            # Scaled Backward Pass & Gradient Clipping
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(optimizer)
            self.scaler.update()

            scheduler.step()

            # GPU-side loss accumulation
            total_loss += loss.detach()
            processed_batches += 1

        return (total_loss / processed_batches).item()

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        if len(dataloader) == 0:
            raise ValueError("Cannot evaluate with an empty dataloader")

        self.model.eval()
        total_loss = torch.zeros((), device=self.device)
        use_amp = self.device.type == "cuda"

        all_preds = []
        all_labels = []
        processed_batches = 0

        for batch in tqdm(dataloader, desc="Evaluation Iteration", leave=False):
            prepared = self._prepare_batch(batch)
            labels = prepared["labels"]
            assert labels is not None, "Labels tensor cannot be None"
            
            with autocast(enabled=use_amp):
                logits = self.model(
                    prepared["input_ids"], 
                    prepared["attention_mask"], 
                    prepared["token_type_ids"]
                )
                if self.config.loss_type == 'kldiv':
                    teacher_logits = prepared["logits"]
                    loss = self.criterion(logits, teacher_logits, labels)
                else:
                    loss = self.criterion(logits, labels)

            total_loss += loss.detach()

            if self.config.problem_type == "multi_label":
                preds = (torch.sigmoid(logits) >= 0.5).int()
            else:
                preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            processed_batches += 1

        full_preds = torch.cat(all_preds, dim=0).numpy()
        full_labels = torch.cat(all_labels, dim=0).numpy()

        avg_loss = (total_loss / processed_batches).item()
        macro_f1 = float(f1_score(full_labels, full_preds, average="macro", zero_division=0))
        micro_f1 = float(f1_score(full_labels, full_preds, average="micro", zero_division=0))

        return {"loss": avg_loss, "macro_f1": macro_f1, "micro_f1": micro_f1}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> str | None:
        if self.config.epochs == 0:
            logger.info("Configured epochs=0; skipping training and validation.")
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
        
        effective_train_batches = len(train_loader)
        total_steps = effective_train_batches * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        
        best_macro_f1 = -inf
        best_checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
        best_epoch = 0
        for epoch in range(self.config.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")

            self._sync_teacher_weight(epoch)
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)

            self._remove_teacher_weight_for_evaluation()  
            metrics = self.evaluate(val_loader)
            
            logger.info(
                f"Train Loss: {train_loss:.4f} | Val Loss: {metrics['loss']:.4f} | "
                f"Val Macro-F1: {metrics['macro_f1']:.4f} | Val Micro-F1: {metrics['micro_f1']:.4f}"
            )
            
            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                torch.save(self.model.state_dict(), best_checkpoint_path)
                best_epoch = epoch + 1
                logger.info(f"Saved best checkpoint to {best_checkpoint_path} with Macro-F1: {best_macro_f1:.4f}")

        SoftTargetExporter.save_best_epoch(self.config.checkpoint_dir, best_epoch)
        return best_checkpoint_path