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

from configs.model_configs import ModelConfig
from datasets_manipulation.export_teacher_outputs import SoftTargetExporter
from multi_task.multi_task_model import MultiTaskModel

logger = logging.getLogger(__name__)


class MultiTaskTrainer:
    """Trainer for sequential multi-task fine-tuning across LEDGAR and UNFAIR-ToS."""

    scaler: GradScaler

    def __init__(self, model: MultiTaskModel):
        self.model = model
        self.ledgar_config: ModelConfig = model.ledgar_config
        self.unfair_tos_config: ModelConfig = model.unfair_tos_config
        self.task_configs: dict[str, ModelConfig] = {
            "ledgar": self.ledgar_config,
            "unfair_tos": self.unfair_tos_config,
        }

        self.checkpoint_path = f"./datasets_store/checkpoints/{self.model.unique_id_for_dir}"
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.device = torch.device(self.ledgar_config.device)
        self.model.to(self.device)

        # Comprehends and instantiates custom loss calculation functions for each dataset
        self.criterions: dict[str, nn.Module] = {
            task_name: task_config.get_loss_criterion()
            for task_name, task_config in self.task_configs.items()
        }

        # Explicitly sets the order in which data splits are trained sequentially 
        # and seeds the starting teacher weight parameter to 0 to begin the first epoch.
        self.train_task_order = ("ledgar", "unfair_tos")
        self._sync_teacher_weight(epoch_index=0)

    def _set_teacher_weight(self, weight: float) -> None:
        for criterion in self.criterions.values():
            if hasattr(criterion, "set_teacher_weight"):
                criterion.set_teacher_weight(weight)  # type: ignore[attr-defined]

    # Dynamically shifts how much the student model listens to the teacher vs. ground-truth labels
    def _sync_teacher_weight(self, epoch_index: int) -> None:
        teacher_weight = self.ledgar_config.get_kd_teacher_weight(epoch_index, max(self.ledgar_config.epochs, 1))
        self._set_teacher_weight(teacher_weight)
        logger.info(
            f"KD teacher weight set to {teacher_weight:.4f} for epoch {epoch_index + 1}/{max(self.ledgar_config.epochs, 1)}"
        )

    # This ensures evaluation performance metrics reflect only hard ground-truth targets.
    def _remove_teacher_weight_for_evaluation(self) -> None:
        self._set_teacher_weight(0.0)
        logger.info(
            "KD teacher weight set to 0 for evaluation to ensure student performance is measured against ground-truth labels."
        )

    # Scours incoming data dictionaries for a "task" descriptor key to understand what tracking path to use.
    def _task_name_from_batch(self, batch: Dict[str, Any]) -> str:
        task_value = batch.get("task")
        if task_value is None:
            raise KeyError("Multi-task batches must contain a 'task' column.")

        # 1. Unpack PyTorch Tensors if batch["task"] is loaded as a tensor
        if isinstance(task_value, torch.Tensor):
            if task_value.dtype == torch.uint8 and task_value.ndim == 2:
                task_value = [
                    bytes(row[row != 0].tolist()).decode("utf-8", errors="ignore")
                    for row in task_value
                ]
            elif task_value.ndim == 0:
                task_value = task_value.item()
            else:
                task_value = task_value.tolist()

        # 2. Process list / tuple / set containers
        if isinstance(task_value, (list, tuple, set)):
            extracted_tasks: list[str] = []
            for item in task_value:
                if isinstance(item, str):
                    extracted_tasks.append(item)
                elif isinstance(item, (list, tuple)):
                    if item and all(isinstance(x, int) for x in item):
                        extracted_tasks.append(bytes([x for x in item if x != 0]).decode("utf-8", errors="ignore"))
                    else:
                        raise TypeError(f"Expected a string task label in batch list, got {type(item)!r}")
                else:
                    raise TypeError(f"Expected a string task label in batch list, got {type(item)!r}")

            unique_tasks = sorted(set(extracted_tasks))
            if len(unique_tasks) != 1:
                raise ValueError(f"Mixed-task batches are not supported: {unique_tasks}")
            task_value = unique_tasks[0]

        # 3. Enforce strict type checking for non-string raw values
        if not isinstance(task_value, str):
            raise TypeError(f"Expected a string task label, got {type(task_value)!r}")

        # 4. Verify task exists in task_configs
        if task_value not in self.task_configs:
            raise ValueError(f"Unknown task '{task_value}'. Expected one of {sorted(self.task_configs.keys())}.")

        return task_value

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor | str | None]:
        task_name = self._task_name_from_batch(batch)
        task_config = self.task_configs[task_name]

        # Enable non_blocking=True for fast async host-to-device transfers
        labels = batch["labels"].to(self.device, non_blocking=True)  # type: ignore
        if task_config.problem_type == "multi_label":
            labels = labels.float()
        else:
            labels = labels.long()

        token_type_ids = batch.get("token_type_ids")
        prepared: Dict[str, torch.Tensor | str | None] = {
            "input_ids": batch["input_ids"].to(self.device, non_blocking=True),  # type: ignore
            "attention_mask": batch["attention_mask"].to(self.device, non_blocking=True),  # type: ignore
            "token_type_ids": token_type_ids.to(self.device, non_blocking=True) if token_type_ids is not None else None,  # type: ignore
            "labels": labels,
            "task": task_name,
        }

        if "logits" in batch:
            prepared["logits"] = batch["logits"].to(self.device, non_blocking=True)  # type: ignore

        return prepared

    def _compute_loss(
        self,
        task_name: str,
        logits: torch.Tensor,
        prepared_batch: Dict[str, torch.Tensor | str | None],
    ) -> torch.Tensor:
        task_config = self.task_configs[task_name]
        criterion = self.criterions[task_name]
        labels = prepared_batch["labels"]

        assert isinstance(labels, torch.Tensor), "Labels must be a Tensor"

        if task_config.loss_type == "kldiv":
            teacher_logits = prepared_batch.get("logits")
            if teacher_logits is None:
                raise KeyError(
                    f"Task '{task_name}' is configured for KD but the batch does not contain teacher logits."
                )
            return criterion(logits, teacher_logits, labels)  # type: ignore[misc]

        return criterion(logits, labels)  # type: ignore[misc]

    def train_epoch(self, train_loaders: Dict[str, DataLoader], optimizer: AdamW, scheduler: Any) -> float:
        if not train_loaders:
            raise ValueError("Cannot train with no dataloaders")

        self.model.train()
        total_loss = torch.zeros((), device=self.device)
        processed_batches = 0

        use_amp = self.device.type == "cuda"
        if not hasattr(self, "scaler"):
            self.scaler = GradScaler(enabled=use_amp)

        active_tasks = [
            task
            for task in self.train_task_order
            if task in train_loaders and train_loaders[task] is not None and len(train_loaders[task]) > 0
        ]

        if not active_tasks:
            raise ValueError("Cannot train with no valid dataloaders")

        iterators = {task: iter(train_loaders[task]) for task in active_tasks}
        remaining_tasks = list(active_tasks)
        total_expected_batches = sum(len(train_loaders[task]) for task in active_tasks)

        with tqdm(total=total_expected_batches, desc="Multi-task Training (Round-Robin)") as pbar:
            while remaining_tasks:
                for task in list(remaining_tasks):
                    try:
                        batch = next(iterators[task])
                    except StopIteration:
                        remaining_tasks.remove(task)
                        continue

                    optimizer.zero_grad(set_to_none=True)

                    prepared = self._prepare_batch(batch)
                    labels = prepared["labels"]
                    batch_task = prepared["task"]
                    input_ids = prepared["input_ids"]
                    attention_mask = prepared["attention_mask"]
                    token_type_ids = prepared["token_type_ids"]

                    assert isinstance(labels, torch.Tensor), "Labels must be a Tensor"
                    assert isinstance(batch_task, str), "Task must be a string"
                    assert isinstance(input_ids, torch.Tensor), "input_ids must be a Tensor"
                    assert isinstance(attention_mask, torch.Tensor), "attention_mask must be a Tensor"

                    # AMP autocast forward pass
                    with autocast(enabled=use_amp):
                        logits = self.model(
                            input_ids, 
                            attention_mask, 
                            token_type_ids, 
                            task=batch_task
                        )
                        loss = self._compute_loss(batch_task, logits, prepared)

                    # Scaled backward pass
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.ledgar_config.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()

                    scheduler.step()

                    # GPU-side loss tracking
                    total_loss += loss.detach()
                    processed_batches += 1
                    pbar.update(1)

        if processed_batches == 0:
            raise ValueError("No batches were processed during multi-task training")

        return (total_loss / processed_batches).item()

    @torch.no_grad()
    def evaluate(self, dataloaders: Dict[str, DataLoader]) -> Dict[str, float]:
        if not dataloaders:
            raise ValueError("Cannot evaluate with no dataloaders")

        self.model.eval()
        overall_loss = torch.zeros((), device=self.device)
        overall_batches = 0
        use_amp = self.device.type == "cuda"

        task_results: dict[str, dict[str, float]] = {}

        for task_name in self.train_task_order:
            dataloader = dataloaders.get(task_name)
            if dataloader is None or len(dataloader) == 0:
                continue

            task_loss = torch.zeros((), device=self.device)
            task_batches = 0
            all_preds = []
            all_labels = []

            for batch in tqdm(dataloader, desc=f"Evaluation {task_name}"):
                prepared = self._prepare_batch(batch)
                labels = prepared["labels"]
                task = prepared["task"]
                input_ids = prepared["input_ids"]
                attention_mask = prepared["attention_mask"]
                token_type_ids = prepared["token_type_ids"]

                assert isinstance(labels, torch.Tensor), "Labels must be a Tensor"
                assert isinstance(task, str), "Task must be a string"
                assert isinstance(input_ids, torch.Tensor), "input_ids must be a Tensor"
                assert isinstance(attention_mask, torch.Tensor), "attention_mask must be a Tensor"

                with autocast(enabled=use_amp):
                    logits = self.model(
                        input_ids, 
                        attention_mask, 
                        token_type_ids, 
                        task=task
                    )
                    loss = self._compute_loss(task, logits, prepared)

                task_loss += loss.detach()
                overall_loss += loss.detach()

                if self.task_configs[task].problem_type == "multi_label":
                    preds = (torch.sigmoid(logits) >= 0.5).int()
                else:
                    preds = torch.argmax(logits, dim=-1)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

                task_batches += 1
                overall_batches += 1

            full_preds = torch.cat(all_preds, dim=0).numpy()
            full_labels = torch.cat(all_labels, dim=0).numpy()

            task_results[task_name] = {
                "loss": (task_loss / task_batches).item(),
                "macro_f1": float(f1_score(full_labels, full_preds, average="macro", zero_division=0)),
                "micro_f1": float(f1_score(full_labels, full_preds, average="micro", zero_division=0)),
            }

        if not task_results:
            raise ValueError("No evaluation batches were processed")

        macro_f1 = sum(result["macro_f1"] for result in task_results.values()) / len(task_results)
        micro_f1 = sum(result["micro_f1"] for result in task_results.values()) / len(task_results)

        metrics: Dict[str, float] = {
            "loss": (overall_loss / overall_batches).item(),
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
        }

        for task_name, result in task_results.items():
            metrics[f"{task_name}_loss"] = result["loss"]
            metrics[f"{task_name}_macro_f1"] = result["macro_f1"]
            metrics[f"{task_name}_micro_f1"] = result["micro_f1"]

        return metrics

    def fit(self, train_loaders: Dict[str, DataLoader], val_loaders: Dict[str, DataLoader]) -> str | None:
        if self.ledgar_config.epochs == 0:
            logger.info("Configured epochs=0; skipping training and validation.")
            return None

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.ledgar_config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.ledgar_config.learning_rate)

        effective_train_batches = sum(len(loader) for loader in train_loaders.values())

        total_steps = effective_train_batches * self.ledgar_config.epochs
        warmup_steps = int(total_steps * self.ledgar_config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_macro_f1 = -inf
        best_checkpoint_path = os.path.join(self.checkpoint_path, "best_multi_task_model.pt")
        best_epoch = 0
        for epoch in range(self.ledgar_config.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.ledgar_config.epochs}")

            self._sync_teacher_weight(epoch)
            train_loss = self.train_epoch(train_loaders, optimizer, scheduler)

            self._remove_teacher_weight_for_evaluation()
            metrics = self.evaluate(val_loaders)

            logger.info(
                "Train Loss: %.4f | Val Loss: %.4f | Val Macro-F1: %.4f | Val Micro-F1: %.4f",
                train_loss,
                metrics["loss"],
                metrics["macro_f1"],
                metrics["micro_f1"],
            )
            logger.info(
                "LEDGAR F1: %.4f | UNFAIR-ToS F1: %.4f",
                metrics.get("ledgar_macro_f1", 0.0),
                metrics.get("unfair_tos_macro_f1", 0.0),
            )

            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                torch.save(self.model.state_dict(), best_checkpoint_path)
                best_epoch = epoch + 1
                logger.info(f"Saved best checkpoint to {best_checkpoint_path} with Macro-F1: {best_macro_f1:.4f}")

        SoftTargetExporter.save_best_epoch(self.checkpoint_path, best_epoch)
        return best_checkpoint_path