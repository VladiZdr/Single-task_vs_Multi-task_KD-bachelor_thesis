from __future__ import annotations

import logging
import os
from math import inf
from typing import Any, Dict

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


class MultiTaskTrainer:
    """Trainer for sequential multi-task fine-tuning across LEDGAR and UNFAIR-ToS."""

    def __init__(self, model: nn.Module, ledgar_config: ModelConfig, unfair_tos_config: ModelConfig):
        self.model = model
        self.ledgar_config = ledgar_config
        self.unfair_tos_config = unfair_tos_config
        self.task_configs: dict[str, ModelConfig] = {"ledgar": ledgar_config, "unfair_tos": unfair_tos_config,}

        self.device = torch.device(ledgar_config.device)
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
    # If missing, it raises a fatal error.
    def _task_name_from_batch(self, batch: Dict[str, Any]) -> str:
        task_value = batch.get("task")
        if task_value is None:
            raise KeyError("Multi-task batches must contain a 'task' column.")

        # Ensures every single item in that batch belongs to the same task.
        if isinstance(task_value, (list, tuple)):
            unique_tasks = sorted(set(task_value))
            if len(unique_tasks) != 1:
                raise ValueError(f"Mixed-task batches are not supported: {unique_tasks}")
            task_value = unique_tasks[0]

        if not isinstance(task_value, str):
            raise TypeError(f"Expected a string task label, got {type(task_value)!r}")

        if task_value not in self.task_configs:
            raise ValueError(f"Unknown task '{task_value}'. Expected one of {sorted(self.task_configs.keys())}.")

        return task_value

    def _prepare_batch(self, batch: Dict[str, torch.Tensor | list[str] | str]) -> Dict[str, torch.Tensor | str | None]:
        task_name = self._task_name_from_batch(batch)
        task_config = self.task_configs[task_name]

        # Extracts data labels and pushes them onto the device. If the configuration states a task is multi_label 
        # (classification containing multiple active classes), it formats parameters into a float(). 
        # Otherwise, it converts them into a long() integer for classic single-choice multi-class operations.
        labels = batch["labels"].to(self.device)        #type: ignore
        if task_config.problem_type == "multi_label":
            labels = labels.float()
        else:
            labels = labels.long()

        token_type_ids = batch.get("token_type_ids")
        prepared: Dict[str, torch.Tensor | str | None] = {
            "input_ids": batch["input_ids"].to(self.device),                                                 #type: ignore
            "attention_mask": batch["attention_mask"].to(self.device),                                       #type: ignore
            "token_type_ids": token_type_ids.to(self.device) if token_type_ids is not None else None,        #type: ignore
            "labels": labels,
            "task": task_name,
        }

        if "logits" in batch:
            prepared["logits"] = batch["logits"].to(self.device)                                             #type: ignore

        return prepared

    # Localizes reference links back out to the specified task's structural specifications and target loss modules.
    def _compute_loss(self, task_name: str, logits: torch.Tensor, prepared_batch: Dict[str, torch.Tensor | str | None],) -> torch.Tensor:
        task_config = self.task_configs[task_name]
        criterion = self.criterions[task_name]
        labels = prepared_batch["labels"]

        assert isinstance(labels, torch.Tensor)

        if task_config.loss_type == "kldiv":
            teacher_logits = prepared_batch.get("logits")
            if teacher_logits is None:
                raise KeyError(
                    f"Task '{task_name}' is configured for KD but the batch does not contain teacher logits."
                )
            return criterion(logits, teacher_logits, labels)  # type: ignore[misc]

        return criterion(logits, labels)  # type: ignore[misc]

    def train_epoch(self, train_loaders: Dict[str, DataLoader], optimizer: AdamW, scheduler: Any,) -> float:
        if not train_loaders:
            raise ValueError("Cannot train with no dataloaders")

        self.model.train()
        total_loss = 0.0
        processed_batches = 0

        # Sequential loop execution. This trainer processes the entire ledgar pipeline first before advancing forward onto the unfair_tos components.
        for task_name in self.train_task_order:
            dataloader = train_loaders.get(task_name)
            if dataloader is None or len(dataloader) == 0:
                continue

            # Wraps data loading elements within a tqdm progress visualization widget
            for batch in tqdm(dataloader, desc=f"Training {task_name}"):

                # Wipes clean the historical backpropagation gradient parameters remaining from prior forward loops.
                optimizer.zero_grad(set_to_none=True)

                # Extracts data items out of local CPU processing arrays, pushes them down into designated target accelerator platform
                prepared = self._prepare_batch(batch)
                labels = prepared["labels"]
                task = prepared["task"]
                assert isinstance(labels, torch.Tensor)
                assert isinstance(task, str)

                # Triggers the forward loop step across the shared Transformer network into specific active classification task target layer
                logits = self.model(prepared["input_ids"], prepared["attention_mask"], prepared["token_type_ids"], task=task)

                assert logits.shape[0] == labels.shape[0], "Batch size mismatch"
                assert logits.shape[1] == self.task_configs[task].num_labels, "Logit dimension mismatch"

                #Extracts loss evaluation scores combining tracking indices and structural constraints altogether.
                loss = self._compute_loss(task, logits, prepared)

                # Updates the model's weights: backpropagation -> caps exploding gradients -> updates the actual model parameters -> scales down the active learning rate over time
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.ledgar_config.max_grad_norm)
                optimizer.step()
                scheduler.step()

                # Computes and returns the mean loss value across the entire training epoch loop.
                total_loss += float(loss.item())
                processed_batches += 1

        if processed_batches == 0:
            raise ValueError("No batches were processed during multi-task training")

        return total_loss / processed_batches

    @torch.no_grad()
    def evaluate(self, dataloaders: Dict[str, DataLoader]) -> Dict[str, float]:
        if not dataloaders:
            raise ValueError("Cannot evaluate with no dataloaders")

        self.model.eval()
        overall_loss = 0.0
        overall_batches = 0

        task_results: dict[str, dict[str, float]] = {}

        for task_name in self.train_task_order:
            dataloader = dataloaders.get(task_name)
            if dataloader is None or len(dataloader) == 0:
                continue

            task_loss = 0.0
            task_batches = 0
            all_preds = []
            all_labels = []

            for batch in tqdm(dataloader, desc=f"Evaluation {task_name}"):
                prepared = self._prepare_batch(batch)
                labels = prepared["labels"]
                task = prepared["task"]
                assert isinstance(labels, torch.Tensor)
                assert isinstance(task, str)

                logits = self.model(prepared["input_ids"], prepared["attention_mask"], prepared["token_type_ids"], task=task)

                loss = self._compute_loss(task, logits, prepared)
                task_loss += float(loss.item())
                overall_loss += float(loss.item())

                if self.task_configs[task].problem_type == "multi_label":
                    preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
                else:
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()

                #Appends batch arrays to global history lists to prepare for metric scoring.
                all_preds.extend(preds)
                all_labels.extend(labels.detach().cpu().numpy())

                task_batches += 1
                overall_batches += 1

            task_results[task_name] = {
                "loss": task_loss / task_batches,
                "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
                "micro_f1": float(f1_score(all_labels, all_preds, average="micro", zero_division=0)),
            }

        if not task_results:
            raise ValueError("No evaluation batches were processed")

        macro_f1 = sum(result["macro_f1"] for result in task_results.values()) / len(task_results)
        micro_f1 = sum(result["micro_f1"] for result in task_results.values()) / len(task_results)

        metrics: Dict[str, float] = {
            "loss": overall_loss / overall_batches,
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

        # Sets up the weight optimization framework using the selected learning rate.
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.ledgar_config.learning_rate)

        # Dynamically calculates the exact number of forward updates that will run across all sub-loaders.
        effective_train_batches = 0
        for loader in train_loaders.values():
            loader_batches = len(loader)
            effective_train_batches += loader_batches

        # Sets up a linear learning rate scheduler. It gradually ramps up the learning rate during the initial warm-up period, 
        # and then tapers it off to zero as it approaches total step capacity.
        total_steps = effective_train_batches * self.ledgar_config.epochs
        warmup_steps = int(total_steps * self.ledgar_config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_macro_f1 = -inf
        best_checkpoint_path = os.path.join("./datasets_store/checkpoints/multi_task_model", "best_multi_task_model.pt")

        # Every epoch updates the distillation teacher weight parameter, trains across both tasks sequentially, 
        # turns off distillation targets, and runs an evaluation loop over validation data.
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
                logger.info(f"Saved best checkpoint to {best_checkpoint_path} with Macro-F1: {best_macro_f1:.4f}")

        return best_checkpoint_path
