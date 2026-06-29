import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal
from Loss_functions import LossFunctions


@dataclass
class TeacherConfig:
    task_name: str
    num_labels: int
    problem_type: Literal["single_label", "multi_label"]
    loss_type: Literal["cross_entropy", "bce_with_logits"]
    model_name_or_path: str = "nlpaueb/legal-bert-base-uncased"
    
    # Cut data for quicker testing
    num_of_batches: int = -1  # -1 means use all batches in the dataloader
    percent_of_data: int = 100  # Percentage of data to use for training

    # Optimization Hyperparameters
    batch_size: int = 16
    learning_rate: float = 3e-5
    epochs: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    loss_reduction : Literal["mean", "sum"] = "mean"
    
    # Hardware Routing
    device: Literal["auto", "cuda", "cpu"] = "auto"
    seed: int = 42
    
    # Path Resolution
    checkpoint_dir: str = ""
    output_dir: str = ""
    
    def __post_init__(self):
        if self.num_labels <= 0:
            raise ValueError(f"num_labels must be positive, got {self.num_labels}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs < 0:
            raise ValueError(f"epochs must be non-negative, got {self.epochs}")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError(f"warmup_ratio must be between 0 and 1, got {self.warmup_ratio}")

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        if self.device == "cpu":
            self.mixed_precision = False

        if not self.checkpoint_dir:
            self.checkpoint_dir = f"./checkpoints/{self.task_name}_teacher"
        if not self.output_dir:
            self.output_dir = f"./ds_with_teacher_outputs/{self.task_name}_teacher_outputs"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        valid = {
        ("single_label", "cross_entropy"),
        ("multi_label", "bce_with_logits")
        }
        if (self.problem_type, self.loss_type) not in valid:
            raise ValueError(
                f"Invalid configuration: "
                f"{self.problem_type=} {self.loss_type=}"
            )

    def get_loss_criterion(self) -> nn.Module:
        return LossFunctions.get_loss_function(self.problem_type, self.loss_type, self.loss_reduction)

    
        
