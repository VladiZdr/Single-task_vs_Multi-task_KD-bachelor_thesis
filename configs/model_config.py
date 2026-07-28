import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal
from configs.Loss_functions import LossFunctions


@dataclass
class ModelConfig:
    task_name: Literal["ledgar", "unfair_tos"]
    num_labels: Literal[8, 100]
    problem_type: Literal["single_label", "multi_label"]
    loss_type: Literal["cross_entropy", "bce_with_logits", "kldiv"]
    # In ModelConfig definition
    model_name_or_path: Literal[
        "google/bert_uncased_L-4_H-256_A-4",
        "nlpaueb/legal-bert-base-uncased",
        "tfidf_baseline",
    ]
    
    # Use only "percent_of_data" % of the dataset for quicker testing 
    # If used on already cut DB (student on teacher outputs) -> it will cut it further
    percent_of_data: int = 100  

    # Optimization Hyperparameters
    batch_size: int = 8
    learning_rate: float = 3e-5
    epochs: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    T: float = 1.0
    loss_reduction : Literal["mean", "sum"] = "mean"

    # Parameter for low resource experiments (cuts only train set)
    # Combined with "percent_of_data" leads to double cut on train set
    low_resource_percent: Literal[1, 10, 25, 50, 100] = 100

    # Knowledge Distillation Hyperparameters
    kd_teacher_weight_schedule: Literal["constant", "linear_epoch"] = "constant"
    kd_teacher_weight_start: float = 1.0
    kd_teacher_weight_end: float = 0.0
    
    # Hardware Routing
    device: Literal["auto", "cuda", "cpu"] = "auto"
    seed: int = 42
    
    # Path Resolution
    checkpoint_dir: str = ""
    output_dir: str = ""
    unique_id_for_dir: str = ""
    teacher: ModelConfig | None = None
    # Empty means "choose automatically" based on whether a teacher is attached (for KD) or "raw" for non KD
    preprocessed_data_dir: str = ""

    # Ensure correct configuration
    def __post_init__(self):
        # Correct task <-> label count
        if self.task_name == "ledgar" and self.num_labels != 100:
            raise ValueError(f"For task 'ledgar', num_labels must be 100, got {self.num_labels}.")
        if self.task_name == "unfair_tos" and self.num_labels != 8:
            raise ValueError(f"For task 'unfair_tos', num_labels must be 8, got {self.num_labels}.")

        # Correct values for hyperparamters
        if self.num_labels <= 0:
            raise ValueError(f"num_labels must be positive, got {self.num_labels}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs < 0:
            raise ValueError(f"epochs must be non-negative, got {self.epochs}")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError(f"warmup_ratio must be between 0 and 1, got {self.warmup_ratio}")
        if not 0 < self.percent_of_data <= 100:
            raise ValueError(f"percent_of_data must be between 1 and 100, got {self.percent_of_data}")
        if self.low_resource_percent not in (1, 10, 25, 50, 100):
            raise ValueError(
                f"low_resource_percent must be one of 1, 10, 25, 50, or 100, got {self.low_resource_percent}"
            )
        if not 0.0 <= self.kd_teacher_weight_start <= 1.0:
            raise ValueError(f"kd_teacher_weight_start must be between 0 and 1, got {self.kd_teacher_weight_start}")
        if not 0.0 <= self.kd_teacher_weight_end <= 1.0:
            raise ValueError(f"kd_teacher_weight_end must be between 0 and 1, got {self.kd_teacher_weight_end}")
        valid = {
                ("single_label", "cross_entropy"),
                ("multi_label", "bce_with_logits"),
                ("single_label", "kldiv"),
                ("multi_label", "kldiv")
                }
        if (self.problem_type, self.loss_type) not in valid:
                    raise ValueError(
                        f"Invalid configuration: "
                        f"{self.problem_type=} {self.loss_type=}"
                    )

        # Correct configuration for KD
        if self.loss_type == "kldiv" and self.teacher is None:
            raise ValueError("loss_type='kldiv' requires a teacher config.")
        if self.teacher is not None:
            if self.teacher.task_name != self.task_name:
                raise ValueError(
                    "Teacher and student must use the same task. "
                    f"Got student={self.task_name!r}, teacher={self.teacher.task_name!r}."
                )

            expected_preprocessed_dir = self.teacher.output_dir
            if not self.preprocessed_data_dir:
                self.preprocessed_data_dir = expected_preprocessed_dir
            elif self.preprocessed_data_dir != expected_preprocessed_dir:
                raise ValueError(
                    "When a teacher is attached, the student's preprocessed_data_dir must match "
                    "the teacher output_dir. "
                    f"Got student={self.preprocessed_data_dir!r}, teacher={expected_preprocessed_dir!r}."
                )

        # Default value for source DS
        elif not self.preprocessed_data_dir:
            self.preprocessed_data_dir = "raw"
            
        # Create unique directories for each model
        if len(self.unique_id_for_dir) > 25:
            raise ValueError(f"Path directory too long. Shorten unique_id_for_dir to 25 characters or less. Current length: {len(self.unique_id_for_dir)}")
        if not self.checkpoint_dir:
            self.checkpoint_dir = f"./datasets_store/checkpoints/{self.task_name}_{self.unique_id_for_dir}"
        if not self.output_dir:
            self.output_dir = f"./datasets_store/ds_with_teacher_outputs/{self.task_name}_teacher_outputs_{self.unique_id_for_dir}"

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        if self.device == "cpu":
            self.mixed_precision = False

    def get_loss_criterion(self) -> nn.Module:
        return LossFunctions.get_loss_function(self.problem_type, self.loss_type, self.loss_reduction, self.T)

    # Teacher anealing with linear decline from kd_teacher_weight_start -> kd_teacher_weight_end
    def get_kd_teacher_weight(self, epoch_index: int, total_epochs: int) -> float:
        if self.loss_type != "kldiv":
            return 1.0

        if self.kd_teacher_weight_schedule == "constant" or total_epochs <= 1:
            return self.kd_teacher_weight_start

        progress = max(0.0, min(epoch_index / (total_epochs - 1), 1.0))

        return self.kd_teacher_weight_start - progress * (self.kd_teacher_weight_start - self.kd_teacher_weight_end)

@dataclass
class MultiTaskModelConfig:
    ledgar_config: ModelConfig
    unfair_tos_config: ModelConfig
    unique_id_for_dir: str

    # Ensure Multi-task tasks share the same encoder and optimization / KD setup. 
    def __post_init__(self) -> None:
        shared_fields = (
            "model_name_or_path",
            "learning_rate",
            "epochs",
            "weight_decay",
            "warmup_ratio",
            "max_grad_norm",
            "T",
            "loss_reduction",
            "kd_teacher_weight_schedule",
            "kd_teacher_weight_start",
            "kd_teacher_weight_end",
            "low_resource_percent",
            "device",
            "seed",
        )

        mismatches: list[str] = []
        for field_name in shared_fields:
            ledgar_value = getattr(self.ledgar_config, field_name)
            unfair_value = getattr(self.unfair_tos_config, field_name)
            if ledgar_value != unfair_value:
                mismatches.append(
                    f"{field_name}: ledgar={ledgar_value!r}, unfair_tos={unfair_value!r}"
                )

        if mismatches:
            raise ValueError(
                "Multi-task configs must share the same encoder and hyperparameters, "
                "but the following fields differ:\n- " + "\n- ".join(mismatches)
            )

@dataclass
class TfidfBaselineConfig(ModelConfig):
    max_features: int = 10000
    hidden_dim: int = 0 # If we have a hidden layer -> how many input dimensions for it
    model_name_or_path: Literal[
        "google/bert_uncased_L-4_H-256_A-4",
        "nlpaueb/legal-bert-base-uncased",
        "tfidf_baseline",
    ] = "tfidf_baseline"

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "model_name_or_path", "tfidf_baseline")
        
        if self.max_features <= 0:
            raise ValueError(f"max_features must be positive, got {self.max_features}")
        if self.hidden_dim < 0:
            raise ValueError(f"hidden_dim must be non-negative, got {self.hidden_dim}")
        
        # Resolve cache directory for TF-IDF dataset tensors
        self.preprocessed_data_dir = (f"./datasets_store/tf_idf_tensors/{self.task_name}_{self.unique_id_for_dir}")