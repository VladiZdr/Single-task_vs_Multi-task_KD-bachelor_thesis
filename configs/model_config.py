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
    model_name_or_path: Literal["google/bert_uncased_L-4_H-256_A-4", "nlpaueb/legal-bert-base-uncased"] = "nlpaueb/legal-bert-base-uncased"
    
    # Cut data for quicker testing
    num_of_batches: int = -1    # -1 means use all batches in the dataloader (affects only training, evaluation, and export will still process all batches)
    percent_of_data: int = 100  # Use only "percent_of_data" % of the dataset for quicker testing

    # Optimization Hyperparameters
    batch_size: int = 8
    learning_rate: float = 3e-5
    epochs: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    T: float = 1.0
    alpha: float = 0.5
    loss_reduction : Literal["mean", "sum"] = "mean"
    kd_teacher_weight_schedule: Literal["constant", "linear_epoch"] = "constant"
    kd_teacher_weight_start: float = 1.0
    kd_teacher_weight_end: float = 1.0
    
    # Hardware Routing
    device: Literal["auto", "cuda", "cpu"] = "auto"
    seed: int = 42
    
    # Path Resolution
    checkpoint_dir: str = ""
    output_dir: str = ""
    unique_id_for_dir: str = ""
    # raw means the dataset will be preprocessed from scratch, otherwise it will be loaded from disk
    preprocessed_data_dir: Literal["raw", "./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_outputs_tester", "./datasets_store/unfair_tos_preprocessed", "./datasets_store/ds_with_teacher_outputs/ledgar_teacher_outputs", "./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_outputs"]  = "raw" 
    
    def __post_init__(self):
        if self.task_name == "ledgar" and self.num_labels != 100:
            raise ValueError(f"For task 'ledgar', num_labels must be 100, got {self.num_labels}.")
        if self.task_name == "unfair_tos" and self.num_labels != 8:
            raise ValueError(f"For task 'unfair_tos', num_labels must be 8, got {self.num_labels}.")
        
        if self.num_labels <= 0:
            raise ValueError(f"num_labels must be positive, got {self.num_labels}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.epochs < 0:
            raise ValueError(f"epochs must be non-negative, got {self.epochs}")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError(f"warmup_ratio must be between 0 and 1, got {self.warmup_ratio}")

        if not 0.0 <= self.kd_teacher_weight_start <= 1.0:
            raise ValueError(f"kd_teacher_weight_start must be between 0 and 1, got {self.kd_teacher_weight_start}")
        if not 0.0 <= self.kd_teacher_weight_end <= 1.0:
            raise ValueError(f"kd_teacher_weight_end must be between 0 and 1, got {self.kd_teacher_weight_end}")

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        if self.device == "cpu":
            self.mixed_precision = False

        if len(self.unique_id_for_dir) > 25:
            raise ValueError(f"Path directory too long. Shorten unique_id_for_dir to 25 characters or less. Current length: {len(self.unique_id_for_dir)}")
        if not self.checkpoint_dir:
            self.checkpoint_dir = f"./datasets_store/checkpoints/{self.task_name}_{self.unique_id_for_dir}"
        if not self.output_dir:
            self.output_dir = f"./datasets_store/ds_with_teacher_outputs/{self.task_name}_teacher_outputs_{self.unique_id_for_dir}"
        os.makedirs(self.checkpoint_dir, exist_ok=True) # if the directory doesn't exist, create it
        os.makedirs(self.output_dir, exist_ok=True)     # if the directory exists, do nothing -> it will be overwritten

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

    def get_loss_criterion(self) -> nn.Module:
        return LossFunctions.get_loss_function(self.problem_type, self.loss_type, self.loss_reduction, self.T, self.alpha)

    def get_kd_teacher_weight(self, epoch_index: int, total_epochs: int) -> float:
        if self.loss_type != "kldiv":
            return 1.0

        if self.kd_teacher_weight_schedule == "constant" or total_epochs <= 1:
            return self.kd_teacher_weight_start

        progress = max(0.0, min(epoch_index / (total_epochs - 1), 1.0))
        return self.kd_teacher_weight_start + progress * (self.kd_teacher_weight_end - self.kd_teacher_weight_start)

# Testers  
ledgar_teacher_tester = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",

    num_of_batches=-1,
    percent_of_data=1,  

    batch_size = 16,
    learning_rate = 3e-5,
    epochs = 0,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    max_grad_norm = 1.0,
    T = 1.0,
    alpha = 0.5,
    loss_reduction = "mean",
    kd_teacher_weight_schedule = "constant",
    kd_teacher_weight_start = 1.0,
    kd_teacher_weight_end = 1.0,

    device = "auto",
    seed = 42,

    checkpoint_dir = "",
    output_dir = "",
    unique_id_for_dir = "tester",
    preprocessed_data_dir = "raw"
)
    
    # 2. Pipeline Definition for UNFAIR-ToS Terms Identification

unfair_tos_teacher_tester = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",

    percent_of_data=1,  
    num_of_batches=-1,  # all
    
    batch_size=4,
    epochs=1,
    learning_rate=3e-5,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    max_grad_norm = 1.0,
    T = 1.0,
    alpha = 0.5,
    loss_reduction = "mean",
    kd_teacher_weight_schedule = "constant",
    kd_teacher_weight_start = 1.0,
    kd_teacher_weight_end = 1.0,

    device = "auto",
    seed = 42,

    checkpoint_dir = "",
    output_dir = "",
    unique_id_for_dir = "tester",
    preprocessed_data_dir = "raw"
)

unfair_tos_supervised_student_tester = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    percent_of_data=1,  
    num_of_batches=-1,  
    
    batch_size=4,
    epochs=1,
    learning_rate=3e-5,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    max_grad_norm = 1.0,
    T = 1.0,
    alpha = 0.5,
    loss_reduction = "mean",
    kd_teacher_weight_schedule = "constant",
    kd_teacher_weight_start = 1.0,
    kd_teacher_weight_end = 1.0,

    device = "auto",
    seed = 42,

    checkpoint_dir = "",
    output_dir = "",
    unique_id_for_dir = "supervised_student_tester",
    preprocessed_data_dir = "raw"
)

unfair_tos_check_correct_load_preprocessed_dataset = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",

    percent_of_data=1,  
    num_of_batches=-1,  
    
    batch_size=4,
    epochs=1,
    learning_rate=3e-5,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    max_grad_norm = 1.0,
    T = 1.0,
    alpha = 0.5,
    loss_reduction = "mean",
    kd_teacher_weight_schedule = "constant",
    kd_teacher_weight_start = 1.0,
    kd_teacher_weight_end = 1.0,

    device = "auto",
    seed = 42,

    checkpoint_dir = "",
    output_dir = "",
    unique_id_for_dir = "check_correct_load",
    preprocessed_data_dir = "./datasets_store/unfair_tos_preprocessed"
)

unfair_tos_kd_student_tester = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    percent_of_data=1,  
    num_of_batches=-1,  
    
    batch_size=4,
    epochs=2,
    learning_rate=3e-5,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    max_grad_norm = 1.0,
    T = 1.0,
    alpha = 0.5,
    loss_reduction = "mean",
    kd_teacher_weight_schedule = "linear_epoch",
    kd_teacher_weight_start = 1.0,
    kd_teacher_weight_end = 0.0,

    device = "auto",
    seed = 42,

    checkpoint_dir = "",
    output_dir = "",
    unique_id_for_dir = "kd_student_tester",
    preprocessed_data_dir = "./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_outputs_tester"
)

"""
# TODO: add kd_teacher_weight_schedule
# Teachers
ledgar_teacher = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",

    num_of_batches=-1,
    percent_of_data=100,  

    batch_size = 8,
    learning_rate = 3e-5,
    epochs = 5,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    max_grad_norm = 1.0,
    T = 1.0,
    alpha = 0.5,
    loss_reduction = "mean",

    device = "auto",
    seed = 42,

    checkpoint_dir = "./datasets_store/checkpoints/ledgar_teacher",
    output_dir = "./datasets_store/ds_with_teacher_outputs/ledgar_teacher_outputs",
    unique_id_for_dir = "Teacher",
    preprocessed_data_dir = "raw"
)

unfair_tos_teacher = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="nlpaueb/legal-bert-base-uncased",

    num_of_batches=-1,
    percent_of_data=100,  

    batch_size = 8,
    learning_rate = 3e-5,
    epochs = 5,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    max_grad_norm = 1.0,
    T = 1.0,
    alpha = 0.5,
    loss_reduction = "mean",

    device = "auto",
    seed = 42,

    checkpoint_dir = "./datasets_store/checkpoints/unfair_tos_teacher",
    output_dir = "./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_outputs",
    unique_id_for_dir = "Teacher",
    preprocessed_data_dir = "raw"
)

# Supervised Students Baselines   
ledgar_supervised_student_baseline = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="cross_entropy",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    num_of_batches=-1,
    percent_of_data=100,  

    batch_size = 8,
    learning_rate = 3e-5,
    epochs = 5,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    max_grad_norm = 1.0,
    T = 1.0,
    alpha = 0.5,
    loss_reduction = "mean",

    device = "auto",
    seed = 42,

    checkpoint_dir = "./datasets_store/checkpoints/ledgar_supervised_student",
    output_dir = "./datasets_store/ds_with_teacher_outputs/ledgar_supervised_student_outputs",
    unique_id_for_dir = "Baseline",
    preprocessed_data_dir = "raw"
)

unfair_tos_supervised_student_baseline = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="bce_with_logits",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    num_of_batches=-1,
    percent_of_data=100,  

    batch_size = 8,
    learning_rate = 3e-5,
    epochs = 5,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    max_grad_norm = 1.0,
    T = 1.0,
    alpha = 0.5,
    loss_reduction = "mean",

    device = "auto",
    seed = 42,

    checkpoint_dir = "./datasets_store/checkpoints/unfair_tos_supervised_student",
    output_dir = "./datasets_store/ds_with_teacher_outputs/unfair_tos_supervised_student_outputs",
    unique_id_for_dir = "Baseline",
    preprocessed_data_dir = "raw"
)
 
# Single-task Knowledge Distillation Students
ledgar_kd_student = ModelConfig(
    task_name="ledgar",
    num_labels=100,
    problem_type="single_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    num_of_batches=-1,
    percent_of_data=100,  

    batch_size = 8,
    learning_rate = 3e-5,
    epochs = 5,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    max_grad_norm = 1.0,
    T = 1.0,
    alpha = 0.5,
    loss_reduction = "mean",

    device = "auto",
    seed = 42,

    checkpoint_dir = "./datasets_store/checkpoints/ledgar_kd_student",
    output_dir = "./datasets_store/ds_with_teacher_outputs/ledgar_kd_student_outputs",
    unique_id_for_dir = "Single_task_KD_Student",
    preprocessed_data_dir = "./datasets_store/ds_with_teacher_outputs/ledgar_teacher_outputs"
)

unfair_tos_kd_student = ModelConfig(
    task_name="unfair_tos",
    num_labels=8,
    problem_type="multi_label",
    loss_type="kldiv",
    model_name_or_path="google/bert_uncased_L-4_H-256_A-4",

    num_of_batches=-1,
    percent_of_data=100,  

    batch_size = 8,
    learning_rate = 3e-5,
    epochs = 5,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    max_grad_norm = 1.0,
    T = 1.0,
    alpha = 0.5,
    loss_reduction = "mean",

    device = "auto",
    seed = 42,

    checkpoint_dir = "./datasets_store/checkpoints/unfair_tos_kd_student",
    output_dir = "./datasets_store/ds_with_teacher_outputs/unfair_tos_kd_student_outputs",
    unique_id_for_dir = "Single_task_KD_Student",
    preprocessed_data_dir = "./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_outputs"
)
 
"""
