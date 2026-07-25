import logging
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from torch.utils.data import DataLoader
from configs.model_config import ModelConfig
import configs.model_templates as model_config
from datasets_manipulation.prepare_datasets import (
    get_torch_columns_for_split,
    prep_dataset_from_raw,
    align_dataset_tokenization,
    sample_low_resource_dataset,
    sample_percent_dataset_for_testing,
    smart_load_dataset,
)
from fine_tuning.legal_model import LegalModel
from fine_tuning.legal_model_trainer import LegalModelTrainer
from fine_tuning.export_teacher_outputs import SoftTargetExporter
import torch
import numpy as np
import random
import os

# Configure unified logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("FineTuneTeacherPipeline")

# Makes experiments reproducible
def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Avoid parallel CPU workers (num_workers > 0) accidentally generate identical random numbers
# Prevents every worker process applying the exact same "random" augmentations to different batches
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def prepare_dataloaders(task_config: ModelConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    set_all_seeds(task_config.seed)
    
    # 1. Load tokenized dataset from disk or raw
    if task_config.preprocessed_data_dir == "raw":
        preprocessed = prep_dataset_from_raw(task_config)
    else:
        preprocessed = smart_load_dataset(task_config)

    if isinstance(preprocessed, DatasetDict):
        pass
    elif isinstance(preprocessed, HFDataset):
        raise ValueError(
            f"prep_dataset('{task_config.task_name}') returned a single Dataset, "
            "but this pipeline expects a DatasetDict with train/validation/test splits."
        )
    else:
        raise TypeError(f"Unexpected dataset type: {type(preprocessed)}")

    # 2. Align standard input_ids/attention_mask to the active model view
    if hasattr(task_config, "model_name_or_path"):
        preprocessed = align_dataset_tokenization(preprocessed, task_config.model_name_or_path)

    # 3. Format each split individually so no split drops its unique or view columns
    is_kldiv = getattr(task_config, "loss_type", None) == "kldiv"
    
    for split_name in ["train", "validation", "test"]:
        if split_name in preprocessed:
            split_cols = get_torch_columns_for_split(
                preprocessed[split_name],                                                           #type: ignore
                include_logits=is_kldiv and ("logits" in preprocessed[split_name].column_names),    #type: ignore
            )
            preprocessed[split_name].set_format(type="torch", columns=split_cols)                   #type: ignore

    train_dataset = preprocessed["train"]
    val_dataset = preprocessed["validation"]
    test_dataset = preprocessed["test"]

    # 4. Create minibatches with isolated worker seeds
    generator = torch.Generator()
    generator.manual_seed(task_config.seed)

    train_loader = DataLoader(
        train_dataset,                                                                       #type: ignore
        batch_size=task_config.batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(val_dataset, batch_size=task_config.batch_size, shuffle=False)   #type: ignore
    test_loader = DataLoader(test_dataset, batch_size=task_config.batch_size, shuffle=False) #type: ignore

    return train_loader, val_loader, test_loader

def run_task_pipeline(task_config: ModelConfig) -> None:
    logger.info(
        f"Initializing optimization pipeline for task: {task_config.task_name.upper()}_{task_config.unique_id_for_dir} "
        f"with {task_config.epochs} epochs, {task_config.batch_size} batch size, "
        f"{task_config.percent_of_data}% testing data slice, and "
        f"{task_config.low_resource_percent}% low-resource train slice."
    )
    if task_config.epochs == 0:
        logger.info(f"Skipping task {task_config.task_name} because epochs=0.")
        return

    train_loader, val_loader, test_loader = prepare_dataloaders(task_config=task_config)

    # Build Legal-BERT with classification layer
    model = LegalModel(task_config)
    trainer = LegalModelTrainer(model, task_config)

    # Train the model for specified epochs -> evaluate -> save best checkpoint
    best_weights_path = trainer.fit(train_loader, val_loader)
    if best_weights_path is None:
        logger.info(f"No training checkpoint produced for {task_config.task_name}; skipping export.")
        return
    
    # Reload the best performing model weights for the extraction phase
    logger.info(f"Reloading best model weights from {best_weights_path} for serialization...")
    model.load_state_dict(torch.load(best_weights_path, map_location=torch.device(task_config.device)))
    
    # Export predictions for downstream knowledge distillation
    SoftTargetExporter.export_all_splits(model, {"train": train_loader, "validation": val_loader, "test": test_loader}, task_config)
    logger.info(f"Task pipeline for {task_config.task_name}_{task_config.unique_id_for_dir} successfully executed.\n" + "="*80)

testers = [
    model_config.ledgar_teacher_tester,
    model_config.unfair_tos_teacher_tester,

    #model_config.ledgar_teacher_low_ressource_tester,
    #model_config.unfair_tos_teacher_low_ressource_tester,
    model_config.unfair_tos_kd_check_correct_low_ressource,
    #model_config.ledgar_kd_check_correct_low_ressource,

    model_config.unfair_tos_supervised_student_tester,

    model_config.unfair_tos_check_correct_load_preprocessed_dataset,

    model_config.unfair_tos_kd_student_tester,
    model_config.ledgar_kd_student_tester,
]

main_models = [
    # Teachers
        model_config.ledgar_teacher,
        model_config.unfair_tos_teacher,
    # Baseline Students
        model_config.ledgar_supervised_student_baseline,
        model_config.unfair_tos_supervised_student_baseline,
    # Knowledge Distillation Students
        model_config.ledgar_kd_student,
        model_config.unfair_tos_kd_student,
    # Low-resource experiments
        model_config.ledgar_supervised_student_low_resource,
        model_config.unfair_tos_supervised_student_low_resource,
        model_config.ledgar_kd_student_low_resource,
        model_config.unfair_tos_kd_student_low_resource,
    # Three-seed final experiments
        model_config.ledgar_supervised_student_final_seed_1,
        model_config.ledgar_supervised_student_final_seed_2,
        model_config.ledgar_supervised_student_final_seed_3,
        model_config.unfair_tos_supervised_student_final_seed_1,
        model_config.unfair_tos_supervised_student_final_seed_2,
        model_config.unfair_tos_supervised_student_final_seed_3,
        model_config.ledgar_kd_student_final_seed_1,
        model_config.ledgar_kd_student_final_seed_2,
        model_config.ledgar_kd_student_final_seed_3,
        model_config.unfair_tos_kd_student_final_seed_1,
        model_config.unfair_tos_kd_student_final_seed_2,
        model_config.unfair_tos_kd_student_final_seed_3,
]

models_to_run = testers

def run_pipelines() -> None:
    for config in models_to_run:
        run_task_pipeline(config)

if __name__ == "__main__":
    run_pipelines()
