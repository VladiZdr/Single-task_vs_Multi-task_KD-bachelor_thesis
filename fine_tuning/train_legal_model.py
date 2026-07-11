import logging
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from torch.utils.data import DataLoader
from configs.model_config import ModelConfig
import configs.model_templates as model_config
from datasets_manipulation.prepare_datasets import prep_dataset_from_raw, smart_load_dataset
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
    
    # Load tokenized from disk if available, otherwise preprocess from raw data
    if task_config.preprocessed_data_dir == "raw":
        preprocessed = prep_dataset_from_raw(dataset_name=task_config.task_name, seed=task_config.seed, percent_of_data=task_config.percent_of_data)
    # If the preprocessed dataset is already available, load it directly from disk 
    # (smart_load_dataset handles both Hugging Face DatasetDict and Datasets of .safetensors)
    else:
        preprocessed = smart_load_dataset(task_config.preprocessed_data_dir)

    if isinstance(preprocessed, DatasetDict):
        train_dataset = preprocessed["train"]
        val_dataset = preprocessed["validation"]
        test_dataset = preprocessed["test"]
    elif isinstance(preprocessed, HFDataset):
        raise ValueError(
            f"prep_dataset('{task_config.task_name}') returned a single Dataset, "
            "but this pipeline expects a DatasetDict with train/validation/test splits."
        )
    else:
        raise TypeError(f"Unexpected dataset type: {type(preprocessed)}")

    # Force Torch formatting
    cols = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    # Extract teacher logits when Knowledge Distillation is active
    if task_config.loss_type == 'kldiv':
        cols.append("logits")
    train_dataset.set_format(type="torch", columns=cols)
    val_dataset.set_format(type="torch", columns=cols)
    test_dataset.set_format(type="torch", columns=cols)

    # Create minibatches
    generator = torch.Generator()
    generator.manual_seed(task_config.seed)
    train_loader = DataLoader(train_dataset, batch_size=task_config.batch_size, shuffle=True, #type: ignore
        generator=generator,        #Tells the DataLoader to run seeding function exactly once 
        worker_init_fn=seed_worker, #inside each worker process right when it boots up, isolating their random states. 
    ) 
    val_loader = DataLoader(val_dataset, batch_size=task_config.batch_size, shuffle=False)    # type: ignore
    test_loader = DataLoader(test_dataset, batch_size=task_config.batch_size, shuffle=False)  # type: ignore

    return train_loader, val_loader, test_loader

def run_task_pipeline(task_config: ModelConfig) -> None:
    logger.info(f"Initializing optimization pipeline for task: {task_config.task_name.upper()}_{task_config.unique_id_for_dir} with {task_config.epochs} epochs, {task_config.batch_size} batch size, and {task_config.percent_of_data}% of the dataset.")
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


models_to_run = [
    # Testers
        #model_config.ledgar_teacher_tester,
        #model_config.unfair_tos_teacher_tester,
        #model_config.unfair_tos_supervised_student_tester,
        #model_config.unfair_tos_check_correct_load_preprocessed_dataset,
        #model_config.unfair_tos_kd_student_tester,
        #model_config.ledgar_kd_student_tester,

    # Teachers
        model_config.ledgar_teacher,
        model_config.unfair_tos_teacher,
    # Baseline Students
        model_config.ledgar_supervised_student_baseline,
        model_config.unfair_tos_supervised_student_baseline,
    # Knowledge Distillation Students
        model_config.ledgar_kd_student,
        model_config.unfair_tos_kd_student
    ]

def run_pipelines() -> None:
    for config in models_to_run:
        run_task_pipeline(config)

if __name__ == "__main__":
    run_pipelines()
