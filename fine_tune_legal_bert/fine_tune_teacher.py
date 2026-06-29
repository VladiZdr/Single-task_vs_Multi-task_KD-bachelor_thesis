import logging
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from torch.utils.data import DataLoader
from configs.teacher_config import TeacherConfig
from datasets_man_and_store.prepare_datasets import prep_dataset
from fine_tune_legal_bert.teacher_model import TeacherModel
from fine_tune_legal_bert.legal_model_trainer import LegalModelTrainer
from fine_tune_legal_bert.export_teacher_outputs import SoftTargetExporter
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

def run_task_pipeline(task_config: TeacherConfig) -> None:
    logger.info(f"Initializing optimization pipeline for task: {task_config.task_name.upper()}")
    if task_config.epochs == 0:
        logger.info(f"Skipping task {task_config.task_name} because epochs=0.")
        return

    set_all_seeds(task_config.seed)
    
    # Load tokenized datasets
    preprocessed = prep_dataset(task_config.task_name, seed=task_config.seed)
    
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

    # Build Legal-BERT with classification layer
    model = TeacherModel(task_config)
    
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
    logger.info(f"Task pipeline for {task_config.task_name} successfully executed.\n" + "="*80)

def main() -> None:
    # 1. Pipeline Definition for LEDGAR Provision Classification
    ledgar_config = TeacherConfig(
        task_name="ledgar",
        num_labels=100,
        problem_type="single_label",
        loss_type="cross_entropy",
        batch_size=32,
        epochs=0,
        learning_rate=2e-5
    )
    
    # 2. Pipeline Definition for UNFAIR-ToS Terms Identification
    unfair_tos_config = TeacherConfig(
        task_name="unfair_tos",
        num_labels=8,
        problem_type="multi_label",
        loss_type="bce_with_logits",
        batch_size=4,
        num_of_batches=100,  # Limit to 100 batches for quicker testing (influences only training, evaluation and export will still process all batches)
        epochs=1,
        learning_rate=3e-5
    )
    
    # Run the configurations sequentially
    for config in [ledgar_config, unfair_tos_config]:
        run_task_pipeline(config)

if __name__ == "__main__":
    main()
