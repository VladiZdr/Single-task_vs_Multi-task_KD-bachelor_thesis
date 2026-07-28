import logging
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from torch.utils.data import DataLoader
from configs.model_configs import ModelConfig
import configs.model_templates as model_config
from datasets_manipulation.prepare_datasets import prep_dataset_from_raw, smart_load_dataset
from single_task.legal_model import LegalModel
from single_task.legal_model_trainer import LegalModelTrainer
from datasets_manipulation.export_teacher_outputs import SoftTargetExporter
import torch
import numpy as np
import random

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

def load_preprocessed_dataset(task_config: ModelConfig) -> tuple[HFDataset, HFDataset, HFDataset, HFDataset, HFDataset, HFDataset]:
    # Load tokenized dataset from disk or raw
    if task_config.preprocessed_data_dir == "raw":
        preprocessed_training, preprocessed_for_export = prep_dataset_from_raw(task_config)
    else:
        preprocessed_training = smart_load_dataset(task_config)
        preprocessed_for_export = preprocessed_training
    
    if isinstance(preprocessed_training, DatasetDict) and isinstance(preprocessed_for_export, DatasetDict):
        train_dataset = preprocessed_training["train"]
        val_dataset = preprocessed_training["validation"]
        test_dataset = preprocessed_training["test"]
    
        train_dataset_for_export = preprocessed_for_export["train"]
        val_dataset_for_export = preprocessed_for_export["validation"]
        test_dataset_for_export = preprocessed_for_export["test"]
    elif isinstance(preprocessed_training, HFDataset):
        raise ValueError(
                f"prep_dataset('{task_config.task_name}') returned a single Dataset, "
                "but this pipeline expects a DatasetDict with train/validation/test splits."
            )
    else:
        raise TypeError(f"Unexpected dataset type: {type(preprocessed_training)}")
    
    return train_dataset, val_dataset, test_dataset, train_dataset_for_export, val_dataset_for_export, test_dataset_for_export

def create_minibatches_for_taining_and_export(task_config: ModelConfig, train_dataset, val_dataset, test_dataset, train_dataset_for_export, val_dataset_for_export, test_dataset_for_export):
    # Create minibatches for training
    generator = torch.Generator()
    generator.manual_seed(task_config.seed)
    train_loader = DataLoader(train_dataset, batch_size=task_config.batch_size, shuffle=True, #type: ignore
        generator=generator,        #Tells the DataLoader to run seeding function exactly once 
        worker_init_fn=seed_worker, #inside each worker process right when it boots up, isolating their random states. 
    ) 
    val_loader = DataLoader(val_dataset, batch_size=task_config.batch_size, shuffle=False)    # type: ignore
    test_loader = DataLoader(test_dataset, batch_size=task_config.batch_size, shuffle=False)  # type: ignore
    
    # Create train dataloader used for unshuffled export of teacher logits
    unshuffled_train_loader = DataLoader(train_dataset, batch_size=task_config.batch_size, shuffle=False)    # type: ignore
    
    # Create minibatches for export
    train_loader_export = DataLoader(train_dataset_for_export, batch_size=task_config.batch_size, shuffle=False)    # type: ignore
    val_loader_export = DataLoader(val_dataset_for_export, batch_size=task_config.batch_size, shuffle=False)        # type: ignore
    test_loader_export = DataLoader(test_dataset_for_export, batch_size=task_config.batch_size, shuffle=False)      # type: ignore
    dataloaders_for_export = {"train": train_loader_export, "validation": val_loader_export, "test": test_loader_export}

    return train_loader, val_loader, test_loader, unshuffled_train_loader, dataloaders_for_export

def prepare_dataloaders(task_config: ModelConfig) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, dict]:
    set_all_seeds(task_config.seed)
    
    train_dataset, val_dataset, test_dataset, train_dataset_for_export, val_dataset_for_export, test_dataset_for_export = load_preprocessed_dataset(task_config)

    # Force Torch formatting
    cols = ["input_ids", "attention_mask", "token_type_ids", "labels", "task", "sample_index"]
    if task_config.model_name_or_path == "nlpaueb/legal-bert-base-uncased":
        cols.append("text")
    if task_config.loss_type == 'kldiv':
        cols.append("logits")
    train_dataset.set_format(type="torch", columns=cols)
    val_dataset.set_format(type="torch", columns=cols)
    test_dataset.set_format(type="torch", columns=cols)
    train_dataset_for_export.set_format(type="torch", columns=cols)
    val_dataset_for_export.set_format(type="torch", columns=cols)
    test_dataset_for_export.set_format(type="torch", columns=cols)

    train_loader, val_loader, test_loader, unshuffled_train_loader, dataloaders_for_export = create_minibatches_for_taining_and_export(task_config, train_dataset, val_dataset, test_dataset, train_dataset_for_export, val_dataset_for_export, test_dataset_for_export)
    return train_loader, val_loader, test_loader, unshuffled_train_loader, dataloaders_for_export

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

    train_loader, val_loader, test_loader, unshuffled_train_loader, dataloaders_for_export = prepare_dataloaders(task_config=task_config)

    # Build Legal-BERT with classification layer
    model = LegalModel(task_config)
    trainer = LegalModelTrainer(model)

    # Train the model for specified epochs -> evaluate -> save best checkpoint
    best_weights_path = trainer.fit(train_loader, val_loader)
    if best_weights_path is None:
        logger.info(f"No training checkpoint produced for {task_config.task_name}; skipping export.")
        return
    
    # Reload the best performing model weights for the extraction phase
    logger.info(f"Reloading best model weights from {best_weights_path} for serialization...")
    model.load_state_dict(torch.load(best_weights_path, map_location=torch.device(task_config.device)))

    # Export predictions for downstream knowledge distillation
    SoftTargetExporter.export_all_splits(model, 
                                         dataloaders_inference = {"train": unshuffled_train_loader, "validation": val_loader, "test": test_loader},
                                         dataloaders_export = dataloaders_for_export,
                                         config = task_config)
    logger.info(f"Task pipeline for {task_config.task_name}_{task_config.unique_id_for_dir} successfully executed.\n" + "="*160)

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
