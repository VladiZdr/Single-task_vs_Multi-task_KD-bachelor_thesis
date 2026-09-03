from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from torch.utils.data import DataLoader, default_collate

from configs.model_configs import ModelConfig
import configs.model_templates as model_config
from configs.model_templates_testers import single_task_testers
from configs.model_templates import (
    single_task_main_modules,
    constants_single_task_models,
    temperature_single_task_models,
    annealing_single_task_models,
)
from configs.model_templates_low_ress import low_resource_single_task_models, models_for_eval
from configs.model_templates_dif_seeds import different_seed_single_task_models
from configs.model_templates_annealing_low_res import all_singletask_submodels
from datasets_manipulation.prepare_datasets import prep_dataset_from_raw, smart_load_dataset
from single_task.legal_model import LegalModel
from single_task.legal_model_trainer import LegalModelTrainer
from datasets_manipulation.export_teacher_outputs import SoftTargetExporter

# Configure unified logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
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


# Avoid parallel CPU workers (num_workers > 0) accidentally generating identical random numbers
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_preprocessed_dataset( task_config: ModelConfig) -> tuple[HFDataset, HFDataset, HFDataset, HFDataset, HFDataset, HFDataset]:
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

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_dataset_for_export,
        val_dataset_for_export,
        test_dataset_for_export,
    )


def build_task_collate_fn(task_name: str):
    """Custom collate function that guarantees 'task' key exists in every batch."""
    def collate_fn(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
        collated = default_collate(batch)
        if "task" not in collated:
            collated["task"] = task_name
        return collated

    return collate_fn


def create_minibatches_for_training_and_export(
    task_config: ModelConfig,
    train_dataset: HFDataset,
    val_dataset: HFDataset,
    test_dataset: HFDataset,
    train_dataset_for_export: HFDataset,
    val_dataset_for_export: HFDataset,
    test_dataset_for_export: HFDataset,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, dict[str, DataLoader]]:
    pin_memory = torch.cuda.is_available()
    num_workers = getattr(task_config, "num_workers", 0)
    persistent_workers = num_workers > 0
    collate_fn = build_task_collate_fn(task_config.task_name)

    generator = torch.Generator()
    generator.manual_seed(task_config.seed)

    # Create minibatches for training
    train_loader = DataLoader(
        train_dataset,  # type: ignore
        batch_size=task_config.batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_worker,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,  # type: ignore
        batch_size=task_config.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,  # type: ignore
        batch_size=task_config.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    # Create train dataloader used for unshuffled export of teacher logits
    unshuffled_train_loader = DataLoader(
        train_dataset,  # type: ignore
        batch_size=task_config.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    # Create minibatches for export
    train_loader_export = DataLoader(
        train_dataset_for_export,  # type: ignore
        batch_size=task_config.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )
    val_loader_export = DataLoader(
        val_dataset_for_export,  # type: ignore
        batch_size=task_config.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )
    test_loader_export = DataLoader(
        test_dataset_for_export,  # type: ignore
        batch_size=task_config.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    dataloaders_for_export = {
        "train": train_loader_export,
        "validation": val_loader_export,
        "test": test_loader_export,
    }

    return train_loader, val_loader, test_loader, unshuffled_train_loader, dataloaders_for_export


def prepare_dataloaders(task_config: ModelConfig) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, dict[str, DataLoader]]:
    set_all_seeds(task_config.seed)

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_dataset_for_export,
        val_dataset_for_export,
        test_dataset_for_export,
    ) = load_preprocessed_dataset(task_config)

    # Columns required by the model, trainer, and SoftTargetExporter
    requested_cols = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "labels",
        "sample_index",
        "text",
        "task",
    ]
    if getattr(task_config, "loss_type", None) == "kldiv":
        requested_cols.append("logits")

    datasets = [
        train_dataset,
        val_dataset,
        test_dataset,
        train_dataset_for_export,
        val_dataset_for_export,
        test_dataset_for_export,
    ]

    # Preserve all available requested columns (including string columns like 'text')
    for ds in datasets:
        existing_cols = ds.column_names
        valid_cols = [c for c in requested_cols if c in existing_cols]
        ds.set_format(type="torch", columns=valid_cols)

    return create_minibatches_for_training_and_export(
        task_config,
        train_dataset,
        val_dataset,
        test_dataset,
        train_dataset_for_export,
        val_dataset_for_export,
        test_dataset_for_export,
    )


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

    train_loader, val_loader, test_loader, unshuffled_train_loader, dataloaders_for_export = prepare_dataloaders(
        task_config=task_config
    )

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
    SoftTargetExporter.export_all_splits(
        model,
        dataloaders_inference={"train": unshuffled_train_loader, "validation": val_loader, "test": test_loader},
        dataloaders_export=dataloaders_for_export,
        config=task_config,
    )
    logger.info(
        f"Task pipeline for {task_config.task_name}_{task_config.unique_id_for_dir} successfully executed.\n"
        + "=" * 160
    )


testers = single_task_testers
main_models = single_task_main_modules
constants_models = constants_single_task_models
temperature_models = temperature_single_task_models
annealing_models = annealing_single_task_models
low_resource_models = low_resource_single_task_models
low_res_annealing = all_singletask_submodels

models_to_run = low_res_annealing


def run_pipelines() -> None:
    for config in models_to_run:
        run_task_pipeline(config)


if __name__ == "__main__":
    run_pipelines()
