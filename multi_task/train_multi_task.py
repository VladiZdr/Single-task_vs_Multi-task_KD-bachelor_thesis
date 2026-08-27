from __future__ import annotations

import logging
import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from torch.utils.data import DataLoader, default_collate

from configs.model_configs import ModelConfig, MultiTaskModelConfig
from configs.model_templates import (
    multi_task_main_modules,
    constants_multi_task_models,
    temperature_multi_task_models,
    annealing_multi_task_models,
)
from configs.model_templates_low_ress import low_resource_multi_task_models
from configs.model_templates_dif_seeds import different_seed_multi_task_models
from configs.model_templates_testers import multi_task_testers
from datasets_manipulation.prepare_datasets import prep_dataset_from_raw, smart_load_dataset
from multi_task.multi_task_model import MultiTaskModel
from multi_task.multi_task_trainer import MultiTaskTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("MultiTaskFineTunePipeline")


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Turning on deterministic = True and disabling benchmark ensures that neural network math operations yield identical, 100% reproducible results across different training runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Ensures each independent background worker process gets its own distinct, reproducible random seed.
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Injects a programmatic tracking token inline so the multi-task model knows which classification head to use during training.
def attach_task_column(dataset: HFDataset, task_config: ModelConfig) -> HFDataset:
    if "task" in dataset.column_names:
        return dataset
    return dataset.add_column("task", [task_config.task_name] * len(dataset))


def get_dataset_splits(task_config: ModelConfig) -> Tuple[HFDataset, HFDataset, HFDataset]:
    if task_config.preprocessed_data_dir == "raw":
        preprocessed, _ = prep_dataset_from_raw(task_config)
    else:
        preprocessed = smart_load_dataset(task_config)

    # Expects a standard Hugging Face dictionary containing split tables (train, validation, test)
    if isinstance(preprocessed, DatasetDict):
        train_dataset = preprocessed["train"]
        val_dataset = preprocessed["validation"]
        test_dataset = preprocessed["test"]
    elif isinstance(preprocessed, HFDataset):
        raise ValueError(
            f"prep_dataset('{task_config.task_name}') returned a single Dataset, but this pipeline expects train/validation/test splits."
        )
    else:
        raise TypeError(f"Unexpected dataset type: {type(preprocessed)}")

    return train_dataset, val_dataset, test_dataset


def format_splits(
    train_dataset: HFDataset,
    val_dataset: HFDataset,
    test_dataset: HFDataset,
    task_config: ModelConfig,
) -> None:
    # Explicit numerical columns allowed for PyTorch Tensor conversion
    tensor_candidates = {"input_ids", "attention_mask", "token_type_ids", "labels", "logits", "sample_index"}

    requested_cols = ["input_ids", "attention_mask", "token_type_ids", "labels", "sample_index"]
    if getattr(task_config, "loss_type", None) == "kldiv":
        requested_cols.append("logits")

    # Safely apply torch format ONLY to numerical columns that exist in each split
    for ds in (train_dataset, val_dataset, test_dataset):
        existing_cols = ds.column_names
        valid_tensor_cols = [c for c in requested_cols if c in existing_cols and c in tensor_candidates]
        ds.set_format(type="torch", columns=valid_tensor_cols)


def build_task_collate_fn(task_name: str):
    """Custom collate function that guarantees 'task' key exists in every collated batch."""
    def collate_fn(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
        collated = default_collate(batch)
        if "task" not in collated:
            collated["task"] = [task_name] * len(batch)
        return collated

    return collate_fn


def _load_split_dataloaders(task_config: ModelConfig) -> Dict[str, DataLoader]:
    set_all_seeds(task_config.seed)

    train_dataset, val_dataset, test_dataset = get_dataset_splits(task_config)

    train_dataset = attach_task_column(train_dataset, task_config)
    val_dataset = attach_task_column(val_dataset, task_config)
    test_dataset = attach_task_column(test_dataset, task_config)

    format_splits(train_dataset, val_dataset, test_dataset, task_config)

    pin_memory = torch.cuda.is_available()
    num_workers = getattr(task_config, "num_workers", 0)
    persistent_workers = num_workers > 0
    collate_fn = build_task_collate_fn(task_config.task_name)

    # Instantiates a standalone PyTorch random sampling generation object tied down strictly to your project seed.
    generator = torch.Generator()
    generator.manual_seed(task_config.seed)

    # Wraps the structured datasets into iterable PyTorch streaming objects (DataLoader).
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

    return {"train": train_loader, "validation": val_loader, "test": test_loader}


def prepare_multitask_dataloaders(
    ledgar_config: ModelConfig, unfair_tos_config: ModelConfig
) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader], Dict[str, DataLoader]]:
    set_all_seeds(ledgar_config.seed)

    ledgar_loaders = _load_split_dataloaders(ledgar_config)
    unfair_loaders = _load_split_dataloaders(unfair_tos_config)

    train_loaders = {"ledgar": ledgar_loaders["train"], "unfair_tos": unfair_loaders["train"]}
    val_loaders = {"ledgar": ledgar_loaders["validation"], "unfair_tos": unfair_loaders["validation"]}
    test_loaders = {"ledgar": ledgar_loaders["test"], "unfair_tos": unfair_loaders["test"]}

    return train_loaders, val_loaders, test_loaders


def run_multitask_pipeline(multitask_model_config: MultiTaskModelConfig) -> None:
    unique_id_for_dir = multitask_model_config.unique_id_for_dir
    ledgar_config = multitask_model_config.ledgar_config
    unfair_tos_config = multitask_model_config.unfair_tos_config

    logger.info(
        "Initializing multi-task pipeline for %s with Round-Robin, %s epochs.",
        unique_id_for_dir,
        ledgar_config.epochs,
    )

    if ledgar_config.epochs == 0:
        logger.info("Skipping multi-task pipeline because epochs=0.")
        return

    train_loaders, val_loaders, test_loaders = prepare_multitask_dataloaders(ledgar_config, unfair_tos_config)

    model = MultiTaskModel(multitask_model_config)
    trainer = MultiTaskTrainer(model)

    # If the model completes training and saves its parameters, it returns the disk location path.
    best_weights_path = trainer.fit(train_loaders, val_loaders)
    if best_weights_path is None:
        logger.info("No multi-task checkpoint produced; skipping test evaluation.")
        return

    # Testing evaluation: reload optimal saved weights back into the model architecture
    logger.info(f"Reloading best model weights from {best_weights_path} for test evaluation...")
    model.load_state_dict(torch.load(best_weights_path, map_location=torch.device(ledgar_config.device)))
    test_metrics = trainer.evaluate(test_loaders)

    logger.info(
        "Test Loss: %.4f | Test Macro-F1: %.4f | Test Micro-F1: %.4f | LEDGAR Macro-F1: %.4f | UNFAIR-ToS Macro-F1: %.4f",
        test_metrics["loss"],
        test_metrics["macro_f1"],
        test_metrics["micro_f1"],
        test_metrics.get("ledgar_macro_f1", 0.0),
        test_metrics.get("unfair_tos_macro_f1", 0.0),
    )

    logger.info("Multi-task pipeline successfully executed for %s.\n" + "=" * 80, unique_id_for_dir)


testers = multi_task_testers

main_models = multi_task_main_modules
constants_models = constants_multi_task_models
temperature_models = temperature_multi_task_models
annealing_models = annealing_multi_task_models
low_resource_models = low_resource_multi_task_models

models_to_run = different_seed_multi_task_models


def run_multitask_pipelines() -> None:
    for model in models_to_run:
        run_multitask_pipeline(model)


if __name__ == "__main__":
    run_multitask_pipelines()
