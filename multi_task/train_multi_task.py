from __future__ import annotations
import logging
import random
from typing import Dict
import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from torch.utils.data import DataLoader
from configs.model_config import ModelConfig
from datasets_manipulation.prepare_datasets import prep_dataset_from_raw, smart_load_dataset
from multi_task.multi_task_model import MultiTaskModel
from multi_task.multi_task_trainer import MultiTaskTrainer
import configs.model_templates as model_config

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

# It ensures each independent background worker process gets its own distinct, reproducible random seed.
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _load_split_dataloaders(task_config: ModelConfig) -> Dict[str, DataLoader]:
    if task_config.preprocessed_data_dir == "raw":
        preprocessed = prep_dataset_from_raw(dataset_name=task_config.task_name, seed=task_config.seed, percent_of_data=task_config.percent_of_data)
    else:
        preprocessed = smart_load_dataset(task_config.preprocessed_data_dir)

    # It expects a standard Hugging Face dictionary containing split tables (train, validation, test)
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

    # Injects a programmatic tracking token inline. This loops through the datasets and adds a text column filled with the task name (e.g., "ledgar"), 
    # so the multi-task model knows which classification head to use during training.
    def attach_task_column(dataset):
        if "task" in dataset.column_names:
            return dataset
        return dataset.add_column("task", [task_config.task_name] * len(dataset))

    train_dataset = attach_task_column(train_dataset)
    val_dataset = attach_task_column(val_dataset)
    test_dataset = attach_task_column(test_dataset)

    # Selects the required data columns.
    cols = ["input_ids", "attention_mask", "token_type_ids", "labels", "task"]
    if task_config.loss_type == "kldiv":
        cols.append("logits")

    # Changes the dataset output format, transforming Hugging Face text storage spaces directly into active PyTorch Tensors
    train_dataset.set_format(type="torch", columns=cols)
    val_dataset.set_format(type="torch", columns=cols)
    test_dataset.set_format(type="torch", columns=cols)

    # Instantiates a standalone PyTorch random sampling generation object tied down strictly to your project seed.
    generator = torch.Generator()
    generator.manual_seed(task_config.seed)

    # Wraps the structured datasets into iterable PyTorch streaming objects (DataLoader). 
    # The training data is randomized using locked seed generator, while validation and testing data stream through sequentially (shuffle=False).
    train_loader = DataLoader(
        train_dataset,                      # type: ignore
        batch_size=task_config.batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(val_dataset, batch_size=task_config.batch_size, shuffle=False)      # type: ignore
    test_loader = DataLoader(test_dataset, batch_size=task_config.batch_size, shuffle=False)    # type: ignore

    return {"train": train_loader, "validation": val_loader, "test": test_loader}

# Collects and organizes data streams. It runs data loader generator for both tasks and regroups them into multi-task dictionaries sorted by split type 
# (all training loaders together, all validation loaders together, etc.).
def prepare_multitask_dataloaders(ledgar_config: ModelConfig, unfair_tos_config: ModelConfig) -> tuple[dict[str, DataLoader], dict[str, DataLoader], dict[str, DataLoader]]:
    set_all_seeds(ledgar_config.seed)

    ledgar_loaders = _load_split_dataloaders(ledgar_config)
    unfair_loaders = _load_split_dataloaders(unfair_tos_config)

    train_loaders = {"ledgar": ledgar_loaders["train"], "unfair_tos": unfair_loaders["train"]}
    val_loaders = {"ledgar": ledgar_loaders["validation"], "unfair_tos": unfair_loaders["validation"]}
    test_loaders = {"ledgar": ledgar_loaders["test"], "unfair_tos": unfair_loaders["test"]}

    return train_loaders, val_loaders, test_loaders

def run_multitask_pipeline(ledgar_config: ModelConfig, unfair_tos_config: ModelConfig) -> None:
    logger.info(
        "Initializing multi-task pipeline with LEDGAR as the leading task, %s epochs, and LEDGAR batch size %s.",
        ledgar_config.epochs,
        ledgar_config.batch_size,
    )

    if ledgar_config.epochs == 0:
        logger.info("Skipping multi-task pipeline because epochs=0.")
        return

    train_loaders, val_loaders, test_loaders = prepare_multitask_dataloaders(ledgar_config, unfair_tos_config)

    model = MultiTaskModel(ledgar_config, unfair_tos_config)
    trainer = MultiTaskTrainer(model, ledgar_config, unfair_tos_config)

    # If the model completes training and saves its parameters, it returns the disk location path. If no file is generated, it stops early.
    best_weights_path = trainer.fit(train_loaders, val_loaders)
    if best_weights_path is None:
        logger.info("No multi-task checkpoint produced; skipping test evaluation.")
        return

    # Testing evaluation. It loads the optimal saved weights back into the model architecture from disk and runs a final validation check across the untouched testing datasets.
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

    logger.info("Multi-task pipeline successfully executed.\n" + "=" * 80)


# Bundles paired task configuration objects into a structured execution queue array list.
models_to_run = [
    model_config.multi_task_kd_model_tester,
    model_config.multi_task_supervised_model_tester,
    #model_config.multi_task_supervised_model,
    #model_config.multi_task_kd_model
]

def run_multitask_pipelines() -> None:
    for model in models_to_run:
        run_multitask_pipeline(model.ledgar_config, model.unfair_tos_config)


if __name__ == "__main__":
    run_multitask_pipelines()
