from __future__ import annotations
import logging
import random
from typing import Dict
import numpy as np
import torch
from datasets import Dataset
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from torch.utils.data import DataLoader
from configs.model_config import ModelConfig, MultiTaskModelConfig
from datasets_manipulation.prepare_datasets import (
    get_torch_columns_for_split,
    prep_dataset_from_raw,
    sample_low_resource_dataset,
    sample_percent_dataset_for_testing,
    smart_load_dataset,
    align_dataset_tokenization
)
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

def _load_split_dataloaders(task_config: ModelConfig) -> dict[str, DataLoader]:
    set_all_seeds(task_config.seed)

    if task_config.preprocessed_data_dir == "raw":
        preprocessed = prep_dataset_from_raw(task_config)
    else:
        preprocessed = smart_load_dataset(task_config)

    if isinstance(preprocessed, DatasetDict):
        pass
    elif isinstance(preprocessed, HFDataset):
        raise ValueError(
            f"prep_dataset('{task_config.task_name}') returned a single Dataset, "
            "but this pipeline expects train/validation/test splits."
        )
    else:
        raise TypeError(f"Unexpected dataset type: {type(preprocessed)}")

    # 1. Align standard input_ids to target model view
    if hasattr(task_config, "model_name_or_path") and task_config.model_name_or_path:
        preprocessed = align_dataset_tokenization(preprocessed, task_config.model_name_or_path)

    # 2. Attach task metadata string column
    def attach_task_column(dataset: Dataset) -> Dataset:
        if "task" in dataset.column_names:
            return dataset
        return dataset.add_column("task", [task_config.task_name] * len(dataset))

    generator = torch.Generator()
    generator.manual_seed(task_config.seed)

    is_kldiv = getattr(task_config, "loss_type", None) == "kldiv"
    loaders = {}

    for split_name in ["train", "validation", "test"]:
        if split_name not in preprocessed:
            continue

        split_ds = attach_task_column(preprocessed[split_name])                     #type:ignore

        # 3. Determine columns per split (only request logits if actually present)
        split_has_logits = is_kldiv and ("logits" in split_ds.column_names)
        
        # Exclude 'task' string column from PyTorch tensor casting!
        torch_cols = get_torch_columns_for_split(
            split_ds,
            include_task=False, 
            include_logits=split_has_logits,
        )

        # Format numeric/tokenizer columns to torch tensors while leaving metadata as python objects
        split_ds.set_format(type="torch", columns=torch_cols, output_all_columns=True)

        is_train = split_name == "train"
        loaders[split_name] = DataLoader(
            split_ds, # type: ignore
            batch_size=task_config.batch_size,
            shuffle=is_train,
            generator=generator if is_train else None,
            worker_init_fn=seed_worker if is_train else None,
        )

    return loaders

def prepare_multitask_dataloaders(
    ledgar_config: ModelConfig, 
    unfair_tos_config: ModelConfig
) -> tuple[dict[str, DataLoader], dict[str, DataLoader], dict[str, DataLoader]]:
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

    model = MultiTaskModel(ledgar_config=ledgar_config, unfair_tos_config=unfair_tos_config, unique_id_for_dir=unique_id_for_dir)
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

    logger.info("Multi-task pipeline successfully executed for %s.\n" + "=" * 80, unique_id_for_dir)

testers = [
    model_config.multi_task_kd_model_tester,
    model_config.multi_task_supervised_model_tester,
    #model_config.multi_task_check_low_resource,
]
main_models = [
    model_config.multi_task_supervised_model,
    model_config.multi_task_kd_model,
    model_config.multi_task_supervised_model_low_resource,
    model_config.multi_task_kd_model_low_resource,
    model_config.multi_task_supervised_model_final_seed_1,
    model_config.multi_task_supervised_model_final_seed_2,
    model_config.multi_task_supervised_model_final_seed_3,
    model_config.multi_task_kd_model_final_seed_1,
    model_config.multi_task_kd_model_final_seed_2,
    model_config.multi_task_kd_model_final_seed_3,
]

# Bundles paired task configuration objects into a structured execution queue array list.
models_to_run = testers

def run_multitask_pipelines() -> None:
    for model in models_to_run:
        run_multitask_pipeline(model)


if __name__ == "__main__":
    run_multitask_pipelines()
