import shutil
import os
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict
from datasets_manipulation.raw_loader import load_dataset_raw
from datasets_manipulation.preprocess_dataset import preprocess_dataset, _load_valid_dataset_dict
from safetensors.torch import load_file
import numpy as np
from sklearn.model_selection import train_test_split


def _get_label_column(dataset: Dataset) -> str:
    if "labels" in dataset.column_names:
        return "labels"
    if "label" in dataset.column_names:
        return "label"
    raise ValueError(f"Expected dataset to contain a label column, got columns: {dataset.column_names}")


def _sample_ledgar_train_split(train_dataset: Dataset, low_resource_percent: int, seed: int) -> Dataset:
    label_column = _get_label_column(train_dataset)
    sample_size = max(1, int(len(train_dataset) * (low_resource_percent / 100)))
    if sample_size >= len(train_dataset):
        return train_dataset

    labels = np.asarray(train_dataset[label_column])
    indices = np.arange(len(train_dataset))

    selected_indices, _ = train_test_split(
        indices,
        train_size=sample_size,
        random_state=seed,
        stratify=labels,
    )

    return train_dataset.select(sorted(selected_indices.tolist()))


def _multi_hot_labels(labels: list, num_labels: int = 8) -> np.ndarray:
    if not labels:
        return np.zeros((0, num_labels), dtype=np.int64)

    first_label = labels[0]
    if isinstance(first_label, list) and len(first_label) == num_labels and all(label in (0, 1) for label in first_label):
        return np.asarray(labels, dtype=np.int64)

    encoded = np.zeros((len(labels), num_labels), dtype=np.int64)
    for row_index, label_ids in enumerate(labels):
        for label_id in label_ids:
            encoded[row_index, label_id] = 1
    return encoded


def _sample_unfair_tos_train_split(train_dataset: Dataset, low_resource_percent: int, seed: int) -> Dataset:
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError as error:
        raise ImportError(
            "UNFAIR-ToS low-resource sampling requires iterative-stratification. "
            "Install it with: pip install iterative-stratification"
        ) from error

    label_column = _get_label_column(train_dataset)
    sample_size = max(1, int(len(train_dataset) * (low_resource_percent / 100)))
    if sample_size >= len(train_dataset):
        return train_dataset

    labels = _multi_hot_labels(train_dataset[label_column])
    features = np.zeros((len(train_dataset), 1))

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        train_size=sample_size,
        random_state=seed,
    )
    selected_indices, _ = next(splitter.split(features, labels))

    return train_dataset.select(sorted(selected_indices.tolist()))


def _sample_low_resource_train_split(raw: DatasetDict, dataset_name: str, low_resource_percent: int, seed: int) -> DatasetDict:
    if "train" not in raw:
        raise ValueError("Low-resource sampling requires a DatasetDict with a train split.")

    if dataset_name == "ledgar":
        sampled_train = _sample_ledgar_train_split(raw["train"], low_resource_percent, seed)
    elif dataset_name == "unfair_tos":
        sampled_train = _sample_unfair_tos_train_split(raw["train"], low_resource_percent, seed)
    else:
        raise ValueError(f"Unsupported low-resource sampling dataset: {dataset_name}")

    return DatasetDict(
        {
            split: sampled_train if split == "train" else dataset
            for split, dataset in raw.items()
        }
    )


def sample_low_resource_dataset(dataset: DatasetDict, dataset_name: str, low_resource_percent: int, seed: int) -> DatasetDict:
    if low_resource_percent >= 100:
        return dataset
    if low_resource_percent not in (1, 10, 25, 50):
        raise ValueError(
            f"low_resource_percent must be one of 1, 10, 25, 50, or 100, got {low_resource_percent}"
        )

    return _sample_low_resource_train_split(dataset, dataset_name, low_resource_percent, seed)

def sample_percent_dataset_for_testing(dataset: DatasetDict | Dataset, percent_of_data: int) -> DatasetDict | Dataset:
    if percent_of_data >= 100:
        return dataset
    if percent_of_data <= 0:
        raise ValueError(f"percent_of_data must be positive, got {percent_of_data}")

    def sample_split(split: Dataset) -> Dataset:
        sample_size = max(1, int(len(split) * (percent_of_data / 100)))
        return split.select(range(sample_size))

    if isinstance(dataset, DatasetDict):
        return DatasetDict({split_name: sample_split(split) for split_name, split in dataset.items()})

    return sample_split(dataset)


def _raw_dataset_dir(dataset_name: str, percent_of_data: int, low_resource_percent: int, seed: int) -> Path:
    if percent_of_data >= 100 and low_resource_percent >= 100:
        return Path("datasets_store") / f"{dataset_name}_raw"

    name_parts = [f"{dataset_name}_raw"]
    if percent_of_data < 100:
        name_parts.append(f"test{percent_of_data}pct")
    if low_resource_percent < 100:
        name_parts.append(f"low{low_resource_percent}pct_seed{seed}")

    return Path("datasets_store") / "_".join(name_parts)


def prep_dataset_from_raw(dataset_name: str, sample: int = 0, seed: int = 42, percent_of_data: int = 100, low_resource_percent: int = 100) -> DatasetDict | Dataset:
    """
    Prepares the dataset by loading the raw data and preprocessing it.
    Args:
        dataset_name (str): Name of the dataset to prepare.
        sample (int): Index of the sample to display for verification.
        seed (int): Seed used when raw data needs to be split locally.
        percent_of_data (int): Percentage of every split to keep for quick tests.
        low_resource_percent (int): Stratified percentage of the train split to keep for low-resource runs.
    """
    # Load raw dataset
    raw = load_dataset_raw(dataset_name, seed=seed)

    raw = sample_percent_dataset_for_testing(raw, percent_of_data)

    if low_resource_percent < 100:
        if not isinstance(raw, DatasetDict):
            raise ValueError(f"Expected a DatasetDict for low-resource sampling, got: {type(raw).__name__}")
        raw = sample_low_resource_dataset(raw, dataset_name, low_resource_percent, seed)

    raw_dataset_dir = _raw_dataset_dir(dataset_name, percent_of_data, low_resource_percent, seed)

    path_preprocessed = str(raw_dataset_dir).replace("_raw", "_preprocessed")
    if os.path.exists(path_preprocessed):
        shutil.rmtree(path_preprocessed)

    if percent_of_data < 100 or low_resource_percent < 100:
        if raw_dataset_dir.exists():
            shutil.rmtree(raw_dataset_dir)
        raw.save_to_disk(str(raw_dataset_dir))

    # Preprocess the dataset
    return preprocess_dataset(raw_dataset_dir=raw_dataset_dir, sample=sample)

def load_teacher_safetensors_to_datasetdict(data_dir: str) -> DatasetDict:
    """Loads split safetensor files into a Hugging Face DatasetDict."""
    splits = ["train", "validation", "test"] 
    dataset_dict = {}
    
    for split in splits:
        file_path = os.path.join(data_dir, f"teacher_{split}_outputs.safetensors")
        if os.path.exists(file_path):
            # 1. Load the tensors
            tensors_dict = load_file(file_path)
            
            # 2. Convert to Hugging Face Dataset (converting to numpy first avoids memory duplication warnings)
            dataset_dict[split] = Dataset.from_dict({k: v.numpy() for k, v in tensors_dict.items()})
            
    if not dataset_dict:
        raise FileNotFoundError(f"No teacher output .safetensors files found in {data_dir}")
        
    return DatasetDict(dataset_dict)

def smart_load_dataset(task_config) -> DatasetDict:
    """
    Dynamically detects the dataset format on disk and loads it 
    using the appropriate strategy.
    """
    data_dir = task_config.preprocessed_data_dir
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"The directory '{data_dir}' does not exist.")

    # Check if the directory contains teacher safetensors files
    has_teacher_safetensors = any(
        f.startswith("teacher_") and f.endswith(".safetensors") 
        for f in os.listdir(data_dir)
    )

    if has_teacher_safetensors:
        # Not a standard Hugging Face dataset directory layout
        preprocessed = load_teacher_safetensors_to_datasetdict(data_dir)
    else:
        # Fall back to standard Hugging Face loading
        preprocessed = _load_valid_dataset_dict(data_dir)
    
    if task_config.percent_of_data < 100:
        preprocessed = sample_percent_dataset_for_testing(preprocessed, task_config.percent_of_data)

    if not isinstance(preprocessed, DatasetDict):
        raise ValueError(f"Expected a DatasetDict after loading {task_config.preprocessed_data_dir}.")
    if task_config.low_resource_percent < 100:
        preprocessed = sample_low_resource_dataset(preprocessed, task_config.task_name, task_config.low_resource_percent, task_config.seed)

    return preprocessed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["ledgar", "unfair_tos"])
    parser.add_argument("--percent-of-data", type=int, default=100)
    parser.add_argument("--low-resource-percent", type=int, default=100)
    args = parser.parse_args()

    prep_dataset_from_raw(
        dataset_name=args.dataset,
        percent_of_data=args.percent_of_data,
        low_resource_percent=args.low_resource_percent,
    )
