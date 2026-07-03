import shutil
import os
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict
from datasets_manipulation.raw_loader import load_dataset_raw
from datasets_manipulation.preprocess_dataset import preprocess_dataset
from safetensors.torch import load_file

def prep_dataset_from_raw(dataset_name: str, sample: int = 0, seed: int = 42, percent_of_data: int = 100) -> DatasetDict | Dataset:
    """
    Prepares the dataset by loading the raw data and preprocessing it.
    Args:
        dataset_name (str): Name of the dataset to prepare.
        sample (int): Index of the sample to display for verification.
        seed (int): Seed used when raw data needs to be split locally.
        percent_of_data (int): Percentage of data to use for training.
    """
    # Delete the raw dataset directory if it exists to ensure a fresh start
    path_raw = f"datasets_store/{dataset_name}_raw"
    if os.path.exists(path_raw):
        shutil.rmtree(path_raw)
    path_preprocessed = f"datasets_store/{dataset_name}_preprocessed"
    if os.path.exists(path_preprocessed):
        shutil.rmtree(path_preprocessed)

    # Load raw dataset
    raw = load_dataset_raw(dataset_name, seed=seed)

    raw_dataset_dir = Path("datasets_store") / f"{dataset_name}_raw"

    # If percent_of_data is less than 100, we select a subset of the dataset for quicker testing.
    if percent_of_data < 100:
        if percent_of_data <= 0:
            raise ValueError(f"percent_of_data must be positive, got {percent_of_data}")

        # Check if it's a DatasetDict (has multiple splits like train, test)
        if isinstance(raw, DatasetDict):
            raw = DatasetDict(
                {
                    split: raw[split].select(range(max(1, int(len(raw[split]) * (percent_of_data / 100)))))
                    for split in raw.keys()
                }
            )
        else:
            # If it's a single Dataset
            raw = raw.select(range(max(1, int(len(raw) * (percent_of_data / 100)))))

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["ledgar", "unfair_tos"])
    parser.add_argument("--percent-of-data", type=int, default=100)
    args = parser.parse_args()

    prep_dataset_from_raw(dataset_name=args.dataset, percent_of_data=args.percent_of_data)
