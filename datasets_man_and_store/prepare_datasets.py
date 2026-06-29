import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict
from datasets_man_and_store.raw_loader import load_dataset_raw
from datasets_man_and_store.preprocess_dataset import preprocess_dataset

def prep_dataset(dataset_name: str, sample: int = 101, seed: int = 42, percent_of_data: int = 100) -> DatasetDict | Dataset:
    """
    Prepares the dataset by loading the raw data and preprocessing it.
    Args:
        dataset_name (str): Name of the dataset to prepare.
        sample (int): Index of the sample to display for verification.
        seed (int): Seed used when raw data needs to be split locally.
        percent_of_data (int): Percentage of data to use for training.
    """
    # Load raw dataset
    raw = load_dataset_raw(dataset_name, seed=seed)
    raw_dataset_dir = Path("datasets_man_and_store") / f"{dataset_name}_raw"

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

        subset_dir = raw_dataset_dir.parent / f"{raw_dataset_dir.name}_{percent_of_data}pct"
        raw.save_to_disk(str(subset_dir))
        raw_dataset_dir = subset_dir

    # Preprocess the dataset
    return preprocess_dataset(raw_dataset_dir=raw_dataset_dir, sample=sample)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["ledgar", "unfair_tos"])
    parser.add_argument("--percent-of-data", type=int, default=100)
    args = parser.parse_args()

    prep_dataset(dataset_name=args.dataset, percent_of_data=args.percent_of_data)
