import argparse
from datasets import Dataset, DatasetDict
from datasets_man_and_store.raw_loader import load_dataset_raw
from datasets_man_and_store.preprocess_dataset import preprocess_dataset

def prep_dataset(dataset_name: str, sample: int = 101, seed: int = 42) -> DatasetDict | Dataset:
    """
    Prepares the dataset by loading the raw data and preprocessing it.
    Args:
        dataset_name (str): Name of the dataset to prepare.
        sample (int): Index of the sample to display for verification.
        seed (int): Seed used when raw data needs to be split locally.
    """
    #print(f"\n----------------CURRENT DATASET: {dataset_name}")

    # Load raw dataset
    raw = load_dataset_raw(dataset_name, seed=seed)

    # Display a sample from the raw dataset for verification
    #print("\nExample raw:\n", raw["train"][sample])

    # Preprocess the dataset
    return preprocess_dataset(name=dataset_name, sample=sample)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["ledgar", "unfair_tos"])
    args = parser.parse_args()

    prep_dataset(dataset_name=args.dataset)
