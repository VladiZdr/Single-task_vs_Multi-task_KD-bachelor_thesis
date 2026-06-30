import os
from datasets import DatasetDict, load_dataset


def load_dataset_raw(dataset, train_split=0.8, val_split=0.1, test_split=0.1, seed=42):
    raw_dir = os.path.join("datasets_store", f"{dataset}_raw")
    if os.path.isdir(raw_dir):
        return DatasetDict.load_from_disk(raw_dir)

    dataset_raw = load_dataset("coastalcph/lex_glue", dataset)

    # if splits are already present, return the dataset as is
    if isinstance(dataset_raw, DatasetDict) and {"train", "validation", "test"} <= set(dataset_raw):
        raw = dataset_raw

    else:
        assert abs(train_split + val_split + test_split - 1.0) < 1e-8
        data = dataset_raw["train"] if isinstance(dataset_raw, DatasetDict) else dataset_raw

        train_test = data.train_test_split(test_size=val_split + test_split, seed=seed)
        val_test = train_test["test"].train_test_split(
            test_size=test_split / (val_split + test_split),
            seed=seed,
        )
        raw = DatasetDict(
            {
                "train": train_test["train"],
                "validation": val_test["train"],
                "test": val_test["test"],
            }
        )

    raw.save_to_disk(raw_dir)
    return raw

def load_ledgar_raw(train_split=0.8, val_split=0.1, test_split=0.1, seed=42):
    return load_dataset_raw("ledgar", train_split, val_split, test_split, seed)

def load_unfair_tos_raw(train_split=0.8, val_split=0.1, test_split=0.1, seed=42):
    return load_dataset_raw("unfair_tos", train_split, val_split, test_split, seed)