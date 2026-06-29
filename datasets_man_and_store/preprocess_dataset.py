import gc
import re
import shutil
import os
from pathlib import Path
from shutil import copytree, rmtree
from typing import cast
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer

def preprocess_dataset(raw_dataset_dir, sample) -> DatasetDict | Dataset:
    dataset_dir = create_preprocessed_dataset_dir(raw_dataset_dir=raw_dataset_dir)

    raw_dataset_dir = Path(raw_dataset_dir)
    dataset_name = raw_dataset_dir.name.replace("_raw", "")

    add_task_marker(dataset_dir = dataset_dir, task_marker = dataset_name)
    clean_text(dataset_dir = dataset_dir)
    rename_label_to_labels(dataset_dir = dataset_dir)
    to_multi_hot(dataset_dir = dataset_dir)
    tokenize_text(dataset_dir = dataset_dir)

    tokenized_ds = load_from_disk(str(dataset_dir))

    #print("\nExample after preprocessing:\n", tokenized_ds["train"][sample])
    return tokenized_ds

def _load_valid_dataset_dict(dataset_dir):
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    # Checks whether the path points to a directory (folder)
    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset path is not a directory: {dataset_dir}")

    try:
        dataset = load_from_disk(str(dataset_dir))
    except Exception as error:
        raise ValueError(f"Not a valid Hugging Face dataset directory: {dataset_dir}") from error

    if not isinstance(dataset, DatasetDict):
        raise ValueError(f"Expected a DatasetDict with splits, got: {type(dataset).__name__}")

    if "train" not in dataset:
        raise ValueError(f"Expected dataset to contain a train split: {dataset_dir}")

    return dataset


def _update_dataset_dir(dataset_dir, process_dataset):
    dataset_dir = Path(dataset_dir)
    dataset = _load_valid_dataset_dict(dataset_dir)
    processed_dataset = process_dataset(dataset)

    if processed_dataset is None:
        return dataset

    # Datasets cannot save over the same directory they were loaded from,
    # because the dataset is currently being read from that directory.
    temp_dir = dataset_dir.with_name(dataset_dir.name + "_tmp")

    # Remove the entire directory tree of the temporary directory if it already exists, to avoid conflicts.
    if temp_dir.exists():
        rmtree(temp_dir)

    processed_dataset.save_to_disk(str(temp_dir))

    # Drop references before removing or renaming dataset directories on disk.
    del dataset
    del processed_dataset
    # The loaded dataset may still keep file handles open
    gc.collect()

    rmtree(dataset_dir)
    temp_dir.rename(dataset_dir)

    return None


def create_preprocessed_dataset_dir(raw_dataset_dir, output_dir=None):
    raw_dataset_dir = Path(raw_dataset_dir)
    _load_valid_dataset_dict(raw_dataset_dir)

    if output_dir is None:
        output_name = raw_dataset_dir.name.replace("_raw", "_preprocessed")
        output_dir = raw_dataset_dir.parent / output_name
    else:
        output_dir = Path(output_dir)

    # Create the preprocessed working copy only once.
    if not output_dir.exists():
        copytree(raw_dataset_dir, output_dir)

    return output_dir


def rename_label_to_labels(dataset_dir):
    dataset_dir = Path(dataset_dir)

    def process(dataset):
        # dataset["train"] is inferred by some type checkers as a generic object,
        # so they may complain that it has no column_names attribute. The cast removes that warning.
        if "label" not in cast(Dataset, dataset["train"]).column_names:
            return None

        return dataset.rename_column("label", "labels")

    return _update_dataset_dir(dataset_dir, process)


def add_task_marker(dataset_dir, task_marker):
    dataset_dir = Path(dataset_dir)

    def process(dataset):
        train_dataset = cast(Dataset, dataset["train"])

        # If the requested marker is already present, nothing needs to be changed.
        if "task" in train_dataset.column_names and train_dataset[0]["task"] == task_marker:
            return None

        # The map() method applies a function to every example in the dataset.
        return dataset.map(
            # The lambda (unnamed) function here adds a new column "task" with the specified task_marker value for every example.
            lambda example: {"task": task_marker},
            # does not modify dataset in place. Instead, it returns a new dataset.
            keep_in_memory=True,
        )

    return _update_dataset_dir(dataset_dir, process)


def normalize_text(text):
    #which characters to replace with space, which to remove, and which to normalize
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(dataset_dir):
    dataset_dir = Path(dataset_dir)

    def process(dataset):
        if "text" not in cast(Dataset, dataset["train"]).column_names:
            return None

        return dataset.map(
            lambda example: {"text": normalize_text(example["text"])},
            keep_in_memory=True,
        )

    return _update_dataset_dir(dataset_dir, process)


def to_multi_hot(dataset_dir):
    dataset_dir = Path(dataset_dir)
    num_labels = 8

    def process(dataset):
        # LEDGAR-style datasets have one integer label per example, so there is
        # nothing to multi-hot encode unless a "labels" column exists.
        if "labels" not in dataset["train"].column_names:
            return None

        first_label = dataset["train"][0]["labels"]

        # If the first label is not a list, this is single-label classification
        # and the integer label ID can be kept as-is.
        if not isinstance(first_label, list):
            return None

        # If labels are already vectors such as [0, 1, 0, ...], avoid encoding twice.
        if len(first_label) == num_labels and all(label in (0, 1) for label in first_label):
            return None

        label_ids = set()
        for split in dataset.values():
            for labels in split["labels"]:
                label_ids.update(labels)

        def encode(example):
            # Convert label ID lists such as [1, 4] into multi-hot vectors such as
            # [0, 1, 0, 0, 1, 0, 0, 0] for multi-label classification.
            multi_hot = [0] * num_labels
            for label in example["labels"]:
                multi_hot[label] = 1
            return {"labels": multi_hot}

        return dataset.map(encode, keep_in_memory=True)

    return _update_dataset_dir(dataset_dir, process)


def tokenize_text(dataset_dir, model_name="nlpaueb/legal-bert-base-uncased", max_length=256):
    dataset_dir = Path(dataset_dir)

    def process(dataset):
        train_dataset = cast(Dataset, dataset["train"])

        # Skip tokenization when rerunning preprocessing.
        if {"input_ids", "attention_mask", "labels"} <= set(train_dataset.column_names):
            return None

        if "text" not in train_dataset.column_names:
            return None

        # Load the tokenizer that matches the Legal-BERT model used for the experiments.
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Tokenize one example's text into input_ids and attention_mask.
        def tokenize(example):
            return tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )

        # Remove raw text and other non-label columns after tokenization.
        columns_to_remove = [
            column for column in train_dataset.column_names if column != "labels" and column != "task"
        ]

        # Apply tokenization to every split and keep only the generated model inputs plus labels.
        return dataset.map(
            tokenize,
            remove_columns=columns_to_remove,
            keep_in_memory=True,
        )

    return _update_dataset_dir(dataset_dir, process)
