from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from datasets import Dataset, DatasetDict
    import datasets_manipulation.prepare_datasets as prepare_datasets
    from datasets_manipulation.prepare_datasets import _multi_hot_labels, sample_low_resource_dataset
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local runtime
    pytest.skip(f"Skipping low-resource sampling tests because a dependency is missing: {exc}", allow_module_level=True)


def _make_split(*, prefix: str, labels: list[Any], label_key: str) -> Dataset:
    size = len(labels)
    return Dataset.from_dict(
        {
            "text": [f"{prefix} {index}" for index in range(size)],
            label_key: labels,
            "row_id": list(range(size)),
        }
    )


def _label_prevalence(split: Dataset, label_column: str) -> np.ndarray:
    matrix = _multi_hot_labels(split[label_column])
    if len(matrix) == 0:
        return np.zeros(0, dtype=np.float64)
    return matrix.mean(axis=0)


@pytest.fixture
def synthetic_ledgar_dataset() -> DatasetDict:
    train_labels = [label for label in range(10) for _ in range(100)]
    validation_labels = [index % 10 for index in range(100)]
    test_labels = [(index * 3) % 10 for index in range(100)]

    return DatasetDict(
        {
            "train": _make_split(prefix="LEDGAR train", labels=train_labels, label_key="label"),
            "validation": _make_split(prefix="LEDGAR validation", labels=validation_labels, label_key="label"),
            "test": _make_split(prefix="LEDGAR test", labels=test_labels, label_key="label"),
        }
    )


@pytest.fixture
def synthetic_unfair_tos_dataset() -> DatasetDict:
    train_patterns = [
        [0],
        [0, 1],
        [1],
        [2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7],
    ]
    validation_patterns = [
        [1],
        [1, 2],
        [2],
        [3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 0],
        [0],
    ]
    test_patterns = [
        [2],
        [2, 4],
        [4],
        [5],
        [5, 7],
        [7, 1],
        [1, 3],
        [3, 6],
        [6, 0],
        [0],
    ]

    train_labels = [pattern for pattern in train_patterns for _ in range(100)]
    validation_labels = [pattern for pattern in validation_patterns for _ in range(10)]
    test_labels = [pattern for pattern in test_patterns for _ in range(10)]

    return DatasetDict(
        {
            "train": _make_split(prefix="UNFAIR train", labels=train_labels, label_key="labels"),
            "validation": _make_split(prefix="UNFAIR validation", labels=validation_labels, label_key="labels"),
            "test": _make_split(prefix="UNFAIR test", labels=test_labels, label_key="labels"),
        }
    )


def test_val_and_test_remain_unchanged(synthetic_ledgar_dataset: DatasetDict) -> None:
    original = synthetic_ledgar_dataset
    sampled = sample_low_resource_dataset(original, "ledgar", 10, seed=42)

    assert len(sampled["train"]) == 100
    assert len(sampled["validation"]) == len(original["validation"])
    assert len(sampled["test"]) == len(original["test"])
    assert sampled["validation"][:] == original["validation"][:]
    assert sampled["test"][:] == original["test"][:]


def test_sampling_is_not_first_n_percent(synthetic_ledgar_dataset: DatasetDict) -> None:
    original = synthetic_ledgar_dataset
    sampled = sample_low_resource_dataset(original, "ledgar", 10, seed=42)

    assert sampled["train"]["text"] != original["train"]["text"][:100]
    assert sampled["train"]["row_id"] != list(range(100))


@pytest.mark.parametrize(
    ("percent", "expected_per_class"),
    [
        (1, 1),
        (10, 10),
        (25, 25),
        (50, 50),
    ],
)
def test_ledgar_stratification_preserves_label_distribution(
    synthetic_ledgar_dataset: DatasetDict,
    percent: int,
    expected_per_class: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = synthetic_ledgar_dataset
    expected_size = max(1, int(len(original["train"]) * (percent / 100)))
    captured_kwargs: dict[str, Any] = {}

    real_train_test_split = prepare_datasets.train_test_split

    def spy_train_test_split(*args: Any, **kwargs: Any):
        captured_kwargs.clear()
        captured_kwargs.update(kwargs)
        return real_train_test_split(*args, **kwargs)

    monkeypatch.setattr(prepare_datasets, "train_test_split", spy_train_test_split)

    sampled = sample_low_resource_dataset(original, "ledgar", percent, seed=42)

    assert len(sampled["train"]) == expected_size
    assert sampled["train"]["row_id"] != list(range(expected_size))

    expected_counts = {label: expected_per_class for label in range(10)}
    assert Counter(sampled["train"]["label"]) == expected_counts
    np.testing.assert_array_equal(captured_kwargs["stratify"], np.asarray(original["train"]["label"]))


@pytest.mark.parametrize("percent", [10, 25, 50])
def test_unfair_tos_multilabel_stratification(
    synthetic_unfair_tos_dataset: DatasetDict,
    percent: int,
) -> None:
    pytest.importorskip("iterstrat")

    original = synthetic_unfair_tos_dataset
    expected_size = max(1, int(len(original["train"]) * (percent / 100)))
    sampled = sample_low_resource_dataset(original, "unfair_tos", percent, seed=42)

    assert abs(len(sampled["train"]) - expected_size) <= 2
    assert sampled["train"]["row_id"] != list(range(expected_size))

    original_prevalence = _label_prevalence(original["train"], "labels")
    sampled_prevalence = _label_prevalence(sampled["train"], "labels")
    sequential_prevalence = _label_prevalence(original["train"].select(range(expected_size)), "labels")

    assert np.all(sampled_prevalence > 0)
    assert np.count_nonzero(sampled_prevalence) == 8

    sampled_mae = np.abs(sampled_prevalence - original_prevalence).mean()
    sequential_mae = np.abs(sequential_prevalence - original_prevalence).mean()
    assert sampled_mae < sequential_mae * 0.25


def test_percentage_100_returns_original_dataset_object(synthetic_ledgar_dataset: DatasetDict) -> None:
    original = synthetic_ledgar_dataset
    sampled = sample_low_resource_dataset(original, "ledgar", 100, seed=123)

    assert sampled is original
    assert sampled["train"][:] == original["train"][:]
    assert sampled["validation"][:] == original["validation"][:]
    assert sampled["test"][:] == original["test"][:]


@pytest.mark.parametrize("percent", [0, 5, 15, 30, 99, 105, -10])
def test_invalid_percentages_raise_value_error(
    synthetic_ledgar_dataset: DatasetDict,
    percent: int,
) -> None:
    with pytest.raises(ValueError):
        sample_low_resource_dataset(synthetic_ledgar_dataset, "ledgar", percent, seed=42)


def test_missing_train_split_raises_value_error() -> None:
    dataset = DatasetDict(
        {
            "validation": _make_split(prefix="validation", labels=[0, 1, 2], label_key="label"),
            "test": _make_split(prefix="test", labels=[0, 1, 2], label_key="label"),
        }
    )

    with pytest.raises(ValueError, match="train split"):
        sample_low_resource_dataset(dataset, "ledgar", 10, seed=42)


def test_unsupported_dataset_name_raises_value_error(synthetic_ledgar_dataset: DatasetDict) -> None:
    with pytest.raises(ValueError, match="Unsupported low-resource sampling dataset"):
        sample_low_resource_dataset(synthetic_ledgar_dataset, "invalid_dataset", 10, seed=42)


def test_seed_reproducibility(synthetic_ledgar_dataset: DatasetDict) -> None:
    original = synthetic_ledgar_dataset

    same_seed_first = sample_low_resource_dataset(original, "ledgar", 25, seed=42)
    same_seed_second = sample_low_resource_dataset(original, "ledgar", 25, seed=42)
    different_seed = sample_low_resource_dataset(original, "ledgar", 25, seed=123)

    assert same_seed_first["train"]["row_id"] == same_seed_second["train"]["row_id"]
    assert same_seed_first["train"]["text"] == same_seed_second["train"]["text"]
    assert same_seed_first["validation"][:] == same_seed_second["validation"][:]
    assert same_seed_first["test"][:] == same_seed_second["test"][:]

    assert same_seed_first["train"]["row_id"] != different_seed["train"]["row_id"]
    assert same_seed_first["train"]["text"] != different_seed["train"]["text"]


def test_multi_hot_labels_conversion() -> None:
    sparse_labels = [[0, 3], [], [1, 7]]
    expected_sparse = np.array(
        [
            [1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.int64,
    )

    existing_multi_hot = [
        [1, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 0],
    ]
    expected_existing = np.asarray(existing_multi_hot, dtype=np.int64)

    np.testing.assert_array_equal(_multi_hot_labels([]), np.zeros((0, 8), dtype=np.int64))
    np.testing.assert_array_equal(_multi_hot_labels(sparse_labels), expected_sparse)
    np.testing.assert_array_equal(_multi_hot_labels(existing_multi_hot), expected_existing)
