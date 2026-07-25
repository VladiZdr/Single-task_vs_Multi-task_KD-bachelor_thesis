from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datasets import Dataset, DatasetDict

from datasets_manipulation.prepare_datasets import (
    align_dataset_tokenization,
    get_torch_columns_for_split,
)
from datasets_manipulation.preprocess_dataset import get_tokenizer_view_for_model

# Defines a test verifying model name to view mapping logic. 
# Validates that the Legal/Google BERT repository string correctly resolves to the internal view identifier.
def test_tokenizer_view_mapping_matches_supported_models() -> None:
    assert get_tokenizer_view_for_model("nlpaueb/legal-bert-base-uncased") == "legal_bert"
    assert get_tokenizer_view_for_model("google/bert_uncased_L-4_H-256_A-4") == "google_bert"

# Test ensuring that selecting a active model tokenizer view updates primary model input columns 
# (input_ids, attention_mask) without deleting other tokenizer view data stored under prefixed column names.
def test_align_dataset_tokenization_overwrites_standard_columns_without_dropping_views() -> None:
    # Builds a mock Hugging Face DatasetDict containing dummy train data with standard columns 
    # (input_ids) as well as prefixed tokenizer views (legal_bert__ and google_bert__).
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "input_ids": [[1, 1], [1, 1]],
                    "attention_mask": [[1, 1], [1, 1]],
                    "labels": [0, 1],
                    "legal_bert__input_ids": [[2, 2], [2, 2]],
                    "legal_bert__attention_mask": [[0, 0], [0, 0]],
                    "google_bert__input_ids": [[3, 3], [3, 3]],
                    "google_bert__attention_mask": [[4, 4], [4, 4]],
                }
            )
        }
    )

    # Aligns the dataset using the model keys.
    legal_aligned = align_dataset_tokenization(dataset, "nlpaueb/legal-bert-base-uncased")
    google_aligned = align_dataset_tokenization(dataset, "google/bert_uncased_L-4_H-256_A-4")

    # Verifies Legal BERT alignment:
    # 1.input_ids replaced with values from legal_bert__input_ids ([2, 2]).
    # 2.attention_mask replaced with values from legal_bert__attention_mask ([0, 0]).
    # 3.The non-active view column google_bert__input_ids ([3, 3]) remains intact.
    assert legal_aligned["train"][0]["input_ids"] == [2, 2]
    assert legal_aligned["train"][0]["attention_mask"] == [0, 0]
    assert legal_aligned["train"][0]["google_bert__input_ids"] == [3, 3]

    assert google_aligned["train"][0]["input_ids"] == [3, 3]
    assert google_aligned["train"][0]["attention_mask"] == [4, 4]
    assert google_aligned["train"][0]["legal_bert__input_ids"] == [2, 2]

# Defines a test to ensure get_torch_columns_for_split detects standard model inputs 
# as well as all multi-view tokenizer features needed for PyTorch dataset formatting.
def test_get_torch_columns_for_split_includes_extra_tokenizer_views() -> None:
    # Mock single dataset split
    split = Dataset.from_dict(
        {
            "input_ids": [[1, 2]],
            "attention_mask": [[1, 1]],
            "token_type_ids": [[0, 0]],
            "labels": [1],
            "task": ["ledgar"],
            "logits": [[0.1, 0.9]],
            "legal_bert__input_ids": [[4, 5]],
            "legal_bert__attention_mask": [[1, 1]],
            "legal_bert__token_type_ids": [[0, 0]],
            "google_bert__input_ids": [[6, 7]],
            "google_bert__attention_mask": [[1, 1]],
            "google_bert__token_type_ids": [[0, 0]],
        }
    )

    # Calls get_torch_columns_for_split, setting flags to explicitly include task metadata and teacher model logits in the list of PyTorch-convertible columns.
    columns = get_torch_columns_for_split(split, include_task=True, include_logits=True)

    # Asserts that all prefixed tokenizer view columns are preserved and included in the output column list so they can be cast into PyTorch tensors.
    assert columns[:6] == ["input_ids", "attention_mask", "token_type_ids", "labels", "logits", "task"]
    assert "legal_bert__input_ids" in columns
    assert "legal_bert__attention_mask" in columns
    assert "legal_bert__token_type_ids" in columns
    assert "google_bert__input_ids" in columns
    assert "google_bert__attention_mask" in columns
    assert "google_bert__token_type_ids" in columns

def run_all_tokenizer_tests():
    test_tokenizer_view_mapping_matches_supported_models()
    test_align_dataset_tokenization_overwrites_standard_columns_without_dropping_views()
    test_get_torch_columns_for_split_includes_extra_tokenizer_views()
    print("Finished tokenizer tests")
