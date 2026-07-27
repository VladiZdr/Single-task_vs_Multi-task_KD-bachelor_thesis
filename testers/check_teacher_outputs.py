import os
import sys
import torch
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from safetensors import safe_open
from fine_tuning.train_legal_model import models_to_run
from configs.model_config import ModelConfig

# Class attributes listing tensor keys that every exported file contains.
REQUIRED_COLUMNS = {"input_ids", "attention_mask", "logits", "probabilities", "labels", "task", "sample_index"}
OPTIONAL_COLUMNS = {"token_type_ids"}

def check_missing_splits_and_get_train_val_test_splits(directory_path):
    required_splits = ("train", "validation", "test")
    split_files = {
            split: os.path.join(directory_path, f"teacher_{split}_outputs.safetensors")
            for split in required_splits
    }
    
    missing_splits = [split for split, file_path in split_files.items() if not os.path.exists(file_path)]
    if missing_splits:
            raise FileNotFoundError(
                f"Missing exported split files in '{directory_path}': {', '.join(missing_splits)}"
            )
    return split_files

def check_missing_columns(actual_columns, split_name, directory_path):
    missing_columns = REQUIRED_COLUMNS - actual_columns
    if missing_columns:
        raise AssertionError(
            f"Split '{split_name}' in '{directory_path}' is missing required columns: {sorted(missing_columns)}"
        )
    # Prints diagnostic output showing which expected/optional columns are present.
    print("\n[1] Column Presence Verification:")
    for col in sorted(REQUIRED_COLUMNS | OPTIONAL_COLUMNS):
        present = col in actual_columns
        status = "PRESENT" if present else "MISSING"
        print(f"  - {col:<16}: {status}")

def print_info_about_safetensors(tensors, expected_task, num_classes, num_samples):
    # Prints shapes, data types, and device information for all stored tensors.
    print("\n[2] Tensor Metadata & Specifications:")
    for key, tensor in tensors.items():
        shape_str = str(list(tensor.shape))
        print(
            f"  - {key:<16}: shape={shape_str:<18} | dtype={str(tensor.dtype):<13} | device={tensor.device}"
        )
    
    #Verifies sample count dimensions match metadata and across all required tensors.
    if expected_task is not None:
        if expected_task == "unfair_tos":
            assert num_classes == 8, f"Expected 8 label dimensions for UNFAIR-ToS, found {num_classes}."
        elif expected_task == "ledgar":
            assert num_classes == 100, f"Expected 100 label dimensions for LEDGAR, found {num_classes}."
    
    assert tensors["logits"].shape[0] == num_samples, "Metadata sample count does not match logits"
    assert tensors["logits"].shape[1] == num_classes, "Metadata class count does not match logits"
    
    for key in REQUIRED_COLUMNS:
        assert tensors[key].shape[0] == num_samples, f"Batch size mismatch on column: {key}!"

def verify_index_seq(tensors,num_samples, split_name):
    # Verify index sequence: sample_index must be strictly 0, 1, 2, ..., N-1
    
    sample_indices = tensors["sample_index"].to(torch.int64)
    expected_indices = torch.arange(num_samples, dtype=torch.int64)
            
    assert torch.equal(sample_indices, expected_indices), (
        f"ORDER FAILURE: Split '{split_name}' has been shuffled or exported out of order! "
        f"Expected sample_index 0..{num_samples-1}, but found non-sequential indices."
    )
    print(f"  ✓ Row order verified: 'sample_index' is strictly sequential (0 to {num_samples - 1}).")

def verify_probability_consistency(problem_type, tensors, num_samples, split_name):
#  Probability consistency
        if problem_type == "multi_label":
            expected_probs = torch.sigmoid(tensors["logits"])
            assert tensors["labels"].shape == tensors["logits"].shape, "Multi-label exports must store dense label vectors"
        elif problem_type == "single_label":
            expected_probs = torch.softmax(tensors["logits"], dim=-1)
            assert tensors["labels"].shape[0] == num_samples, "Single-label exports must store one label per sample"
        else:
            raise AssertionError(f"Unknown problem type in metadata: {problem_type}")

        # Asserts stored probabilities match recomputed probabilities within floating-point tolerance
        assert torch.allclose(
            tensors["probabilities"],
            expected_probs,
            atol=1e-6,
            rtol=1e-5,
        ), f"Probability tensor in '{split_name}' does not match the logits-derived expectation"

def diagnostic_inspection_output_for_first_sample_of_train(split_name, num_samples, num_classes, expected_task, problem_type, tensors):
    # Prints diagnostic inspection output for first sample to allow manual inspection of input IDs, attention mask, ground truth, logits, and probabilities.
    print("\n[3] Pipeline Integrity Checks:")
    print(f"  ✓ Split '{split_name}' contains {num_samples} records and {num_classes} classes.")
    print(f"  ✓ Metadata task: {expected_task}, problem type: {problem_type}")
    
    idx = 0
    raw_tokens = tensors["input_ids"][idx].tolist()
    clean_tokens = [tok for tok in raw_tokens if tok != 0]
    
    print("\n[4] Sample 0 Inspection:")
    print(f"  Input IDs (trimmed length={len(clean_tokens)}): {clean_tokens}")
    print(f"  Attention mask head: {tensors['attention_mask'][idx][:30].tolist()}")
    print(f"  Labels head: {tensors['labels'][idx].tolist()}")
    print(f"  Logits head: {[round(x, 4) for x in tensors['logits'][idx].tolist()]}")
    print(f"  Probabilities head: {[round(x, 4) for x in tensors['probabilities'][idx].tolist()]}")


@staticmethod
def verify_exports(directory_path: str, model:ModelConfig) -> dict[str, dict[str, object]]:
    """
    Scans the outputs folder, loads each generated SafeTensors split,
    checks that all mandatory columns are present, and validates that the
    stored tensors are consistent with the task metadata.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(
            f"Target directory '{directory_path}' does not exist. Make sure the export step has run first."
        )

    split_files = check_missing_splits_and_get_train_val_test_splits(directory_path)

    verification_summary: dict[str, dict[str, object]] = {}

    # Initializes summary dictionary and prints formatting headers per split file.
    for split_name, file_path in split_files.items():
        if split_name != "train":
            continue

        print("=" * 60)
        print(f"Found Exported File: {os.path.basename(file_path)}")
        print(f"Full Path:           {file_path}")
        print("=" * 60)

        with safe_open(file_path, framework="pt", device="cpu") as exported:
            tensors = {key: exported.get_tensor(key) for key in exported.keys()}
            metadata = exported.metadata() or {}

        # Parses metadata values 
        expected_task = metadata.get("task_name")
        problem_type = metadata.get("problem_type")
        num_samples = int(metadata.get("num_samples", tensors["logits"].shape[0]))
        num_classes = int(metadata.get("num_classes", tensors["logits"].shape[1]))

        actual_columns = set(tensors.keys())
        check_missing_columns(actual_columns, split_name, directory_path)

        print_info_about_safetensors(tensors, expected_task, num_classes, num_samples)

        # Not applied on low_ressource models because the sample idxes are not sequential due to stratified sampling
        if model.low_resource_percent == 100:
            verify_index_seq(tensors,num_samples, split_name)

        verify_probability_consistency(problem_type, tensors, num_samples, split_name)

        
        diagnostic_inspection_output_for_first_sample_of_train(split_files["train"], num_samples, num_classes, expected_task, problem_type, tensors)

        # Records metadata summary for the split and returns verification_summary containing verified metrics across all splits.
        verification_summary[split_name] = {
            "path": file_path,
            "task_name": expected_task,
            "problem_type": problem_type,
            "num_samples": num_samples,
            "num_classes": num_classes,
        }

    return verification_summary

#TODO (could be implemented for each batch)
def check_correct_transition_between_tokenizers(config: ModelConfig, dataloaders_inference: dict, dataloaders_export: dict):
    """
    if config.model_name_or_path = google/bert_uncased_L-4_H-256_A-4 -> tokenized_ds_training = tokenized_ds_export -> do nothing
    else
        1. untokenize dataloaders_inference
        2. tokenize dataloaders_inference with google/bert_uncased_L-4_H-256_A-4
        3. compare (2) with dataloaders_export
    """
    return

def check_all_exports() -> None:
    for model in models_to_run:
        print(f"\n[{model.task_name}]")
        summary = verify_exports(directory_path=model.output_dir, model = model)
        for split_name, split_summary in summary.items():
            assert split_summary["num_samples"] > 0, f"{model.task_name} split '{split_name}' is empty"                           #type: ignore
            assert split_summary["num_classes"] in {8, 100}, f"{model.task_name} split '{split_name}' has unexpected class count"

    print("\nAll teacher-output exports passed verification.")

if __name__ == "__main__":
    check_all_exports()
