import os
import sys
import torch
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datasets_manipulation.export_teacher_outputs import SoftTargetExporter
from safetensors import safe_open
from fine_tuning.train_legal_model import models_to_run

@staticmethod
def verify_exports(directory_path: str) -> dict[str, dict[str, object]]:
    """
    Scans the outputs folder, loads each generated SafeTensors split,
    checks that all mandatory columns are present, and validates that the
    stored tensors are consistent with the task metadata.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(
            f"Target directory '{directory_path}' does not exist. Make sure the export step has run first."
        )

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

    verification_summary: dict[str, dict[str, object]] = {}

    # Initializes summary dictionary and prints formatting headers per split file.
    for split_name, file_path in split_files.items():
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
        missing_columns = SoftTargetExporter.REQUIRED_COLUMNS - actual_columns
        if missing_columns:
            raise AssertionError(
                f"Split '{split_name}' in '{directory_path}' is missing required columns: {sorted(missing_columns)}"
            )

        # Prints diagnostic output showing which expected/optional columns are present.
        print("\n[1] Column Presence Verification:")
        for col in sorted(SoftTargetExporter.REQUIRED_COLUMNS | SoftTargetExporter.OPTIONAL_COLUMNS):
            present = col in actual_columns
            status = "PRESENT" if present else "MISSING"
            print(f"  - {col:<16}: {status}")

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

        for key in SoftTargetExporter.REQUIRED_COLUMNS:
            assert tensors[key].shape[0] == num_samples, f"Batch size mismatch on column: {key}!"

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

        # Records metadata summary for the split and returns verification_summary containing verified metrics across all splits.
        verification_summary[split_name] = {
            "path": file_path,
            "task_name": expected_task,
            "problem_type": problem_type,
            "num_samples": num_samples,
            "num_classes": num_classes,
        }

    return verification_summary

def check_all_exports() -> None:
    for model in models_to_run:
        print(f"\n[{model.task_name}]")
        summary = verify_exports(directory_path=model.output_dir)
        assert set(summary) == {"train", "validation", "test"}, f"{model.task_name} did not contain all three splits"
        for split_name, split_summary in summary.items():
            assert split_summary["num_samples"] > 0, f"{model.task_name} split '{split_name}' is empty"                           #type: ignore
            assert split_summary["num_classes"] in {8, 100}, f"{model.task_name} split '{split_name}' has unexpected class count"

    print("\nAll teacher-output exports passed verification.")

if __name__ == "__main__":
    check_all_exports()
