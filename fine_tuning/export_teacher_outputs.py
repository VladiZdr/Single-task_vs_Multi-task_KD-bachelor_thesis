import torch
from torch.utils.data import DataLoader
from safetensors import safe_open
from safetensors.torch import save_file
import os
import logging
from tqdm import tqdm
from configs.model_config import ModelConfig

logger = logging.getLogger(__name__)

class SoftTargetExporter:
    REQUIRED_COLUMNS = {"input_ids", "attention_mask", "logits", "probabilities", "labels"}
    OPTIONAL_COLUMNS = {"token_type_ids"}

    @staticmethod
    def export_all_splits(model: torch.nn.Module, dataloaders: dict, config: ModelConfig) -> None:
        for split_name, dataloader in dataloaders.items():
            SoftTargetExporter.export(model, dataloader, config, split_name)
    
    @staticmethod
    @torch.no_grad() #no gradients are needed for inference
    def export(model: torch.nn.Module, dataloader: DataLoader, config: ModelConfig, split_name: str) -> None:
        if len(dataloader) == 0:
            raise ValueError(f"Cannot export soft targets for empty split: {split_name}")
        
        model.eval() # turnoff dropout and other training-specific layers
        device = torch.device(config.device)
        model.to(device)
        
        accumulated_input_ids = []
        accumulated_attention_masks = []
        accumulated_token_type_ids = []
        accumulated_logits = []
        accumulated_probs = []
        accumulated_labels = []

        logger.info(f"Extracting soft labels for task: {config.task_name}_{config.unique_id_for_dir}, split: {split_name}")
        
        for batch in tqdm(dataloader, desc=f"Exporting {split_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"]
            
            # Forward pass to get logits
            logits = model(input_ids, attention_mask, token_type_ids)
            
            # Compute probabilities based on task classification paradigm
            if config.problem_type == "multi_label":
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=-1)

            # Move data to CPU and append to lists
            accumulated_input_ids.append(input_ids.cpu())
            accumulated_attention_masks.append(attention_mask.cpu())
            if token_type_ids is not None:
                accumulated_token_type_ids.append(token_type_ids.cpu())
            accumulated_logits.append(logits.cpu())
            accumulated_probs.append(probs.cpu())
            accumulated_labels.append(labels.cpu())
            
        # Because GPUs have limited memory, we can't feed all our legal text into legal_bert at once. 
        # Instead, we break the data into small batches. 
        # Once the model has finished looping through all the batches, this code takes those fragmented pieces
        # and glues them back together into single, continuous matrices so we can calculate our final global metrics
        final_input_ids = torch.cat(accumulated_input_ids, dim=0)
        final_attention_masks = torch.cat(accumulated_attention_masks, dim=0)
        final_token_type_ids = torch.cat(accumulated_token_type_ids, dim=0) if accumulated_token_type_ids else None
        final_logits = torch.cat(accumulated_logits, dim=0)
        final_probs = torch.cat(accumulated_probs, dim=0)
        final_labels = torch.cat(accumulated_labels, dim=0)
        
        # Structural Sanity Assertions
        assert final_logits.shape[0] == final_input_ids.shape[0], "Batch size mismatch between logits and input_ids"
        assert final_logits.shape[0] == len(dataloader.dataset), "Sample count mismatch in output"                      # type: ignore
        assert final_logits.shape[1] == config.num_labels, "Logit dimension mismatch with label count"
        
        # Prepare SafeTensors state dictionary payload 
        # (tensor is considered contiguous when its dimensions match the actual physical layout of the memory cells)
        payload = {
            "input_ids": final_input_ids.contiguous(), # scan our tensor, allocate a brand-new, unbroken block of memory, and copy the data into it sequentially
            "attention_mask": final_attention_masks.contiguous(),
            "logits": final_logits.contiguous(),
            "probabilities": final_probs.contiguous(),
            "labels": final_labels.contiguous()
        }
        if final_token_type_ids is not None:
            payload["token_type_ids"] = final_token_type_ids.contiguous()
        
        metadata = {
            "task_name": config.task_name,
            "problem_type": config.problem_type,
            "split": split_name,
            "num_samples": str(final_logits.shape[0]),
            "num_classes": str(config.num_labels)
        }
        
        export_path = os.path.join(config.output_dir, f"teacher_{split_name}_outputs.safetensors")
        save_file(payload, export_path, metadata=metadata)
        logger.info(f"Successfully serialized soft targets to {export_path}")

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

        for split_name, file_path in split_files.items():
            print("=" * 60)
            print(f"Found Exported File: {os.path.basename(file_path)}")
            print(f"Full Path:           {file_path}")
            print("=" * 60)

            with safe_open(file_path, framework="pt", device="cpu") as exported:
                tensors = {key: exported.get_tensor(key) for key in exported.keys()}
                metadata = exported.metadata() or {}

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

            print("\n[1] Column Presence Verification:")
            for col in sorted(SoftTargetExporter.REQUIRED_COLUMNS | SoftTargetExporter.OPTIONAL_COLUMNS):
                present = col in actual_columns
                status = "PRESENT" if present else "MISSING"
                print(f"  - {col:<16}: {status}")

            print("\n[2] Tensor Metadata & Specifications:")
            for key, tensor in tensors.items():
                shape_str = str(list(tensor.shape))
                print(
                    f"  - {key:<16}: shape={shape_str:<18} | dtype={str(tensor.dtype):<13} | device={tensor.device}"
                )

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

            assert torch.allclose(
                tensors["probabilities"],
                expected_probs,
                atol=1e-6,
                rtol=1e-5,
            ), f"Probability tensor in '{split_name}' does not match the logits-derived expectation"

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

            verification_summary[split_name] = {
                "path": file_path,
                "task_name": expected_task,
                "problem_type": problem_type,
                "num_samples": num_samples,
                "num_classes": num_classes,
            }

        return verification_summary
