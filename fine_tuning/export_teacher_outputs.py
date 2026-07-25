from collections import defaultdict
import logging
import os
import torch
from torch.utils.data import DataLoader
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from configs.model_config import ModelConfig
from datasets_manipulation.preprocess_dataset import (
    TOKENIZER_FIELD_NAMES,
    TOKENIZER_VIEWS,
    get_available_tokenizer_view_columns,
    get_tokenizer_column_name,
    get_tokenizer_view_for_model,
)

logger = logging.getLogger(__name__)

class SoftTargetExporter:
    # Class attributes listing tensor keys that every exported file contains.
    REQUIRED_COLUMNS = {"input_ids", "attention_mask", "logits", "probabilities", "labels"}
    OPTIONAL_COLUMNS = {"token_type_ids"}

    # Ensures exported outputs remain aligned index-for-index with the original dataset.
    @staticmethod
    def _as_in_order_loader(dataloader: DataLoader) -> DataLoader:
        return DataLoader(
            dataloader.dataset,  # type: ignore[arg-type]
            batch_size=dataloader.batch_size,
            shuffle=False,
            collate_fn=dataloader.collate_fn,
            num_workers=dataloader.num_workers,
            pin_memory=getattr(dataloader, "pin_memory", False),
            drop_last=dataloader.drop_last,
        )

    @staticmethod
    def export_all_splits(model: torch.nn.Module, dataloaders: dict, config: ModelConfig) -> None:
        for split_name, dataloader in dataloaders.items():
            SoftTargetExporter.export(model, dataloader, config, split_name)

    # Main method executing inference on a split and saving model predictions to disk.
    @staticmethod
    @torch.no_grad()
    def export(model: torch.nn.Module, dataloader: DataLoader, config: ModelConfig, split_name: str) -> None:
        dataloader = SoftTargetExporter._as_in_order_loader(dataloader)

        if len(dataloader) == 0:
            raise ValueError(f"Cannot export soft targets for empty split: {split_name}")
        
        model.eval() # turnoff dropout and other training-specific layers
        device = torch.device(config.device)
        model.to(device)

        tensor_buffers: dict[str, list[torch.Tensor]] = defaultdict(list)

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
            
            # Compute probabilities based on task 
            if config.problem_type == "multi_label":
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=-1)

            # Move data to CPU and append to lists
            tensor_buffers["input_ids"].append(input_ids.cpu())
            tensor_buffers["attention_mask"].append(attention_mask.cpu())
            if token_type_ids is not None:
                tensor_buffers["token_type_ids"].append(token_type_ids.cpu())
            tensor_buffers["logits"].append(logits.cpu())
            tensor_buffers["probabilities"].append(probs.cpu())
            tensor_buffers["labels"].append(labels.cpu())

            # Identifies any secondary dynamic tokenizer view columns inside the batch and retrieves the supplementary column data.
            for column_name in get_available_tokenizer_view_columns(list(batch.keys())):
                column_value = batch[column_name]
                if isinstance(column_value, torch.Tensor):
                    tensor_buffers[column_name].append(column_value.cpu())
            
        # Because GPUs have limited memory, we can't feed all our legal text into legal_bert at once. 
        # Instead, we break the data into small batches. 
        # Once the model has finished looping through all the batches, this  takes those fragmented pieces
        # and glues them back together into single, continuous matrices so we can calculate our final global metrics
        payload = {
            column_name: torch.cat(tensors, dim=0).contiguous()
            for column_name, tensors in tensor_buffers.items()
            if tensors
        }

        final_input_ids = payload["input_ids"]
        final_attention_masks = payload["attention_mask"]
        final_token_type_ids = payload.get("token_type_ids")
        final_logits = payload["logits"]
        final_probs = payload["probabilities"]
        final_labels = payload["labels"]
        
        # Structural Sanity Assertions
        assert final_logits.shape[0] == final_input_ids.shape[0], "Batch size mismatch between logits and input_ids"
        assert final_logits.shape[0] == len(dataloader.dataset), "Sample count mismatch in output"                      # type: ignore
        assert final_logits.shape[1] == config.num_labels, "Logit dimension mismatch with label count"

        # Determines the active tokenizer view name and extracts unique tokenizer view prefixes found in the payload (view_name__field_name).
        active_tokenizer_view = get_tokenizer_view_for_model(config.model_name_or_path)
        tokenizer_views = sorted({column_name.split("__", 1)[0] for column_name in payload if "__" in column_name})
        unknown_views = [view_name for view_name in tokenizer_views if view_name not in TOKENIZER_VIEWS]
        if unknown_views:
            raise AssertionError(f"Unexpected tokenizer view columns in export: {unknown_views}")

        metadata = {
            "task_name": config.task_name,
            "problem_type": config.problem_type,
            "split": split_name,
            "num_samples": str(final_logits.shape[0]),
            "num_classes": str(config.num_labels),
            "active_tokenizer_view": active_tokenizer_view,
            "tokenizer_views": ",".join(tokenizer_views),
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
            active_tokenizer_view = metadata.get("active_tokenizer_view")
            num_samples = int(metadata.get("num_samples", tensors["logits"].shape[0]))
            num_classes = int(metadata.get("num_classes", tensors["logits"].shape[1]))

            actual_columns = set(tensors.keys())
            missing_columns = SoftTargetExporter.REQUIRED_COLUMNS - actual_columns
            if missing_columns:
                raise AssertionError(
                    f"Split '{split_name}' in '{directory_path}' is missing required columns: {sorted(missing_columns)}"
                )

            # Extracts tokenizer views and validates that:
            # 1.The active tokenizer view in metadata is valid.
            # 2.No unknown views exist.         
            # 3.At least one tokenizer view column set is present.
            tokenizer_views = sorted({column_name.split("__", 1)[0] for column_name in actual_columns if "__" in column_name})
            expected_active_view = active_tokenizer_view if active_tokenizer_view in TOKENIZER_VIEWS else None
            if expected_active_view is None:
                raise AssertionError(
                    f"Split '{split_name}' metadata is missing a valid active_tokenizer_view: {active_tokenizer_view!r}"
                )
            unknown_views = [view_name for view_name in tokenizer_views if view_name not in TOKENIZER_VIEWS]
            if unknown_views:
                raise AssertionError(
                    f"Split '{split_name}' in '{directory_path}' contains unexpected tokenizer views: {unknown_views}"
                )
            if not tokenizer_views:
                raise AssertionError(
                    f"Split '{split_name}' in '{directory_path}' does not contain any prefixed tokenizer views."
                )

            # Validates that metadata view strings match tensor keys and checks that mandatory tokenizer field names exist for each expected view.
            metadata_views = [view_name for view_name in (metadata.get("tokenizer_views") or "").split(",") if view_name]
            if metadata_views and sorted(metadata_views) != tokenizer_views:
                raise AssertionError(
                    f"Split '{split_name}' metadata tokenizer views do not match the exported columns: "
                    f"{metadata_views} vs {tokenizer_views}"
                )
            for view_name in TOKENIZER_VIEWS:
                view_columns = [get_tokenizer_column_name(view_name, field_name) for field_name in TOKENIZER_FIELD_NAMES]
                missing_view_columns = [column_name for column_name in view_columns if column_name not in actual_columns]
                if missing_view_columns:
                    raise AssertionError(
                        f"Split '{split_name}' is missing tokenizer columns for view '{view_name}': "
                        f"{missing_view_columns}"
                    )

            # Prints diagnostic output showing which expected/optional columns are present.
            print("\n[1] Column Presence Verification:")
            for col in sorted(
                SoftTargetExporter.REQUIRED_COLUMNS
                | SoftTargetExporter.OPTIONAL_COLUMNS
                | {
                    get_tokenizer_column_name(view_name, field_name)
                    for view_name in TOKENIZER_VIEWS
                    for field_name in TOKENIZER_FIELD_NAMES
                }
            ):
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

            print(f"  Tokenizer views: {', '.join(tokenizer_views)}")

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
