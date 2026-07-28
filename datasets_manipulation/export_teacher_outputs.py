from collections import defaultdict
import logging
import os
import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file
from tqdm import tqdm
from configs.model_configs import ModelConfig
from functools import lru_cache
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

@lru_cache(maxsize=4)
def _get_tokenizer(model_name_or_path: str):
    """Cached tokenizer fetcher to prevent loading tokenizers on every batch step."""
    return AutoTokenizer.from_pretrained(model_name_or_path)

def _verify_tokenization_match(
    texts: list[str],
    expected_input_ids: torch.Tensor,
    tokenizer,
    model_name: str,
    split_name: str,
    source_name: str,
    target_name: str,
) -> None:
    """Re-tokenizes a list of text strings and verifies that the output matches expected input_ids."""
    max_len = expected_input_ids.shape[1]
    retokenized_ids = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )["input_ids"].to(expected_input_ids.device)

    if not torch.equal(retokenized_ids, expected_input_ids):
        raise ValueError(
            f"Tokenization mismatch on split '{split_name}'! "
            f"Tokenizing {source_name} with '{model_name}' "
            f"does not match {target_name}."
        )

class SoftTargetExporter:
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
    def _encode_string_list(str_list: list[str], max_len: int = 32) -> torch.Tensor:
        """Encodes a list of string task names into a fixed-length uint8 PyTorch tensor for SafeTensors compatibility."""
        encoded_batch = []
        for s in str_list:
            b = str(s).encode("utf-8")[:max_len]
            padded = list(b) + [0] * (max_len - len(b))
            encoded_batch.append(padded)
        return torch.tensor(encoded_batch, dtype=torch.uint8)

    @staticmethod
    def check_correct_dataloaders_length(dataloader_inference: DataLoader, dataloader_export: DataLoader, split_name):
        if len(dataloader_inference) == 0 or len(dataloader_export) == 0:
            raise ValueError(f"Cannot export soft targets for empty split: {split_name}")
        
        if len(dataloader_inference.dataset) != len(dataloader_export.dataset):  # type: ignore
            raise ValueError(
                f"Dataset length mismatch for split '{split_name}': "
                f"inference dataset has {len(dataloader_inference.dataset)} samples, "  # type: ignore
                f"export dataset has {len(dataloader_export.dataset)} samples."  # type: ignore
            )

    @staticmethod
    def check_correct_transition_between_tokenizers(config: ModelConfig, batch_inference, batch_export, sample_idx_inf, sample_idx_exp, split_name):
        # 1. Strict Sample Alignment Check (Primary Guarantee)
        if not torch.equal(sample_idx_inf, sample_idx_exp):
            raise ValueError(
                f"Sample index mismatch on split '{split_name}'! "
                f"Inference indices: {sample_idx_inf.tolist()} vs Export indices: {sample_idx_exp.tolist()}"
            )

        # 2. Batch Size & Structural Consistency Check
        if batch_inference["input_ids"].shape[0] != batch_export["input_ids"].shape[0]:
            raise ValueError(
                f"Batch size mismatch on split '{split_name}'! "
                f"Inference batch: {batch_inference['input_ids'].shape[0]}, Export batch: {batch_export['input_ids'].shape[0]}"
            )

        export_model_name = "google/bert_uncased_L-4_H-256_A-4"

        # 3. Direct ID Check when tokenizers are identical is sufficient
        if config.model_name_or_path == export_model_name:
            if not torch.equal(batch_inference["input_ids"], batch_export["input_ids"]):
                raise ValueError(
                    f"Identical tokenizer configured ('{export_model_name}'), "
                    f"but input_ids differ on split '{split_name}'!"
                )
            return

        # --- Cross-Tokenizer Verification for Different Models ---

        # Step 1: Assert 'text' and 'task' columns exist in both batches
        for col in ("text", "task"):
            if col not in batch_inference or col not in batch_export:
                raise KeyError(
                    f"Missing required column '{col}' during cross-tokenizer check on split '{split_name}'! "
                    f"Inference keys: {list(batch_inference.keys())}, Export keys: {list(batch_export.keys())}"
                )
        tokenizer_inf = _get_tokenizer(config.model_name_or_path)
        tokenizer_exp = _get_tokenizer(export_model_name)

        # Step 2: Check batch_inference['text'] with export tokenizer
        _verify_tokenization_match(
            texts=batch_inference["text"],
            expected_input_ids=batch_export["input_ids"],
            tokenizer=tokenizer_exp,
            model_name=export_model_name,
            split_name=split_name,
            source_name="batch_inference['text']",
            target_name="batch_export['input_ids']",
        )

        # Step 3: Check batch_export['text'] with inference tokenizer
        _verify_tokenization_match(
            texts=batch_export["text"],
            expected_input_ids=batch_inference["input_ids"],
            tokenizer=tokenizer_inf,
            model_name=config.model_name_or_path,
            split_name=split_name,
            source_name="batch_export['text']",
            target_name="batch_inference['input_ids']",
        )
        

    @staticmethod
    def compute_logits_from_dataloader_inference(batch_inf, device, model, config):
        # Extract inputs from dataloader_inference to obtain teacher predictions
        input_ids_inf = batch_inf["input_ids"].to(device)
        attention_mask_inf = batch_inf["attention_mask"].to(device)
        token_type_ids_inf = batch_inf.get("token_type_ids")
        if token_type_ids_inf is not None:
            token_type_ids_inf = token_type_ids_inf.to(device)
        # Forward pass
        logits = model(input_ids_inf, attention_mask_inf, token_type_ids_inf)
        # Compute probabilities based on task 
        if config.problem_type == "multi_label":
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=-1)

        return logits, probs

    @staticmethod
    def extract_columns_from_dataloader_export(batch_exp, sample_idx_exp):
        # 3. Extract columns from dataloader_export for disk serialization
        exp_input_ids = batch_exp["input_ids"]
        exp_attention_mask = batch_exp["attention_mask"]
        exp_token_type_ids = batch_exp.get("token_type_ids")
        labels = batch_exp["labels"]
        sample_idx = sample_idx_exp
        # Parse task from export batch
        task_val = batch_exp["task"]
        if isinstance(task_val, (list, tuple)):
            task_tensor = SoftTargetExporter._encode_string_list(list(task_val))
        elif isinstance(task_val, torch.Tensor):
            task_tensor = task_val.cpu()
        else:
            task_tensor = SoftTargetExporter._encode_string_list([str(task_val)] * exp_input_ids.shape[0]) 

        return exp_input_ids, exp_attention_mask, exp_token_type_ids, labels, sample_idx, task_tensor

    @staticmethod
    def apend_tensors_to_buffers(tensor_buffers, logits, probs, batch_exp, sample_idx_exp):
        exp_input_ids, exp_attention_mask, exp_token_type_ids, labels, sample_idx, task_tensor = SoftTargetExporter.extract_columns_from_dataloader_export(batch_exp, sample_idx_exp)
        # Append tensors to buffers (logits/probs from inference pass; tokens/labels/metadata from export loader)
        tensor_buffers["input_ids"].append(exp_input_ids.cpu())
        tensor_buffers["attention_mask"].append(exp_attention_mask.cpu())
        if exp_token_type_ids is not None:
            tensor_buffers["token_type_ids"].append(exp_token_type_ids.cpu())
        tensor_buffers["logits"].append(logits.cpu())
        tensor_buffers["probabilities"].append(probs.cpu())
        tensor_buffers["labels"].append(labels.cpu())
        tensor_buffers["sample_index"].append(sample_idx.cpu())
        tensor_buffers["task"].append(task_tensor)

        return tensor_buffers

    @staticmethod
    def finalize_batch_for_export(tensor_buffers, config, dataloader_inference):
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
        final_sample_index = payload["sample_index"]
        final_task = payload["task"]
        
        # Structural Sanity Assertions
        assert final_logits.shape[0] == final_input_ids.shape[0], "Batch size mismatch between logits and input_ids"
        assert final_logits.shape[0] == len(dataloader_inference.dataset), "Sample count mismatch in output"                  # type: ignore
        assert final_logits.shape[1] == config.num_labels, "Logit dimension mismatch with label count"
        assert final_sample_index.shape[0] == final_logits.shape[0], "Sample index count mismatch with logits"
        assert final_task.shape[0] == final_logits.shape[0], "Task count mismatch with logits"
        
        # Prepare SafeTensors state dictionary payload 
        payload = {
            "input_ids": final_input_ids.contiguous(),
            "attention_mask": final_attention_masks.contiguous(),
            "logits": final_logits.contiguous(),
            "probabilities": final_probs.contiguous(),
            "labels": final_labels.contiguous(),
            "sample_index": final_sample_index.contiguous(),
            "task": final_task.contiguous()
        }
        if final_token_type_ids is not None:
            payload["token_type_ids"] = final_token_type_ids.contiguous()

        return payload, final_logits

    @staticmethod
    def export_all_splits(model: torch.nn.Module, dataloaders_inference: dict, dataloaders_export: dict, config: ModelConfig) -> None:
        for split_name, dataloader_inference in dataloaders_inference.items():

            if split_name not in dataloaders_export:
                raise KeyError( f"Split '{split_name}' is present in dataloaders_inference but missing from dataloaders_export.")
            dataloader_export = dataloaders_export[split_name]

            SoftTargetExporter.export(model, dataloader_inference, dataloader_export, config, split_name)

    # Main method executing inference on a split and saving model predictions to disk.
    @staticmethod
    @torch.no_grad()
    def export(model: torch.nn.Module, dataloader_inference: DataLoader, dataloader_export: DataLoader, config: ModelConfig, split_name: str) -> None:
        # Enforce sequential in-order processing for both dataloaders
        dataloader_inference = SoftTargetExporter._as_in_order_loader(dataloader_inference)
        dataloader_export = SoftTargetExporter._as_in_order_loader(dataloader_export)

        SoftTargetExporter.check_correct_dataloaders_length(dataloader_inference , dataloader_export, split_name)
        
        model.eval() 
        device = torch.device(config.device)
        model.to(device)

        tensor_buffers: dict[str, list[torch.Tensor]] = defaultdict(list)
        logger.info(f"Extracting soft labels for task: {config.task_name}_{config.unique_id_for_dir}, split: {split_name}")

        # The source for labels, sample_idx, task_val shouldn't matter because it should be the same for the 2 dataloaders.
        # token_type_ids, attention_mask, input_ids should have one for both inference and export (here correct transition must be ensured)
        for batch_inf, batch_exp in zip( tqdm(dataloader_inference, desc=f"Exporting {split_name}"), dataloader_export):
            sample_idx_inf = batch_inf["sample_index"]
            sample_idx_exp = batch_exp["sample_index"]
            if not isinstance(sample_idx_inf, torch.Tensor):
                sample_idx_inf = torch.tensor(sample_idx_inf, dtype=torch.int64)
            if not isinstance(sample_idx_exp, torch.Tensor):
                sample_idx_exp = torch.tensor(sample_idx_exp, dtype=torch.int64)

            SoftTargetExporter.check_correct_transition_between_tokenizers(config, batch_inf , batch_exp, sample_idx_inf, sample_idx_exp, split_name)

            logits, probs = SoftTargetExporter.compute_logits_from_dataloader_inference(batch_inf, device, model, config)

            tensor_buffers = SoftTargetExporter.apend_tensors_to_buffers(tensor_buffers, logits, probs, batch_exp, sample_idx_exp)                        

        
        payload, final_logits = SoftTargetExporter.finalize_batch_for_export(tensor_buffers, config, dataloader_inference)
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