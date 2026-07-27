from collections import defaultdict
import logging
import os
import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file
from tqdm import tqdm
from configs.model_config import ModelConfig

logger = logging.getLogger(__name__)

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
    def export_all_splits(model: torch.nn.Module, dataloaders_inference: dict, dataloaders_export: dict, config: ModelConfig) -> None:
        for split_name, dataloader_inference in dataloaders_inference.items():
            SoftTargetExporter.export(model, dataloader_inference, dataloader_inference, config, split_name)

    # Main method executing inference on a split and saving model predictions to disk.
    # TODO:Since teacher might be using different tokenizer we use dataloader_inference to get the teacher logits and 
    # dataloader_export to get the rest of the columns for export.
    @staticmethod
    @torch.no_grad()
    def export(model: torch.nn.Module, dataloader_inference: DataLoader, dataloader_export: DataLoader, config: ModelConfig, split_name: str) -> None:
        dataloader_inference = SoftTargetExporter._as_in_order_loader(dataloader_inference)

        if len(dataloader_inference) == 0:
            raise ValueError(f"Cannot export soft targets for empty split: {split_name}")
        
        model.eval() 
        device = torch.device(config.device)
        model.to(device)

        tensor_buffers: dict[str, list[torch.Tensor]] = defaultdict(list)

        logger.info(f"Extracting soft labels for task: {config.task_name}_{config.unique_id_for_dir}, split: {split_name}")

        #TODO: change the source for labels, sample_idx, task_val
        # token_type_ids, attention_mask, input_ids should have one for both inference and export (here correct transition must ensureed)
        for batch in tqdm(dataloader_inference, desc=f"Exporting {split_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"]
            
            # 1. Safely parse sample_index (convert to int64 Tensor if it's a Python list)
            sample_idx = batch["sample_index"]
            if not isinstance(sample_idx, torch.Tensor):
                sample_idx = torch.tensor(sample_idx, dtype=torch.int64)

            # 2. Safely parse task 
            task_val = batch["task"]
            if isinstance(task_val, (list, tuple)):
                task_tensor = SoftTargetExporter._encode_string_list(list(task_val))
            elif isinstance(task_val, torch.Tensor):
                task_tensor = task_val.cpu()
            else:
                task_tensor = SoftTargetExporter._encode_string_list([str(task_val)] * input_ids.shape[0])
            
            # Forward pass to get logits
            logits = model(input_ids, attention_mask, token_type_ids)
            
            # Compute probabilities based on task 
            if config.problem_type == "multi_label":
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=-1)

            # Move data to CPU and append to buffers
            tensor_buffers["input_ids"].append(input_ids.cpu())
            tensor_buffers["attention_mask"].append(attention_mask.cpu())
            if token_type_ids is not None:
                tensor_buffers["token_type_ids"].append(token_type_ids.cpu())
            tensor_buffers["logits"].append(logits.cpu())
            tensor_buffers["probabilities"].append(probs.cpu())
            tensor_buffers["labels"].append(labels.cpu())
            tensor_buffers["sample_index"].append(sample_idx.cpu())
            tensor_buffers["task"].append(task_tensor)

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