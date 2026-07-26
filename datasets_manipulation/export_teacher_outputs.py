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
        
        model.eval() # turn off dropout and other evaluation-specific layers
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

            # Move data to CPU and append to buffers
            tensor_buffers["input_ids"].append(input_ids.cpu())
            tensor_buffers["attention_mask"].append(attention_mask.cpu())
            if token_type_ids is not None:
                tensor_buffers["token_type_ids"].append(token_type_ids.cpu())
            tensor_buffers["logits"].append(logits.cpu())
            tensor_buffers["probabilities"].append(probs.cpu())
            tensor_buffers["labels"].append(labels.cpu())

            
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
        assert final_logits.shape[0] == len(dataloader.dataset), "Sample count mismatch in output"                  # type: ignore
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

