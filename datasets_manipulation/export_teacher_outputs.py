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

            if split_name not in dataloaders_export:
                raise KeyError( f"Split '{split_name}' is present in dataloaders_inference but missing from dataloaders_export.")
            dataloader_export = dataloaders_export[split_name]

            SoftTargetExporter.export(model, dataloader_inference, dataloader_export, config, split_name)

    # Main method executing inference on a split and saving model predictions to disk.
    # TODO:Since teacher might be using different tokenizer we use dataloader_inference to get the teacher logits and 
    # dataloader_export to get the rest of the columns for export.
    @staticmethod
    @torch.no_grad()
    def export(model: torch.nn.Module, dataloader_inference: DataLoader, dataloader_export: DataLoader, config: ModelConfig, split_name: str) -> None:
        # Enforce sequential in-order processing for both dataloaders
        dataloader_inference = SoftTargetExporter._as_in_order_loader(dataloader_inference)
        dataloader_export = SoftTargetExporter._as_in_order_loader(dataloader_export)

        if len(dataloader_inference) == 0 or len(dataloader_export) == 0:
            raise ValueError(f"Cannot export soft targets for empty split: {split_name}")

        if len(dataloader_inference.dataset) != len(dataloader_export.dataset):  # type: ignore
            raise ValueError(
                f"Dataset length mismatch for split '{split_name}': "
                f"inference dataset has {len(dataloader_inference.dataset)} samples, "  # type: ignore
                f"export dataset has {len(dataloader_export.dataset)} samples."  # type: ignore
            )
        
        model.eval() 
        device = torch.device(config.device)
        model.to(device)

        tensor_buffers: dict[str, list[torch.Tensor]] = defaultdict(list)

        logger.info(f"Extracting soft labels for task: {config.task_name}_{config.unique_id_for_dir}, split: {split_name}")

        #TODO: The source for labels, sample_idx, task_val shouldn't matter because it should be the same for the 2 dataloaders
        # token_type_ids, attention_mask, input_ids should have one for both inference and export (here correct transition must be ensured)
        for batch_inf, batch_exp in zip( tqdm(dataloader_inference, desc=f"Exporting {split_name}"), dataloader_export):
            sample_idx_inf = batch_inf["sample_index"]
            sample_idx_exp = batch_exp["sample_index"]

            if not isinstance(sample_idx_inf, torch.Tensor):
                sample_idx_inf = torch.tensor(sample_idx_inf, dtype=torch.int64)
            if not isinstance(sample_idx_exp, torch.Tensor):
                sample_idx_exp = torch.tensor(sample_idx_exp, dtype=torch.int64)

            # Alignment check: ensure both streams process the exact same samples in order
            if not torch.equal(sample_idx_inf, sample_idx_exp):
                raise ValueError(
                    f"Sample index mismatch during export on split '{split_name}'! "
                    f"Inference indices: {sample_idx_inf.tolist()}, Export indices: {sample_idx_exp.tolist()}"
                )

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