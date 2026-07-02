import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file, load_file
import os
import logging
from tqdm import tqdm
from configs.model_config import ModelConfig

logger = logging.getLogger(__name__)

class SoftTargetExporter:
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
            accumulated_labels.append(labels)
            
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
        assert (final_logits.shape[0] == final_input_ids.shape[0]), "Batch size mismatch between logits and input_ids"
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
        #print(f"DEBUG: Saving to path: {export_path}")
        save_file(payload, export_path, metadata=metadata)
        logger.info(f"Successfully serialized soft targets to {export_path}")

    @staticmethod
    def verify_unfair_tos_exports(directory_path="./datasets_store/ds_with_teacher_outputs/unfair_tos_teacher_outputs"):
        """
        Scans the outputs folder, loads a generated SafeTensors split,
        checks if all mandatory columns are present, and displays the dimensions 
        and a formatted sample at index 0.
        """
        if not os.path.exists(directory_path):
            print(f"Error: Target directory '{directory_path}' does not exist.")
            print("Make sure you have run the teacher fine-tuning program first.")
            return

        # Scan the directory for serialized teacher outputs
        files = [f for f in os.listdir(directory_path) if f.endswith(".safetensors")]
        if not files:
            print(f"Error: No .safetensors files found in '{directory_path}'.")
            return

        # Select the first available split to inspect (e.g., validation or test)
        target_file = files[0]
        file_path = os.path.join(directory_path, target_file)
        
        print("=" * 60)
        print(f"Found Exported File: {target_file}")
        print(f"Full Path:           {file_path}")
        print("=" * 60)

        try:
            # Load the dictionary of tensors directly from the SafeTensors file
            tensors = load_file(file_path)
        except Exception as e:
            print(f"Error loading safetensors file: {e}")
            return

        # These are the columns expected from your model pipeline exports
        expected_columns = {"input_ids", "attention_mask", "logits", "probabilities", "labels"}
        actual_columns = set(tensors.keys())

        # 1. Column Integrity Validation
        print("\n[1] Column Presence Verification:")
        all_valid = True
        for col in expected_columns:
            present = col in actual_columns
            status = "✓ PRESENT" if present else "✗ MISSING"
            print(f"  - {col:<16}: {status}")
            if not present:
                all_valid = False

        if not all_valid:
            print("\n⚠️ WARNING: Your exported data is missing key fields required for distillation!")
        else:
            print("\n✓ SUCCESS: All essential columns are correctly serialized.")

        # 2. Schema and Datatype Summary
        print("\n[2] Tensor Metadata & Specifications:")
        for key, tensor in tensors.items():
            shape_str = str(list(tensor.shape))
            print(f"  - {key:<16}: shape={shape_str:<18} | dtype={str(tensor.dtype):<13} | device={tensor.device}")

        # 3. Shape Sanity Checking
        print("\n[3] Pipeline Integrity Checks:")
        num_samples = tensors["logits"].shape[0]
        num_classes = tensors["logits"].shape[1]
        
        # Assert validation check to prevent bad logic downstream
        assert num_classes == 8, f"Expected 8 label dimensions for UNFAIR-ToS, found {num_classes}."
        print(f"  ✓ Label cardinality is correctly matched to 8 classes.")
        
        for key in expected_columns:
            assert tensors[key].shape[0] == num_samples, f"Batch size mismatch on column: {key}!"
        print(f"  ✓ Sample count is consistent across all {num_samples} records.")

        # 4. Content Inspection of Sample 0
        print("\n" + "=" * 60)
        print("SAMPLE RECORD INSPECTION (Index 0)")
        print("=" * 60)
        
        idx = 0
        # Retrieve tokens, masking out padding values (0) to keep stdout readable
        raw_tokens = tensors["input_ids"][idx].tolist()
        clean_tokens = [tok for tok in raw_tokens if tok != 0]
        
        print(f"Input IDs (trimmed length={len(clean_tokens)}):\n  {clean_tokens}")
        print(f"Attention Mask (first 30 steps):\n  {tensors['attention_mask'][idx][:30].tolist()}")
        
        # Check probabilities and ground truths
        labels_list = tensors["labels"][idx].tolist()
        probs_list = tensors["probabilities"][idx].tolist()
        logits_list = tensors["logits"][idx].tolist()

        print(f"\nGround Truth Multi-Labels:\n  {labels_list}")
        print(f"\nModel Raw Logits:\n  {[round(x, 4) for x in logits_list]}")
        print(f"\nTeacher Output Probabilities (Sigmoids):\n  {[round(x, 4) for x in probs_list]}")
        print("=" * 60)