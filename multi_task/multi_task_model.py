from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel
from configs.model_config import ModelConfig


class MultiTaskModel(nn.Module):
    """Shared encoder with task-specific classification heads for LEDGAR and UNFAIR-ToS."""

    TASK_NAMES = ("ledgar", "unfair_tos")

    def __init__(self, ledgar_config: ModelConfig, unfair_tos_config: ModelConfig):
        super().__init__()

        # A safety check. Because both tasks are going to share the exact same language model backbone, their underlying configuration strings must match exactly
        if ledgar_config.model_name_or_path != unfair_tos_config.model_name_or_path:
            raise ValueError(
                "MultiTaskModel expects both task configs to use the same encoder checkpoint."
            )

        self.ledgar_config = ledgar_config
        self.unfair_tos_config = unfair_tos_config

        # This encoder will be shared globally across both tasks.
        self.encoder = AutoModel.from_pretrained(ledgar_config.model_name_or_path)

        # Safely looks up the pre-defined dropout rate inside the Hugging Face configuration file. If the model doesn't specify one, it defaults to 0.1 (10%).
        dropout_prob = getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)

        # Dynamically grabs the dimensionality of the encoder's output space
        hidden_size = self.encoder.config.hidden_size

        # Sets up two unique linear classification heads. Both accept the hidden_size vector from the encoder, 
        # but map to different output numbers depending on how many classes each dataset has
        self.classifier_heads = nn.ModuleDict(
            {
                "ledgar": nn.Linear(hidden_size, ledgar_config.num_labels),
                "unfair_tos": nn.Linear(hidden_size, unfair_tos_config.num_labels),
            }
        )

        # Loops through both linear classification layers to manually set their starting weights.
        # Fills the weight matrices using a Xavier/Glorot distribution, keeping the variance of gradients consistent across layers
        for head in self.classifier_heads.values():
            nn.init.xavier_normal_(head.weight) #type: ignore         
            if head.bias is not None:
                nn.init.zeros_(head.bias) #type: ignore     

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor | None = None, task: str | None = None,) -> torch.Tensor:
        # It accepts standard tokenized tensor inputs, along with a mandatory task string indicating which dataset the current batch belongs to.
        if task is None:
            raise ValueError("MultiTaskModel.forward requires a task name.")
        if task not in self.classifier_heads:
            raise ValueError(f"Unsupported task '{task}'. Expected one of {sorted(self.classifier_heads.keys())}.")

        encoder_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_inputs)

        #Extracts the sentence-level representation. outputs.last_hidden_state is a 3D tensor of shape [batch_size, sequence_length, hidden_size]. 
        # The [:, 0, :] slice grabs the very first token (index 0) for every sentence in the batch, which is specifically the [CLS] token used for classification tasks.
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        # Dynamically queries ModuleDict using the task string, passes the pooled representation into that specific linear layer, and returns the logits
        return self.classifier_heads[task](pooled_output)
