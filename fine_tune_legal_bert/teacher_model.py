import torch
import torch.nn as nn
from transformers import AutoModel
from configs.teacher_config import TeacherConfig

class TeacherModel(nn.Module):
    def __init__(self, config: TeacherConfig):
        super(TeacherModel, self).__init__()
        self.config = config

        # Hugging Face loads the pretrained model, we load the Transformer encoder and build our own classifier
        self.encoder = AutoModel.from_pretrained(config.model_name_or_path)

        dropout_prob = getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Classification head
        self.classifier = nn.Linear(self.encoder.config.hidden_size, config.num_labels)

        # Initialize the classification head for stable initial gradients.
        nn.init.xavier_normal_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    # Defines what happens when we call: model(input_ids, attention_mask, token_type_ids)  
    def forward( self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        # The Transformer processes the entire sequence
        encoder_inputs = { "input_ids": input_ids, "attention_mask": attention_mask }

        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        # Unpack the dictionary and feed its keys and values to encoder
        outputs = self.encoder(**encoder_inputs)
        
        # Extract the representation of the [CLS] token for sequence-level classification
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Apply dropout and then pass through the classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
