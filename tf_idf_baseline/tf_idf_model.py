import torch
import torch.nn as nn
from configs.model_config import TfidfBaselineConfig

class TfidfModel(nn.Module):
    def __init__(self, config: TfidfBaselineConfig):
        super().__init__()
        self.config = config
        input_dim = config.max_features
        num_labels = config.num_labels
        hidden_dim = config.hidden_dim

        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_labels)
            )
        else:
            self.net = nn.Linear(input_dim, num_labels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor | None = None, 
        token_type_ids: torch.Tensor | None = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Accepts input_ids as the dense float feature matrix tensor [batch_size, max_features].
        Accepts attention_mask, token_type_ids, and kwargs for signature compatibility 
        with LegalModel during evaluation loops.
        """
        return self.net(input_ids)