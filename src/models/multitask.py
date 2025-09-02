import torch
from torch import nn
from transformers import AutoModel, T5EncoderModel


class MultiTaskClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 3) -> None:
        super().__init__()

        if "t5" in model_name:
            self.encoder = T5EncoderModel.from_pretrained(model_name)
        else:
            self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = (
            self.encoder.config.d_model if "t5" in model_name else self.encoder.config.hidden_size
        )
        self.dropout = nn.Dropout(0.2)
        self.mistake_head = nn.Linear(hidden_size, num_labels)
        self.guidance_head = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # If model has pooler_output (BERT-like)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # fallback: mean pooling over tokens
            pooled = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1)
            pooled = pooled / attention_mask.sum(1, keepdim=True)

        pooled = self.dropout(pooled)
        return self.mistake_head(pooled), self.guidance_head(pooled)
