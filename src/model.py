import torch.nn as nn
from transformers import AutoModel


class SentimentRoBERTa(nn.Module):

    def __init__(self):

        super().__init__()

        self.roberta = AutoModel.from_pretrained(
            "roberta-base"
        )

        hidden_size = self.roberta.config.hidden_size

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(
            hidden_size,
            2
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = outputs.last_hidden_state[:, 0]

        cls_output = self.dropout(cls_output)

        logits = self.classifier(cls_output)

        return logits