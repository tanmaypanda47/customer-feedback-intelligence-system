import torch

from datasets import load_dataset

from transformers import AutoTokenizer

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src.model import SentimentRoBERTa

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


class ReviewDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_len=128
    ):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids":
                encoding["input_ids"].flatten(),

            "attention_mask":
                encoding["attention_mask"].flatten(),

            "label":
                torch.tensor(
                    self.labels[idx],
                    dtype=torch.long
                )
        }


def evaluate():

    dataset = load_dataset(
        "amazon_polarity"
    )

    texts = dataset["test"]["content"][:500]

    labels = dataset["test"]["label"][:500]

    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-base"
    )

    test_dataset = ReviewDataset(
        texts,
        labels,
        tokenizer
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8
    )

    model = SentimentRoBERTa()

    model.load_state_dict(
        torch.load(
            "models/model.pt",
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for batch in test_loader:

            input_ids = batch[
                "input_ids"
            ].to(DEVICE)

            attention_mask = batch[
                "attention_mask"
            ].to(DEVICE)

            labels = batch[
                "label"
            ].to(DEVICE)

            logits = model(
                input_ids,
                attention_mask
            )

            preds = torch.argmax(
                logits,
                dim=1
            )

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                preds.cpu().numpy()
            )

    print("\nClassification Report\n")

    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "Negative",
            "Positive"
        ]
    )

    print(report)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred
    )

    recall = recall_score(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred
    )

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1 Score: {f1:.4f}"
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\nConfusion Matrix\n")
    print(cm)

    pd.DataFrame(report.splitlines()).to_csv(
        "classification_report.csv",
        index=False
    )


if __name__ == "__main__":
    evaluate()