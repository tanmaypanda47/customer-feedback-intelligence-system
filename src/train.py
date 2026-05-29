import torch
import torch.nn as nn

from datasets import load_dataset
from src.dataset import ReviewDataset

from transformers import AutoTokenizer

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src.model import SentimentRoBERTa

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)




def train():

    dataset = load_dataset(
        "amazon_polarity"
    )

    texts = dataset["train"]["content"][:500]

    labels = dataset["train"]["label"][:500]

    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-base"
    )

    train_dataset = ReviewDataset(
        texts,
        labels,
        tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )

    model = SentimentRoBERTa().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5
    )

    loss_fn = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(2):

        total_loss = 0

        for batch_idx, batch in enumerate(
            train_loader
        ):

            input_ids = batch[
                "input_ids"
            ].to(DEVICE)

            attention_mask = batch[
                "attention_mask"
            ].to(DEVICE)

            labels = batch[
                "label"
            ].to(DEVICE)

            optimizer.zero_grad()

            logits = model(
                input_ids,
                attention_mask
            )

            loss = loss_fn(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:

                print(
                    f"Epoch {epoch+1} | "
                    f"Batch {batch_idx} | "
                    f"Loss {loss.item():.4f}"
                )

        print(
            f"Epoch {epoch+1} "
            f"Loss {total_loss:.4f}"
        )

    torch.save(
        model.state_dict(),
        "models/model.pt"
    )

    print("Model Saved")


if __name__ == "__main__":
    train()