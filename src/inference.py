import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

from src.model import SentimentRoBERTa

from src.preprocess import (
    assign_intent,
    INTENT_MAP
)

from src.router import (
    route_ticket,
    detect_priority
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SENTIMENT_MAP = {
    0: "Negative",
    1: "Positive"
}

tokenizer = AutoTokenizer.from_pretrained(
    "roberta-base"
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


def predict(text):

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)

    attention_mask = encoding[
        "attention_mask"
    ].to(DEVICE)

    with torch.no_grad():

        logits = model(
            input_ids,
            attention_mask
        )

        probs = F.softmax(
            logits,
            dim=1
        )

    sentiment_pred = torch.argmax(
        probs,
        dim=1
    ).item()

    confidence = probs[
        0
    ][sentiment_pred].item()

    positive_prob = probs[0][1].item()
    negative_prob = probs[0][0].item()

    intent_pred = assign_intent(text)

    intent = INTENT_MAP[intent_pred]

    return {

        "sentiment":
            SENTIMENT_MAP[sentiment_pred],

        "intent":
            intent,

        "confidence":
            round(confidence * 100, 2),

        "positive_prob":
            round(positive_prob * 100, 2),

        "negative_prob":
            round(negative_prob * 100, 2),

        "assigned_team":
            route_ticket(intent),

        "priority":
            detect_priority(text)
    }