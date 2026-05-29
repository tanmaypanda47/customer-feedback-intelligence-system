from fastapi import FastAPI
from pydantic import BaseModel

from src.inference import predict

app = FastAPI()


class ReviewRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "message":
        "Customer Feedback Intelligence API"
    }


@app.post("/predict")
def predict_review(
    request: ReviewRequest
):

    return predict(
        request.text
    )