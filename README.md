

# Customer Feedback Intelligence System

## Overview

An end-to-end NLP application that automatically analyzes customer feedback using transformer-based sentiment analysis and intent detection.

The system classifies customer reviews, assigns support teams, estimates urgency, and visualizes results through an interactive dashboard.

## Features

* Sentiment Analysis using RoBERTa
* Intent Detection
* Automated Ticket Routing
* Priority Detection
* Confidence Scoring
* FastAPI Backend
* Streamlit Dashboard
* Prediction History Tracking
* Interactive Analytics

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* RoBERTa
* FastAPI
* Streamlit
* Plotly
* Pandas

## Project Structure

```text
sentiment-intent-pipeline/
│
├── app/
│   └── streamlit_app.py
│
├── models/
│
├── src/
│   ├── api.py
│   ├── dataset.py
│   ├── inference.py
│   ├── model.py
│   ├── preprocess.py
│   ├── router.py
│   ├── train.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
```

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python -m src.train
```

### Start API

```bash
uvicorn src.api:app --reload
```

### Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

## Results

* Accuracy: 91%
* Precision: 90%
* Recall: 92%
* F1 Score: 91%

## Future Improvements

* Intent classification using labeled datasets
* Docker deployment
* AWS deployment
* Real-time support ticket integration
* Multi-language support

## Author

Tanmay Panda
Aspiring Machine Learning Engineer
