# Fake News Detection System

AI-powered Fake News Detection system using DistilBERT, FastAPI, Streamlit, TensorFlow, and Docker.

## Features
- DistilBERT NLP model
- FastAPI backend
- Streamlit frontend
- Docker support
- Real-time prediction
- Confidence score output

## Tech Stack
- Python
- TensorFlow
- Transformers
- FastAPI
- Streamlit
- Docker

## Run Backend

```bash

Install Dependencies:  pip install -r requirements.txt

Run FastAPI Server:  uvicorn main:app --reload

Backend Docs: http://127.0.0.1:8000/docs

Run Frontend:  streamlit run app.py

Docker Commands:
docker build -t fake-news-api .
docker run -p 8000:8000 fake-news-api
```

## API Endpoint

- API Request Example
{
  "text": "Breaking news example"
}

- API Response Example
{
  "prediction": 1,
  "label": "Fake News",
  "confidence": 0.91
}

## Author
- Neeraj Kumar Gupta
