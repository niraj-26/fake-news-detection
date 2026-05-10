from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf

from transformers import (
    DistilBertTokenizerFast,
    TFDistilBertForSequenceClassification
)

# ==============================
# Create FastAPI app
# ==============================
app = FastAPI(
    title="Fake News Detection API",
    description="DistilBERT based Fake News Classifier",
    version="1.0"
)

# ==============================
# Load Saved Model & Tokenizer
# ==============================
MODEL_PATH = "distilbert_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)

model = TFDistilBertForSequenceClassification.from_pretrained(
    MODEL_PATH
)

# ==============================
# Input Schema
# ==============================
class NewsRequest(BaseModel):
    text: str

# ==============================
# Home Route
# ==============================
@app.get("/")
def home():
    return {
        "message": "Fake News Detection API is Running"
    }

# ==============================
# Prediction Route
# ==============================
@app.post("/predict")
async def predict(news: NewsRequest):

    # Tokenize input text
    
    inputs = tokenizer(
        news.text,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=64
    )

    # Model prediction
    outputs = model(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask']
    )

    logits = outputs.logits

    # Convert to probabilities
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]

    real_prob = probs[0]
    fake_prob = probs[1]

    if real_prob > fake_prob:
        prediction = 0
        label = "Real News"
        confidence = float(real_prob)

    else:
        prediction = 1
        label = "Fake News"
        confidence = float(fake_prob)

    return {
        "prediction": prediction,
        "label": label,
        "confidence": round(confidence, 4),
        "real_probability": round(float(real_prob), 4),
        "fake_probability": round(float(fake_prob), 4)
    }