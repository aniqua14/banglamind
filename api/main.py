# api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
# ── App instance ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="BanglaMind — Emotion Classifier",
    description="Detects emotions in Bangla social media text using BanglaBERT",
    version="2.0.0"
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "aniqua-nawar/banglamind-emotion")

ID2LABEL = {
    0: "joy",
    1: "sadness",
    2: "anger"
}

# ── Load model at startup ─────────────────────────────────────────────────────
print("Loading BanglaBERT model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"BanglaBERT loaded on {device}")

# ── Request schema ────────────────────────────────────────────────────────────


class TextInput(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "আমি আজকে খুব খুশি"
            }
        }

# ── Response schema ───────────────────────────────────────────────────────────


class EmotionOutput(BaseModel):
    text: str
    emotion: str
    confidence: float
    all_scores: dict

# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/")
def root():
    return {
        "app": "BanglaMind",
        "version": "2.0.0",
        "model": "BanglaBERT fine-tuned on EmoNoBa",
        "emotions": list(ID2LABEL.values()),
        "language": "Bangla (Bengali)",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": True,
        "model": MODEL_PATH
    }


@app.post("/predict", response_model=EmotionOutput)
def predict(input: TextInput):
    # Validate
    if not input.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    # Tokenize
    inputs = tokenizer(
        input.text,
        return_tensors='pt',
        truncation=True,
        max_length=128,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    predicted_id = int(np.argmax(probs))
    predicted_label = ID2LABEL[predicted_id]
    confidence = float(probs[predicted_id])

    all_scores = {
        ID2LABEL[i]: round(float(probs[i]), 4)
        for i in range(len(ID2LABEL))
    }

    return EmotionOutput(
        text=input.text,
        emotion=predicted_label,
        confidence=round(confidence, 4),
        all_scores=all_scores
    )
