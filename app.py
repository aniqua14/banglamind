# app.py
# Gradio frontend for BanglaMind
# This file is the entry point for HuggingFace Spaces

import gradio as gr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "aniqua-nawar/banglamind-emotion")

print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded on {device}")

# ── Label mapping ─────────────────────────────────────────────────────────────
ID2LABEL = {
    0: "joy",
    1: "sadness",
    2: "anger"
}

EMOTION_EMOJI = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠"
}

# ── Prediction function ───────────────────────────────────────────────────────


def predict_emotion(text):
    """
    Takes Bangla text input and returns emotion prediction.
    This function is called by Gradio on every button click.
    """
    if not text or not text.strip():
        return "Please enter some text.", {}

    # Tokenize
    inputs = tokenizer(
        text,
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
    emoji = EMOTION_EMOJI[predicted_label]

    # Format result label
    result = f"{emoji} {predicted_label.upper()} — {confidence:.2%}"

    # Format scores for Gradio label component
    scores = {
        f"{EMOTION_EMOJI[ID2LABEL[i]]} {ID2LABEL[i]}": float(probs[i])
        for i in range(len(ID2LABEL))
    }

    return result, scores


# ── Example texts ─────────────────────────────────────────────────────────────
examples = [
    ["আমি আজকে খুব খুশি"],           # I am very happy today
    ["আমার মন আজ খুব খারাপ"],         # I feel very sad today
    ["এটা দেখে আমি খুব রাগান্বিত"],   # I am very angry seeing this
    ["আমি খুব আনন্দিত"],              # I am very joyful
    ["এই কাজটা দেখে রাগ হচ্ছে"],      # This work makes me angry
]

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="BanglaMind — Bangla Emotion Detector",
) as demo:
    # Header
    gr.Markdown("""
    # 🧠 BanglaMind
    ## Emotion Detection in Bangla Social Media Text
    
    Fine-tuned **BanglaBERT** on **EmoNoBa** dataset for Bangla emotion classification.
    Detects **Joy**, **Sadness**, and **Anger** in Bangla text.
    
    > *Built as part of an NLP research project on low-resource language emotion detection.*
    """)

    with gr.Row():
        with gr.Column(scale=2):
            # Input
            text_input = gr.Textbox(
                label="Enter Bangla Text",
                placeholder="এখানে বাংলা লিখুন... (Write Bangla here...)",
                lines=3
            )

            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("Detect Emotion 🔍", variant="primary")

            # Examples
            gr.Examples(
                examples=examples,
                inputs=text_input,
                label="Example Texts"
            )

        with gr.Column(scale=1):
            # Output
            result_label = gr.Textbox(
                label="Predicted Emotion",
                interactive=False
            )
            score_label = gr.Label(
                label="Confidence Scores",
                num_top_classes=3
            )

    # Footer
    gr.Markdown("""
    ---
    **Model** : BanglaBERT (`sagorsarker/bangla-bert-base`) fine-tuned on EmoNoBa  
    **Dataset** : EmoNoBa — 22,698 Bangla social media comments (AACL-IJCNLP 2022)  
    **Developer** : Aniqua Nawar  
    **GitHub** : [banglamind](https://github.com/aniqua-nawar/banglamind)
    """)

    # Button actions
    submit_btn.click(
        fn=predict_emotion,
        inputs=text_input,
        outputs=[result_label, score_label]
    )
    clear_btn.click(
        fn=lambda: ("", None, ""),
        outputs=[text_input, score_label, result_label]
    )

# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        theme=gr.themes.Soft()
    )
