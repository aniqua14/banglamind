---
title: BanglaMind
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# 🧠 BanglaMind
## Emotion Detection in Bangla Social Media Text

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/aniqua-nawar/banglamind)
[![GitHub](https://img.shields.io/badge/GitHub-banglamind-black)](https://github.com/aniqua14/banglamind)
[![Model](https://img.shields.io/badge/Model-BanglaBERT-orange)](https://huggingface.co/aniqua-nawar/banglamind-emotion)

---

## 🌐 Live Demo

👉 **[Try BanglaMind on Hugging Face Spaces](https://huggingface.co/spaces/aniqua-nawar/banglamind)**

![BanglaMind Demo](https://raw.githubusercontent.com/aniqua14/banglamind/main/assets/demo.png)

---

## 📌 Overview

**BanglaMind** is an end-to-end NLP system for detecting emotions in Bangla social media text. It fine-tunes **BanglaBERT** — a BERT-based Large Language Model pretrained on Bangla text — for 3-class emotion classification:

| Emotion | Bangla | Example |
|---------|--------|---------|
| 😊 Joy | আনন্দ | আমি আজকে খুব খুশি |
| 😢 Sadness | দুঃখ | আমার মন আজ খুব খারাপ |
| 😠 Anger | রাগ | এটা দেখে আমি খুব রাগান্বিত |

This project frames emotion detection as an **LLM fine-tuning problem on a low-resource language** — addressing a genuine research gap in Bangla NLP.

---

## 🏗️ Architecture

```
Bangla Text Input
      ↓
BanglaBERT Tokenizer (Bangla-aware WordPiece)
      ↓
BanglaBERT Encoder (12 layers, 164M parameters)
      ↓
Classification Head (768 → 3)
      ↓
Softmax → {joy, sadness, anger}
```
### Why BanglaBERT over classical ML?

| Aspect | TF-IDF + LogReg | BanglaBERT |
|--------|----------------|------------|
| Context understanding | ✗ | ✓ |
| Negation handling | ✗ | ✓ |
| Bangla script support | Partial | Native |
| Parameters | ~10K features | 164M |

---

## 📊 Dataset

**EmoNoBa** — A Dataset for Analyzing Fine-Grained Emotions on Noisy Bangla Texts  
*(AACL-IJCNLP 2022)*

| Split | Samples |
|-------|---------|
| Train | 12,992 |
| Val | 1,443 |
| Test | 1,525 |

**Class distribution:**

```
Joy     : 6,277 (48%)  ████████████
Sadness : 3,963 (31%)  ████████
Anger   : 2,752 (21%)  █████
```
---

## 📈 Results

### Classical ML Baseline (English — dair-ai/emotion)

| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| TF-IDF + Naive Bayes | 86.3% | 83.5% |
| TF-IDF + Logistic Regression | 92.3% | 91.1% |

### LLM Fine-tuning (Bangla — EmoNoBa)

| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| mBERT (English) | 96.2% | 95.5% |
| **BanglaBERT (Bangla)** | **72.4%** | **71.2%** |

### Per-class F1 (BanglaBERT on Bangla)

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|----|
| Joy | 0.79 | 0.76 | 0.77 |
| Sadness | 0.74 | 0.75 | 0.74 |
| Anger | 0.60 | 0.64 | 0.62 |

> **Note:** Lower F1 on Bangla reflects EmoNoBa annotation noise —
> the original paper reports random baselines outperforming neural
> networks, indicating high label ambiguity in the dataset.

---

## 🗂️ Project Structure
```

banglamind/
├── app.py                    # Gradio frontend
├── Dockerfile                # Container configuration
├── requirements.txt          # Dependencies
├── api/
│   └── main.py              # FastAPI backend
├── src/
│   ├── data_loader.py       # Dataset loading functions
│   └── utils.py             # Text cleaning utilities
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_classical_ml.ipynb
└── assets/
└── demo.png

```
---

## 🚀 Run Locally

### Prerequisites
```bash
Python 3.10+
pip install -r requirements.txt
```

### Run Gradio app
```bash
python app.py
# Open http://localhost:7860
```

### Run FastAPI backend
```bash
uvicorn api.main:app --reload --port 8000
# API docs at http://localhost:8000/docs
```

### API usage
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "আমি আজকে খুব খুশি"}'
```

---

## ⚠️ Limitations

- **Sadness-Anger confusion** — EmoNoBa has high inter-annotator disagreement between these two classes
- **3-class only** — Fear excluded due to insufficient samples (n=162) in available Bangla datasets
- **Domain specific** — Trained on social media text; may not generalize to formal Bangla writing

---

## 🔬 Future Work

- Fine-tune on larger Bangla emotion dataset with cleaner annotations
- Add Fear class when sufficient annotated data is available
- Experiment with BanglaBERT-large and XLM-RoBERTa
- Build multilingual support (Bangla + English mixed text)

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Gradio](https://img.shields.io/badge/Gradio-4.44-purple)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

- **Model**: BanglaBERT (`sagorsarker/bangla-bert-base`)
- **Dataset**: EmoNoBa (AACL-IJCNLP 2022)
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Gradio
- **Deployment**: Hugging Face Spaces (Docker)

---

## 👩‍💻 Developer

**Aniqua Nawar**  
CS Graduate | NLP Researcher  
📧 [GitHub](https://github.com/aniqua14) | 🤗 [HuggingFace](https://huggingface.co/aniqua-nawar)

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@misc{nawar2026banglamind,
  title={BanglaMind: Emotion Detection in Bangla Social Media Text},
  author={Nawar, Aniqua},
  year={2026},
  url={https://huggingface.co/spaces/aniqua-nawar/banglamind}
}
```

**EmoNoBa dataset:**
```bibtex
@inproceedings{emonoba2022,
  title={EmoNoBa: A Dataset for Analyzing Fine-Grained Emotions on Noisy Bangla Texts},
  booktitle={AACL-IJCNLP 2022}
}
```

