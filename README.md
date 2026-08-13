# 🧠 Suicide Risk Detection using Deep Learning

An NLP-based deep learning project exploring how text classification models can identify language patterns associated with suicide risk, for **research and educational purposes**.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![NLP](https://img.shields.io/badge/NLP-text%20classification-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

---

> ⚠️ **Disclaimer:** This project is intended strictly for research and educational purposes. It is **not** a diagnostic or clinical tool and must not be used to make real decisions about a person's mental health or safety. If you or someone you know is struggling, please reach out to a mental health professional or a crisis helpline in your region.

---

## Overview

This project applies deep learning to a text classification problem: identifying language patterns in text that may be associated with suicide risk. It walks through a full NLP pipeline — from data cleaning to model training and comparison — and includes a small app layer for running predictions on new text input.

## Features

- ✅ End-to-end NLP pipeline: cleaning, tokenization, and preprocessing
- ✅ Multiple deep learning architectures trained and compared (DNN, CNN, LSTM)
- ✅ Model evaluation and comparison to identify the strongest approach
- ✅ A lightweight app for testing predictions on new text

## Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python |
| **Deep Learning** | TensorFlow / Keras |
| **NLP** | Text preprocessing, tokenization, embeddings |
| **App layer** | Flask |
| **Notebook** | Jupyter |

## Model Comparison

*(Fill in with your actual results — a filled-in table is far more credible on a public repo than a placeholder.)*

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| DNN | — | — | — | — |
| CNN | — | — | — | — |
| LSTM | — | — | — | — |

## How It Works

1. **Data preprocessing** — raw text is cleaned, tokenized, and prepared for model input.
2. **Model training** — DNN, CNN, and LSTM architectures are each trained on the preprocessed dataset.
3. **Evaluation** — models are compared using standard classification metrics to identify the best-performing approach.
4. **Prediction app** — a small Flask interface lets you input text and see the model's classification.

## Getting Started

### Prerequisites

- Python 3.8+
- `pip`

### Installation

```bash
git clone https://github.com/Shahid-0039/Suicide-Detection-using-DL.git
cd Suicide-Detection-using-DL

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Run the training notebook

Open the training notebook in Jupyter or JupyterLab to reproduce preprocessing, training, and evaluation.

### Run the app

```bash
python app.py
```

## Limitations & Responsible Use

- The model reflects patterns in its training dataset and may not generalize to all forms of expression, languages, or contexts.
- False positives and false negatives are both possible and can have serious consequences if this tool is misused — it is not a substitute for professional judgment.
- This project should be used to study NLP/DL techniques and risk-language patterns, not deployed as a real-world screening tool without extensive clinical validation, ethical review, and professional oversight.

## Roadmap

- [ ] Fill in the model comparison table with real evaluation metrics
- [ ] Add a data source / dataset citation section
- [ ] Add unit tests for preprocessing functions
- [ ] Expand the disclaimer with links to crisis resources for repo visitors

## Author

Maintained by **[Shahid-0039](https://github.com/Shahid-0039)**
