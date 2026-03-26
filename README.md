# Transformer vs RNN/LSTM Text Classification (IMDb)

## 📌 Project Overview

This project implements a complete NLP pipeline for **sentiment classification on the IMDb dataset**, with a focus on comparing different sequence modeling architectures:

* RNN
* LSTM
* Transformer Encoder

The goal is to understand the evolution of sequence models from **recurrent structures to attention-based models**, and evaluate their performance under the same experimental setup.

---

## 🎯 Objectives

* Build a full text classification pipeline from scratch
* Compare RNN, LSTM, and Transformer on the same dataset
* Analyze differences in long-range dependency modeling
* Gain hands-on experience with PyTorch NLP workflows

---

## 📂 Dataset

* Dataset: IMDb Movie Reviews
* Task: Binary Sentiment Classification (Positive / Negative)
* Size:

  * Train: 25,000 samples
  * Test: 25,000 samples

---

## 🧱 Project Structure

```
transformer-imdb-classification/
│
├── data/
├── outputs/
│   ├── models/
│   └── results.txt
├── src/
│   ├── dataset.py
│   ├── model_rnn.py
│   ├── model_transformer.py
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Pipeline

### 1. Text Preprocessing

* Lowercasing
* Simple tokenization
* Vocabulary construction (top-k frequency)
* Mapping tokens to indices

### 2. Sequence Processing

* Padding / truncation to fixed length
* Batch loading with DataLoader

### 3. Model Training

* Loss: CrossEntropyLoss
* Optimizer: Adam
* Device: GPU (CUDA)

---

## 🧠 Models

### RNN

* Vanilla recurrent neural network
* Sequential processing
* Weak long-range dependency modeling

### LSTM

* Gated recurrent unit
* Handles long-term dependencies better than RNN

### Transformer Encoder

* Multi-head self-attention
* Fully parallelizable
* Captures global context directly

---

## 📊 Experimental Results

### 🔥 Model Performance Comparison

| Model       | Best Val Acc | Best Val F1 |
| ----------- | -----------: | ----------: |
| RNN         |       0.6688 |      0.6637 |
| LSTM        |       0.8373 |      0.8317 |
| Transformer |   **0.8478** |  **0.8489** |

---

## 📈 Observations

* RNN performs significantly worse due to poor long-range dependency modeling
* LSTM improves performance via gating mechanisms
* Transformer achieves the best result by modeling global dependencies using self-attention
* Slight overfitting observed in later epochs (train loss ↓, val fluctuates)

---

## 🚀 Key Insights

* Sequence modeling evolves from:

  * **RNN → LSTM → Transformer**
* Transformer eliminates recurrence and enables:

  * Parallel computation
  * Better gradient flow
  * Stronger global context modeling

---

## 🛠️ Improvements

* Masked mean pooling (ignore padding tokens)
* Hyperparameter tuning (lr, dropout, max_len)
* Early stopping to reduce overfitting
* Pretrained embeddings (GloVe / Word2Vec)
* Fine-tuning BERT for stronger performance

---

## ▶️ How to Run

```bash
conda activate dl
python src/train.py
```

---

## 📌 Future Work

* Add training curves visualization
* Attention weight visualization
* Replace tokenizer with HuggingFace tokenizer
* Upgrade to BERT / RoBERTa

---

## 💬 Highlights

* Designed a controlled experiment comparing RNN, LSTM, and Transformer
* Built full NLP pipeline from raw text to model evaluation
* Demonstrated understanding of sequence modeling evolution
* Achieved 84.78% accuracy with Transformer on IMDb dataset

---

## ⭐ Summary

This project provides a practical comparison of sequence models and demonstrates how modern NLP systems transition from recurrence to attention-based architectures.
