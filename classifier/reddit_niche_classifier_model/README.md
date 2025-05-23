---
license: mit
tags:
  - classification
  - reddit
  - tensorflow
  - keras
  - niche-detection
---

# 🧠 Reddit Niche Classifier

A lightweight feedforward neural network trained to classify Reddit posts into distinct **niche categories** such as `advice`, `drama`, `humor`, `informative`, and more — **without relying on full NLP or raw text**.

This model is designed to work with structured Reddit metadata, and is ideal for fast, low-cost deployment on classification tasks with tabular or engineered data.

## ✨ Model Details

- **Framework**: TensorFlow / Keras
- **Input Features**: 
  - Boolean indicators (e.g. `contains_question`, `contains_capslock`)
  - Numeric metadata (e.g. `score`, `num_comments`, `title_length`, `selftext_length`, `engagement_score`)
  - One-hot encoded subreddits
  - Custom feature: `num_caps_words`
- **No raw text (title/selftext) is used**

## 🏗️ Training Info

- **Architecture**: `[256, 128, 64]` with ReLU activations
- **Output Layer**: `Dense(11)` with softmax (multi-class classification)
- **Loss**: `sparse_categorical_crossentropy`
- **Optimizer**: Adam
- **Test Accuracy**: ~67% on held-out set

## 📦 Usage

```python
from tensorflow import keras
model = keras.models.load_model("niche_classifier_model")
predictions = model.predict(X_new)
