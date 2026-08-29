"""
Predict sentiment for a single piece of news/social text using the trained
LSTM model and saved tokenizer (no retraining).

Usage:
    python predict.py "Technology stocks are showing strong growth today."
    python predict.py            # interactive prompt
"""
import os
import sys
import pickle

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.append(os.path.dirname(__file__))
from preprocess import clean_text

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_sentiment_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")


def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        bundle = pickle.load(f)
    return model, bundle["tokenizer"], bundle["max_length"], bundle["idx2label"]


def predict_sentiment(text: str, model, tokenizer, max_length, idx2label):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_length, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx]) * 100
    return idx2label[pred_idx], confidence, probs


def main():
    model, tokenizer, max_length, idx2label = load_artifacts()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter a news/social-media text: ")

    label, confidence, probs = predict_sentiment(text, model, tokenizer, max_length, idx2label)

    print(f"\nInput:\n\"{text}\"")
    print(f"\nPrediction:\n{label.capitalize()}")
    print(f"\nConfidence:\n{confidence:.2f}%")


if __name__ == "__main__":
    main()
