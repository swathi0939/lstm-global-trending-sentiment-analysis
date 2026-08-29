"""
Preprocessing for LSTM Sentiment Analysis
Dataset: trending_topics_rows_140_210.csv (Global Trending Topics 2026, rows 140-210)

Steps:
1. Load CSV
2. Combine headline + short_text -> text
3. Lowercase
4. Strip URLs / special symbols / extra whitespace (UTF-8 safe, multilingual)
5. Encode sentiment labels
6. Tokenize with Keras Tokenizer
7. Convert to sequences + pad
8. Save tokenizer, max_length, and label mapping to model/
"""
import os
import re
import pickle

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "trending_topics_rows_140_210.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

VOCAB_SIZE = 5000
MAX_LEN = 40  # generous cap; actual longest sequence in this dataset is shorter

LABELS = ["negative", "neutral", "positive"]
LABEL2IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX2LABEL = {idx: label for label, idx in LABEL2IDX.items()}


def clean_text(text: str) -> str:
    """Lowercase + strip URLs/special symbols/extra spaces. UTF-8 safe for
    multilingual (hi/en/ur) text: only ASCII punctuation is stripped, letters
    from any script (including Devanagari/Urdu) are preserved."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"[\"'“”‘’]", " ", text)                    # quote marks
    text = re.sub(r"[^\w\s\u0900-\u097F\u0600-\u06FF]", " ", text, flags=re.UNICODE)  # keep word chars + Devanagari + Arabic/Urdu ranges
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_preprocess(data_path: str = DATA_PATH, fit_tokenizer: bool = True):
    df = pd.read_csv(data_path, encoding="utf-8")

    # Combine headline + short_text
    df["text"] = (df["headline"].fillna("") + " " + df["short_text"].fillna("")).apply(clean_text)

    # Encode sentiment labels
    df["label_idx"] = df["sentiment"].map(LABEL2IDX)

    if fit_tokenizer:
        tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
        tokenizer.fit_on_texts(df["text"])
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
            pickle.dump({
                "tokenizer": tokenizer,
                "max_length": MAX_LEN,
                "label2idx": LABEL2IDX,
                "idx2label": IDX2LABEL,
            }, f)
    else:
        with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
            bundle = pickle.load(f)
        tokenizer = bundle["tokenizer"]

    sequences = tokenizer.texts_to_sequences(df["text"])
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
    y = df["label_idx"].values

    return df, X, y, tokenizer


if __name__ == "__main__":
    df, X, y, tokenizer = load_and_preprocess()
    print(f"Loaded {len(df)} records from {DATA_PATH}")
    print("Sentiment distribution:")
    print(df["sentiment"].value_counts())
    print(f"Vocabulary size (fitted): {len(tokenizer.word_index)}")
    print(f"Padded sequence shape: {X.shape}")
    print(f"Sample cleaned text: {df['text'].iloc[0]}")
    print(f"Tokenizer saved to: {os.path.join(MODEL_DIR, 'tokenizer.pkl')}")
