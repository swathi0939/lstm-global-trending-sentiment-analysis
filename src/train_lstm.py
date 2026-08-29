"""
Train LSTM sentiment classifier on Global Trending Topics 2026 (rows 140-210).

Pipeline:
  headline + short_text -> preprocessing -> tokenize -> pad
  -> Embedding -> LSTM -> Dropout -> Dense(relu) -> Dense(softmax)
  -> positive / negative / neutral
"""
import os
import sys
import json

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical

sys.path.append(os.path.dirname(__file__))
from preprocess import load_and_preprocess, VOCAB_SIZE, MAX_LEN, LABELS, MODEL_DIR

RESULTS_PATH = os.path.join(MODEL_DIR, "training_results.json")


def build_model(vocab_size=VOCAB_SIZE, embed_dim=128, max_len=MAX_LEN, lstm_units=64):
    model = Sequential([
        Embedding(vocab_size, embed_dim, input_length=max_len),
        LSTM(lstm_units),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    df, X, y, tokenizer = load_and_preprocess(fit_tokenizer=True)
    print(f"Loaded {len(df)} records. Sentiment counts:\n{df['sentiment'].value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    model = build_model()
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=15,
        batch_size=8,
        verbose=2,
    )

    # ---- Evaluation on held-out test set ----
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    report = classification_report(
        y_test, y_pred, target_names=LABELS, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== TEST RESULTS (actual, not fabricated) ===")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix (rows=true, cols=pred), label order:", LABELS)
    print(cm)

    # ---- Save model ----
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "lstm_sentiment_model.keras")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # ---- Save results for reporting (real numbers only) ----
    results = {
        "records_total": int(len(df)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "epochs": 15,
        "test_accuracy": float(test_accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "confusion_matrix": cm.tolist(),
        "labels": LABELS,
        "classification_report": report,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
