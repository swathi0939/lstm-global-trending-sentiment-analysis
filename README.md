# LSTM Sentiment Analysis — Global Trending Topics 2026

## Project Objective

Classify the sentiment (**positive / negative / neutral**) of trending news and
social-media text using an LSTM (Long Short-Term Memory) recurrent neural
network. The model reads a headline + short description as a sequence of
tokens and predicts the sentiment expressed in it.

## Dataset

**Global Trending Topics 2026** (news + social, multilingual)
**Dataset used in this project:** `data/trending_topics_rows_140_210.csv` — rows 140–210 only
**Number of records used:** 71

### Dataset columns

| Column | Description |
|---|---|
| `id` | Unique record identifier |
| `date` | Date of the trend/post |
| `source` | Data source (e.g. `news_api`, `google_trends`) |
| `language` | Language code (`hi`, `en`, `ur` in this subset) |
| `country` | Country code |
| `topic_category` | High-level topic category |
| `topic_subcategory` | Specific subcategory |
| `headline` | Headline text |
| `short_text` | Short description / social text |
| `sentiment` | Label — **target column** |
| `engagement_score` | Engagement metric |
| `trend_score` | Trend strength metric |

**Input columns (features):** `headline` + `short_text` (combined into a single `text` field)
**Target column:** `sentiment`
**Sentiment classes:** `positive`, `negative`, `neutral`

Class balance in this 71-row subset: 27 neutral, 23 negative, 21 positive.

## Preprocessing (`src/preprocess.py`)

1. Load `data/trending_topics_rows_140_210.csv` (UTF-8).
2. Combine `headline + " " + short_text` into a new `text` column.
3. Lowercase the text.
4. Strip URLs, quote marks, and non-word symbols — while preserving Devanagari
   (Hindi) and Arabic/Urdu Unicode ranges so multilingual text is not corrupted.
5. Encode `sentiment` labels to integers (`negative=0, neutral=1, positive=2`).
6. Fit a Keras `Tokenizer` (vocab size 5000, OOV token) on the cleaned text.
7. Convert text to integer sequences.
8. Pad/truncate all sequences to a fixed length (40 tokens, post-padding).
9. Save the tokenizer, max length, and label mapping to `model/tokenizer.pkl`.

## Train/Test Split

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
71 records → **56 training / 15 testing**, stratified across the three sentiment classes.

## LSTM Architecture (`src/train_lstm.py`)

```text
Input Text (headline + short_text)
        ↓
Embedding Layer (vocab=5000, dim=128)
        ↓
LSTM Layer (64 units)
        ↓
Dropout (0.5)
        ↓
Dense (32, relu)
        ↓
Dense (3, softmax)
        ↓
Positive / Negative / Neutral
```

```python
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),
    LSTM(64),
    Dropout(0.5),
    Dense(32, activation="relu"),
    Dense(3, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
```

Trained for 15 epochs, batch size 8, with a 20% validation split from the training data.

## Testing & Evaluation — actual results

These numbers came from a real run of `src/train_lstm.py` on this dataset
(not fabricated — see `model/training_results.json` for the raw output).

| Metric | Value |
|---|---|
| Test Accuracy | **0.40** (40%) |
| Precision (weighted) | 0.16 |
| Recall (weighted) | 0.40 |
| F1-score (weighted) | 0.23 |

```text
              precision    recall  f1-score   support

    negative       0.00      0.00      0.00         5
     neutral       0.40      1.00      0.57         6
    positive       0.00      0.00      0.00         4

    accuracy                           0.40        15
   macro avg       0.13      0.33      0.19        15
weighted avg       0.16      0.40      0.23        15
```

Confusion matrix (rows = true label, columns = predicted label; order `negative, neutral, positive`):

```text
[[0 5 0]
 [0 6 0]
 [0 4 0]]
```

**Honest limitation:** with only 71 rows (56 for training), the model
collapsed to predicting the majority class (`neutral`) for every test example.
This is a real, expected outcome of an LSTM trained from scratch on a dataset
this small — it does not have enough examples per class to learn generalizable
patterns. To improve this, in order of likely impact:
- Use pretrained word embeddings (e.g. multilingual FastText/GloVe) instead of
  training embeddings from random init.
- Collect more labeled rows (the full Global Trending Topics 2026 dataset,
  not just this 71-row slice).
- Try a simpler baseline (TF-IDF + logistic regression) as a sanity check on
  such a small sample size.
- Use class-weighting or oversampling for the minority classes.

## Saved Artifacts

- `model/lstm_sentiment_model.keras` — trained Keras model
- `model/tokenizer.pkl` — fitted tokenizer + max sequence length + label mapping
- `model/training_results.json` — real metrics from the training run above

These let `src/predict.py` and `app.py` make predictions **without retraining**.

## Prediction (`src/predict.py`)

```text
User text → Cleaning → Tokenizer → Padding → Saved LSTM model → Sentiment + Confidence
```

```bash
python src/predict.py "Technology stocks are showing strong growth today."
```

Actual output from this project's trained model:

```text
Input:
"Technology stocks are showing strong growth today."

Prediction:
Neutral

Confidence:
44.89%
```

(The confidence is low and the prediction defaults toward "neutral" — a direct
consequence of the small-dataset limitation described above, not a display bug.)

## Live Demo (`app.py` — Streamlit)

```bash
streamlit run app.py
```

- **Title:** LSTM Sentiment Analysis — Global Trending Topics 2026
- Text box: *Enter headline or social-media text*
- Button: *Predict Sentiment*
- Displays predicted sentiment (Positive / Negative / Neutral) and confidence %
- Loads the saved model + tokenizer at startup — does not retrain

The app was smoke-tested locally (`streamlit run app.py`) and confirmed to
load and serve without errors.

## Project Structure

```text
lstm-global-trending-sentiment-project/
│
├── data/
│   └── trending_topics_rows_140_210.csv
│
├── src/
│   ├── preprocess.py
│   ├── train_lstm.py
│   └── predict.py
│
├── model/
│   ├── lstm_sentiment_model.keras
│   ├── tokenizer.pkl
│   └── training_results.json
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Install Dependencies

```bash
pip install -r requirements.txt
```

## How to Train

```bash
cd src
python train_lstm.py
```
This regenerates `model/lstm_sentiment_model.keras`, `model/tokenizer.pkl`,
and `model/training_results.json` from scratch.

## How to Run the Live Demo

```bash
streamlit run app.py
```
Requires `model/lstm_sentiment_model.keras` and `model/tokenizer.pkl` to
already exist (run training first if they don't).
