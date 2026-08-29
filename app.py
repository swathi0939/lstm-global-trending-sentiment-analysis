import os
import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.preprocess import clean_text


# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_sentiment_model.keras"
)

TOKENIZER_PATH = os.path.join(
    MODEL_DIR,
    "tokenizer.pkl"
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="LSTM Sentiment Analysis",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(
                circle at top left,
                #1e3a8a 0%,
                #0f172a 35%,
                #020617 100%
            );
        color: white;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        margin-top: 15px;
        margin-bottom: 5px;

        background:
            linear-gradient(
                90deg,
                #60a5fa,
                #a78bfa,
                #22d3ee
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #cbd5e1;
        margin-bottom: 32px;
    }

    /* Project summary cards */
    .info-card {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        box-shadow: 0px 8px 30px rgba(0, 0, 0, 0.20);
        min-height: 120px;
    }

    .info-number {
        font-size: 30px;
        font-weight: bold;
        color: #60a5fa;
    }

    .info-label {
        color: #cbd5e1;
        font-size: 14px;
        margin-top: 5px;
    }

    /* Main prediction area */
    .prediction-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 28px;
        border-radius: 22px;
        box-shadow: 0px 12px 40px rgba(0, 0, 0, 0.28);
        margin-top: 20px;
    }

    /* Prediction result */
    .result-positive {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid #22c55e;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        margin-top: 20px;
    }

    .result-negative {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        margin-top: 20px;
    }

    .result-neutral {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #f59e0b;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        margin-top: 20px;
    }

    /* Button */
    div.stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 12px;
        border: none;

        background:
            linear-gradient(
                90deg,
                #2563eb,
                #7c3aed
            );

        color: white;
        font-size: 18px;
        font-weight: bold;
        transition: 0.3s;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);

        background:
            linear-gradient(
                90deg,
                #1d4ed8,
                #6d28d9
            );
    }

    /* Text area */
    textarea {
        border-radius: 14px !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        margin-top: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL AND TOKENIZER
# =========================================================
@st.cache_resource
def load_artifacts():

    model = load_model(
        MODEL_PATH
    )

    with open(
        TOKENIZER_PATH,
        "rb"
    ) as f:

        bundle = pickle.load(f)

    return (
        model,
        bundle["tokenizer"],
        bundle["max_length"],
        bundle["idx2label"]
    )


# =========================================================
# CHECK MODEL FILES
# =========================================================
if not (
    os.path.exists(MODEL_PATH)
    and
    os.path.exists(TOKENIZER_PATH)
):

    st.error(
        """
        Model or tokenizer not found.

        Run:

        python src/train_lstm.py

        before starting the Streamlit app.
        """
    )

    st.stop()


# =========================================================
# LOAD ARTIFACTS
# =========================================================
try:

    model, tokenizer, max_length, idx2label = (
        load_artifacts()
    )

except Exception as e:

    st.error(
        "Unable to load the trained model or tokenizer."
    )

    st.exception(e)

    st.stop()


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="main-title">
        🧠 LSTM Sentiment Analysis
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Global Trending Topics 2026
        • News + Social Media
        • Multilingual Sentiment Classification
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# INFORMATION CARDS
# =========================================================
col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        """
        <div class="info-card">

        <div class="info-number">
        71
        </div>

        <div class="info-label">
        Records Used
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="info-card">

        <div class="info-number">
        140–210
        </div>

        <div class="info-label">
        Dataset Rows
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="info-card">

        <div class="info-number">
        LSTM
        </div>

        <div class="info-label">
        Deep Learning Model
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        """
        <div class="info-card">

        <div class="info-number">
        3
        </div>

        <div class="info-label">
        Sentiment Classes
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MAIN SECTION
# =========================================================
st.write("")

left, center, right = st.columns(
    [1, 3, 1]
)

with center:

    st.markdown(
        """
        <div class="prediction-card">
        """,
        unsafe_allow_html=True
    )

    st.subheader(
        "🔍 Analyze Sentiment"
    )

    st.write(
        """
        Enter a news headline or social-media sentence.
        The trained LSTM model will classify it as
        Positive, Negative or Neutral.
        """
    )

    text_input = st.text_area(
        "Enter headline or social-media text",
        height=150,
        placeholder=(
            "Example: "
            "The new technology received excellent "
            "feedback from users."
        )
    )


    # =====================================================
    # PREDICT BUTTON
    # =====================================================
    if st.button(
        "🚀 Predict Sentiment"
    ):

        if not text_input.strip():

            st.warning(
                "⚠️ Please enter some text first."
            )

        else:

            # ---------------------------------------------
            # CLEAN TEXT
            # ---------------------------------------------
            cleaned = clean_text(
                text_input
            )


            # ---------------------------------------------
            # TOKENIZATION
            # ---------------------------------------------
            seq = tokenizer.texts_to_sequences(
                [cleaned]
            )


            # ---------------------------------------------
            # PADDING
            # ---------------------------------------------
            padded = pad_sequences(
                seq,
                maxlen=max_length,
                padding="post",
                truncating="post"
            )


            # ---------------------------------------------
            # MODEL PREDICTION
            # ---------------------------------------------
            probs = model.predict(
                padded,
                verbose=0
            )[0]


            pred_idx = int(
                np.argmax(probs)
            )


            label = idx2label[
                pred_idx
            ]


            confidence = (
                float(
                    probs[pred_idx]
                )
                * 100
            )


            # =============================================
            # DISPLAY SENTIMENT RESULT
            # =============================================
            if label.lower() == "positive":

                st.markdown(
                    f"""
                    <div class="result-positive">

                    😊 Positive Sentiment

                    <br><br>

                    Confidence:
                    {confidence:.2f}%

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            elif label.lower() == "negative":

                st.markdown(
                    f"""
                    <div class="result-negative">

                    😞 Negative Sentiment

                    <br><br>

                    Confidence:
                    {confidence:.2f}%

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            else:

                st.markdown(
                    f"""
                    <div class="result-neutral">

                    😐 Neutral Sentiment

                    <br><br>

                    Confidence:
                    {confidence:.2f}%

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # =============================================
            # CONFIDENCE BAR
            # =============================================
            st.write("")

            st.write(
                "### Prediction Confidence"
            )

            st.progress(
                min(
                    int(confidence),
                    100
                )
            )


            # =============================================
            # CLASS PROBABILITIES
            # =============================================
            st.write("")

            st.write(
                "### 📊 Class Probabilities"
            )


            probability_columns = (
                st.columns(
                    len(idx2label)
                )
            )


            for (
                column,
                (idx, lbl)
            ) in zip(
                probability_columns,
                idx2label.items()
            ):

                probability = (
                    float(
                        probs[idx]
                    )
                    * 100
                )

                with column:

                    if (
                        lbl.lower()
                        == "positive"
                    ):

                        icon = "😊"

                    elif (
                        lbl.lower()
                        == "negative"
                    ):

                        icon = "😞"

                    else:

                        icon = "😐"


                    st.metric(
                        f"{icon} {lbl.capitalize()}",
                        f"{probability:.2f}%"
                    )


            # =============================================
            # PREPROCESSED TEXT
            # =============================================
            with st.expander(
                "🧹 View Preprocessed Text"
            ):

                st.write(
                    cleaned
                )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# =========================================================
# ARCHITECTURE SECTION
# =========================================================
st.write("")
st.write("")

st.markdown("---")

st.subheader(
    "⚙️ How the Model Works"
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.info(
        """
        ### 📰 Input

        headline

        +

        short_text
        """
    )


with c2:

    st.info(
        """
        ### 🧹 Preprocessing

        Cleaning

        Tokenization

        Padding
        """
    )


with c3:

    st.info(
        """
        ### 🧠 LSTM

        Embedding

        LSTM

        Dense
        """
    )


with c4:

    st.info(
        """
        ### 🎯 Output

        Positive

        Negative

        Neutral
        """
    )


# =========================================================
# DATASET INFORMATION
# =========================================================
st.write("")

st.subheader(
    "📁 Dataset Information"
)

st.write(
    """
    This project uses only **rows 140–210**
    from the **Global Trending Topics 2026**
    dataset.

    The LSTM uses the combination of:

    **headline + short_text**

    as input and predicts:

    **sentiment**
    """
)


# =========================================================
# MODEL LIMITATION
# =========================================================
st.warning(
    """
    ⚠️ Model Limitation:

    The model was trained using only 71 records.
    Because the dataset subset is very small,
    predictions may not always match the expected
    sentiment.
    """
)


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.markdown(
    """
    <div class="footer">

    LSTM Sentiment Analysis
    • Global Trending Topics 2026

    <br>

    Built using TensorFlow,
    Keras and Streamlit

    </div>
    """,
    unsafe_allow_html=True
)