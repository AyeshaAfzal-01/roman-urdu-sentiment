"""
Streamlit demo: Roman-Urdu sentiment classifier.
Uses a pretrained, already-fine-tuned transformer (Khubaib01/roman-urdu-sentiment-xlm-r).
"""

import streamlit as st
from transformers import pipeline
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "finetuned_model")


@st.cache_resource
def load_model():
    return pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH, truncation=True)

def main():
    st.set_page_config(
        page_title="Roman-Urdu Sentiment",
        page_icon="💬",
        layout="centered",
    )

    st.markdown("""
    <style>
        .block-container{
            max-width:750px;
            padding-top:2rem;
            padding-bottom:3rem;
        }

        .title{
            font-size:2rem;
            font-weight:700;
            margin-bottom:0.2rem;
        }

        .subtitle{
            color:#777;
            margin-bottom:2rem;
        }

        .result-card{
            border:1px solid #E6E6E6;
            border-radius:14px;
            padding:20px;
            margin-top:20px;
        }

        .score{
            color:#666;
            font-size:15px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='title'>Roman-Urdu Sentiment</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Analyze the sentiment of Roman-Urdu text using a fine-tuned transformer.</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading model..."):
        classifier = load_model()

    user_input = st.text_area(
        "Input text",
        placeholder="Yeh movie bohat achi thi...",
        height=130,
        label_visibility="collapsed",
    )

    if st.button("Analyze", use_container_width=True):

        if not user_input.strip():
            st.warning("Please enter some text.")
            return

        with st.spinner("Analyzing..."):
            result = classifier(user_input)[0]

        prediction = result["label"]
        confidence = result["score"]

        colors = {
            "Positive": "#2E8B57",
            "Negative": "#D32F2F",
            "Neutral": "#F9A825"
        }

        emojis = {
            "Positive": "😊",
            "Negative": "😞",
            "Neutral": "😐"
        }

        st.markdown(
            f"""
    <div class="result-card">
        <h3 style="margin-bottom:5px;color:{colors.get(prediction)};">
            {emojis.get(prediction)} {prediction}
        </h3>

        <div class="score">
            Confidence: <b>{confidence*100:.1f}%</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
        )

    st.divider()

    with st.expander("About"):
        st.write(
            """
            This model is based on **XLM-RoBERTa** and is fine-tuned for
            Roman-Urdu sentiment classification using approximately **19.5k**
            manually labeled sentences collected from reviews, comments, and
            social media posts.
            """
        )

if __name__ == "__main__":
    main()