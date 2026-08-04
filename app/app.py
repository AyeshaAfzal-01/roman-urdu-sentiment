"""
Streamlit demo: Roman-Urdu sentiment classifier.
Uses a pretrained, already-fine-tuned transformer (Khubaib01/roman-urdu-sentiment-xlm-r).

Run: streamlit run app/app.py
(run this command from the project root folder)
"""

import streamlit as st
from transformers import pipeline

MODEL_NAME = "Khubaib01/roman-urdu-sentiment-xlm-r"


@st.cache_resource
def load_model():
    # cached so the ~1.1GB model loads only ONCE per app session,
    # not on every button click
    return pipeline("text-classification", model=MODEL_NAME, truncation=True)


def main():
    st.set_page_config(page_title="Roman-Urdu Sentiment Classifier", page_icon="🎭")
    st.title("🎭 Roman-Urdu Sentiment Classifier")
    st.write(
        "Type a sentence in Roman-Urdu (Urdu written in English letters) "
        "and get its predicted sentiment."
    )

    with st.spinner("Loading model (first load takes longer)..."):
        classifier = load_model()

    user_input = st.text_area(
        "Enter text:",
        placeholder="e.g. yeh bohat acha tha",
        height=100,
    )

    if st.button("Predict Sentiment", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text first.")
            return

        with st.spinner("Predicting..."):
            result = classifier(user_input)[0]

        prediction = result["label"]
        confidence = result["score"]

        emoji_map = {"Positive": "\U0001F60A", "Negative": "\U0001F61E", "Neutral": "\U0001F610"}
        st.subheader(f"{emoji_map.get(prediction, '')} {prediction}")
        st.write(f"Confidence: {confidence*100:.1f}%")
        st.progress(float(confidence))

    with st.expander("About this project"):
        st.write(
            "Trained/evaluated on ~19,500 manually-labeled Roman-Urdu sentences "
            "(reviews, comments, tweets). Uses a transformer (xlm-roberta) "
            "fine-tuned specifically for Roman-Urdu, which has no standard "
            "spelling -- a genuinely harder NLP problem than standard English "
            "sentiment analysis. Achieves 0.67 macro F1 on held-out test data, "
            "outperforming a TF-IDF baseline (0.63) and a generic multilingual "
            "embedding model (0.53) that wasn't trained on Roman-Urdu specifically."
        )


if __name__ == "__main__":
    main()