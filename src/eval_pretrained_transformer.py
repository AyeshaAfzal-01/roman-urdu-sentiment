"""
Evaluate a pretrained, already-fine-tuned Roman-Urdu sentiment model
(Khubaib01/roman-urdu-sentiment-xlm-r) on OUR test set, for a fair
comparison against our TF-IDF baseline. No training happens here --
we're just checking whether someone else's fine-tuned model
generalizes well to our data.

Run: python3 src/eval_pretrained_transformer.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from transformers import pipeline

DATA_PATH = "data/roman_urdu_cleaned.csv"
MODEL_NAME = "Khubaib01/roman-urdu-sentiment-xlm-r"

# this model's output labels might not exactly match ours (e.g. casing,
# or extra classes like "toxic") -- we'll map/inspect and adjust as needed
LABEL_MAP = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    "POSITIVE": "Positive",
    "NEGATIVE": "Negative",
    "NEUTRAL": "Neutral",
}


def main():
    df = pd.read_csv(DATA_PATH)
    X = df["clean_text"]
    y = df["label"]

    # same split as before -- fair, apples-to-apples comparison
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Loading pretrained model: {MODEL_NAME}")
    print("(downloads once, then cached)")
    classifier = pipeline("text-classification", model=MODEL_NAME, truncation=True)

    # quick sanity check on a couple of known examples first
    print("\nSanity check:")
    for sample in ["ye banda bohot acha hai", "ye cheez bilkul bekar hai"]:
        print(f"  {sample!r} -> {classifier(sample)}")

    # run on a SMALL subset first to check label format before committing
    # to the full ~3900 test set (which takes a while on CPU)
    print("\nRunning on first 20 test examples to check label format...")
    sample_texts = X_test.tolist()[:20]
    sample_preds = classifier(sample_texts)
    for text, pred in zip(sample_texts[:5], sample_preds[:5]):
        print(f"  {text!r} -> {pred}")

    # if labels look mapped correctly, run full evaluation
    print(f"\nRunning full evaluation on {len(X_test)} test examples (this will take several minutes on CPU)...")
    all_preds_raw = classifier(X_test.tolist(), batch_size=16)
    predicted_labels = [LABEL_MAP.get(p["label"], p["label"]) for p in all_preds_raw]

    print("\n=== Pretrained Khubaib01/roman-urdu-sentiment-xlm-r ===")
    print(classification_report(y_test, predicted_labels))
    f1 = f1_score(y_test, predicted_labels, average="macro")
    print(f"Macro F1: {f1:.4f}")
    print("\nCompare against:")
    print("  TF-IDF + Logistic Regression: 0.6349")
    print("  Multilingual MiniLM embeddings + LR: 0.5272")


if __name__ == "__main__":
    main()