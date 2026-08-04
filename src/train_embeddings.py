"""
Upgrade model: multilingual sentence embeddings + Logistic Regression.

Instead of TF-IDF (word counts), we use a pretrained transformer to convert
each sentence into a meaning-aware vector (embedding), then train a simple
classifier(logistic regression) on top of those embeddings. CPU-only, no fine-tuning needed.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sentence_transformers import SentenceTransformer
import joblib
import os

DATA_PATH = "data/roman_urdu_cleaned.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# multilingual model, small enough to run fast on CPU, handles Roman-Urdu
# reasonably since it saw transliterated/mixed text during pretraining

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def main():
    df = pd.read_csv(DATA_PATH)
    X = df["clean_text"]
    y = df["label"]

    # train test split
    X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train size: {len(X_train)} and Test Size: {len(X_test)}")

    # convert to embeddings
    print(f"Downloading pretrained model: {EMBEDDING_MODEL_NAME}")
    print("First run download 470 mb, then cached locally")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Encoding training sentences into embedding, this will take some time on cpu")
    X_train_emb = embedder.encode(
        X_train.tolist(), show_progress_bar=True, batch_size=32
    )

    print("Encoding testing sentences...")
    X_test_emb = embedder.encode(
        X_test.tolist(), show_progress_bar=True, batch_size=32
    )

    print("\nEmbedding Shape:", X_train_emb.shape)


    ##  training classifier (logistic regression) on top of embeddings 
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_emb, y_train)
    preds = clf.predict(X_test_emb)

    f1 = f1_score(y_test, preds, average="macro")
    print("\n================= Embeddings + Logistic regression ====================")
    print(classification_report(y_test, preds))
    print(f"Macro f1: {f1: .4f}")

    # save everything needed for inference later
    joblib.dump(clf, f"{MODEL_DIR}/embedding_classifier.joblib")
    # the embedder itself will be downloaded from hugging face, no need to save it

if __name__ == "__main__":
    main()