"""
Error analysis: look at specific sentences the baseline model got wrong.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

DATA_PATH = "data/roman_urdu_cleaned.csv"
MODEL_DIR = "models"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(f"{MODEL_DIR}/baseline_model.joblib")
    vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer.joblib")

    X_test_vec = vectorizer.transform(X_test)
    preds = model.predict(X_test_vec)

    results = pd.DataFrame({
        "text": X_test.values,
        "true_label": y_test.values,
        "predicted": preds,
    })

    wrong = results[results["true_label"] != results["predicted"]]
    print(f"Total test examples: {len(results)}")
    print(f"Wrong predictions: {len(wrong)} ({len(wrong)/len(results)*100:.1f}%)")

    print("\nMost common mistakes (true -> predicted):")
    print(wrong.groupby(["true_label", "predicted"]).size().sort_values(ascending=False).head(10))

    print("\n--- Sample misclassified sentences ---")
    for (true_l, pred_l), group in wrong.groupby(["true_label", "predicted"]):
        if len(group) < 20:
            continue
        print(f"\nTrue={true_l}, Predicted={pred_l} (showing 5 of {len(group)}):")
        for text in group["text"].head(5):
            print(f"  - {text}")

    wrong.to_csv("data/misclassified_examples.csv", index=False)
    print("\nSaved all misclassified examples to data/misclassified_examples.csv")


if __name__ == "__main__":
    main()