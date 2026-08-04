# Baseline sentiment classifiers: TF-IDF + Naive Bayes vs TF-IDF + Logistic Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import joblib
import os

DATA_PATH = "data/roman_urdu_cleaned.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)
    X = df['clean_text']
    y = df['label']

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Train length = {len(X_train)} and Test length = {len(X_test)}")

    # vectorizer
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results = {}

    # MODEL 1: naive baye's
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    nb_predictions = nb.predict(X_test_vec)
    nb_f1 = f1_score(y_test, nb_predictions, average="macro")
    results["naive bayes"] = nb_f1
    print("============ Naive Bayes ================")
    print(classification_report(y_test, nb_predictions))

    # MODEL 2: logistic regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train_vec, y_train)
    lr_predictions = lr.predict(X_test_vec)
    lr_f1 = f1_score(y_test, lr_predictions, average="macro")
    results["logistic regression"] = lr_f1
    print("================== Logistic Regression =============")
    print(classification_report(y_test, lr_predictions))

    # summary (macro f1)
    for name, score in results.items():
        print(f"{name} = {score: .4f}")


    # save the better model and vectorizer for reuse
    bestModel = lr if lr_f1 >= nb_f1 else nb
    joblib.dump(bestModel, f"{MODEL_DIR}/baseline_model.joblib")
    joblib.dump(vectorizer, f"{MODEL_DIR}/tfidf_vectorizer.joblib")


if __name__ == "__main__":
    main()