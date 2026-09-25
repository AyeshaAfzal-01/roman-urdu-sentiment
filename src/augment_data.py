"""
Builds an augmented training set to fix the spelling-variant gap found
via canary testing (e.g. "mere supervisor bohat bure hea" -> misclassified).

IMPORTANT: augmentation is applied ONLY to the train split. The test split
is saved untouched, using the exact same random_state=42 split as every
other script in this project -- so results stay comparable, and we're not
"cheating" by letting the model train on anything resembling test data.

Run: python3 src/augment_data.py
Outputs:
  data/train_augmented.csv  (original train + generated variants + targeted templates)
  data/test_holdout.csv     (unmodified test set, for evaluation)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from spelling_variants import generate_variants

DATA_PATH = "data/roman_urdu_cleaned.csv"

# targeted templates: directly covers the exact gap we diagnosed
# (person + adjective + copula-variant, workplace/relationship context)
PEOPLE = ["supervisor", "boss", "manager", "colleague", "teacher", "dost", "bhai", "behn"]
NEGATIVE_ADJ = ["bura", "bure", "buri", "kharab", "bekar"]
POSITIVE_ADJ = ["acha", "achha", "zabardast", "best"]
COPULA_VARIANTS = ["hai", "hy", "h", "hea", "hae"]

TEMPLATES = [
    "mera {person} bohat {adj} {copula}",
    "mere {person} bohat {adj} {copula}",
    "yeh {person} bohat {adj} {copula}",
]


def generate_templated_examples():
    rows = []
    for person in PEOPLE:
        for copula in COPULA_VARIANTS:
            for adj in NEGATIVE_ADJ:
                for template in TEMPLATES:
                    text = template.format(person=person, adj=adj, copula=copula)
                    rows.append({"clean_text": text, "label": "Negative"})
            for adj in POSITIVE_ADJ:
                for template in TEMPLATES:
                    text = template.format(person=person, adj=adj, copula=copula)
                    rows.append({"clean_text": text, "label": "Positive"})
    return pd.DataFrame(rows).drop_duplicates(subset=["clean_text"])


def main():
    df = pd.read_csv(DATA_PATH)
    X = df["clean_text"]
    y = df["label"]

    # SAME split as every other script -- test set stays identical and untouched
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = pd.DataFrame({"clean_text": X_train, "label": y_train}).reset_index(drop=True)
    test_df = pd.DataFrame({"clean_text": X_test, "label": y_test}).reset_index(drop=True)

    print(f"Original train size: {len(train_df)}")

    # 1. substitution augmentation on existing train sentences
    augmented_rows = []
    for _, row in train_df.iterrows():
        variants = generate_variants(row["clean_text"], n=2, seed=42)
        for v in variants:
            augmented_rows.append({"clean_text": v, "label": row["label"]})
    substitution_df = pd.DataFrame(augmented_rows).drop_duplicates(subset=["clean_text"])
    print(f"Substitution-augmented examples generated: {len(substitution_df)}")

    # 2. targeted templates covering the exact diagnosed gap
    templated_df = generate_templated_examples()
    print(f"Targeted template examples generated: {len(templated_df)}")

    # combine, drop any accidental overlap with the test set (safety check)
    combined = pd.concat([train_df, substitution_df, templated_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["clean_text"])
    leaked = combined["clean_text"].isin(test_df["clean_text"])
    if leaked.any():
        print(f"Removing {leaked.sum()} rows that overlapped with test set")
        combined = combined[~leaked]

    print(f"\nFinal augmented train size: {len(combined)} (was {len(train_df)})")
    print(combined["label"].value_counts())

    combined.to_csv("data/train_augmented.csv", index=False)
    test_df.to_csv("data/test_holdout.csv", index=False)
    print("\nSaved data/train_augmented.csv and data/test_holdout.csv")


if __name__ == "__main__":
    main()