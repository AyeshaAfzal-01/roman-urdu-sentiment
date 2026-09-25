"""
Canary tests: hand-picked sentences with obvious, unambiguous sentiment,
used to probe whether a failure is a one-off or a real pattern.

Run: python3 src/canary_test.py
"""

from transformers import pipeline
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "finetuned_model")

# (text, expected_label) -- these are all unambiguous to a human reader
CANARIES = [
    ("mere supervisor bohat bure hea", "Negative"),
    ("mera supervisor bohat bura hai", "Negative"),   # same meaning, standard spelling
    ("mera boss bohat acha hai", "Positive"),
    ("mera dost bohat acha hai", "Positive"),
    ("yeh cheez bohat buri hai", "Negative"),
    ("mujhe yeh pasand nahi aya", "Negative"),         # negation pattern
    ("mujhe yeh bilkul pasand nahi", "Negative"),
    ("teacher bohat acha parhata hai", "Positive"),
    ("teacher bohat bura parhata hai", "Negative"),
    ("office ka mahol bohat kharab hai", "Negative"),
]


def main():
    classifier = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH, truncation=True)

    correct = 0
    print(f"{'TEXT':<45} {'EXPECTED':<10} {'PREDICTED':<10} {'CONF':<6}")
    print("-" * 75)
    for text, expected in CANARIES:
        result = classifier(text)[0]
        pred = result["label"]
        conf = result["score"]
        ok = "OK" if pred == expected else "WRONG"
        correct += (pred == expected)
        print(f"{text:<45} {expected:<10} {pred:<10} {conf*100:.0f}%   {ok}")

    print(f"\n{correct}/{len(CANARIES)} correct")


if __name__ == "__main__":
    main()