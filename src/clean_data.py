"""
Cleans the raw Roman Urdu sentiment dataset.

Steps:
1. Load raw CSV, drop the junk column
2. Fix mislabeled/typo classes
3. Drop missing text and duplicate rows
4. Basic text normalization (lowercase, strip punctuation/links/extra spaces)
5. Save cleaned dataset to data/roman_urdu_clean.csv
"""
import pandas as pd
import re

def clean_text(text: str) -> str:
    text = str(text).lower() # convert the text to lower case
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # replace all urls with whitespace, bcz url don't carry any kind of sentiments
    text = re.sub(r"[^\w\s]", " ", text) # replace everyting except characters and whitespace with whitespace like get rid of punctuation marks
    text = re.sub(r"\d+", " ", text)   # replace all numbers with whitespace
    text = re.sub(r"\s+", " ", text).strip()   # replace all spacing with one consistent white space and get rid of whitespace at the biggening and at the ending of sentence
    return text


def load_and_clean(raw_path: str) -> pd.DataFrame: 
    df = pd.read_csv(raw_path, header=None, names=["text", "label", "junk"])
    df = df.drop(columns=["junk"])  # droppping the third junk column containing NaN
    df["label"] = df["label"].replace({"Neative": "Negative"})
    df = df.dropna(subset=['text'])   # dropping row having no column text value
    df = df.drop_duplicates(subset=['text', 'label'])   # removing duplicate rows. a row is a combination of text and label
    df['clean_text'] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_and_clean("data/roman_urdu_raw.csv")
    print("Cleaned shape:", df.shape)
    print(df["label"].value_counts())
    df.to_csv("data/roman_urdu_cleaned.csv", index=False)
    print("data cleaned and saved as data/roman_urdu_cleaned.csv")
