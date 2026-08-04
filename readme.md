# Roman-Urdu Sentiment Classifier

A sentiment analysis pipeline for **Roman-Urdu** (Urdu written in Latin script) — a genuinely harder NLP problem than standard English sentiment analysis, since Roman-Urdu has **no standardized spelling** (e.g. "acha", "acha", and "achaa" all mean the same thing).

**[Live demo](#)** *(add your deployed Streamlit link here)*

---

## Problem

Classify Roman-Urdu text (tweets, reviews, comments) as **Positive**, **Negative**, or **Neutral**. Roman-Urdu is widely used across Pakistan on social media and e-commerce platforms, but is underserved by NLP tooling compared to English or even standard-script Urdu.

## Dataset

~20,000 manually-labeled sentences from the [Roman-Urdu Dataset](https://github.com/Smat26/Roman-Urdu-Dataset) (Twitter, Facebook, and e-commerce review text). After cleaning (deduplication, fixing a mislabeled class, normalizing text), **19,586 sentences** remain, moderately imbalanced across 3 classes (Neutral being the largest).

## Approach & Results

Rather than jumping to the fanciest available technique, this project **compares four approaches** and evaluates each honestly on the same held-out test set (macro F1, chosen over accuracy specifically because of class imbalance):

| Model | Macro F1 | Notes |
|---|---|---|
| TF-IDF + Naive Bayes | 0.60 | Baseline |
| TF-IDF + Logistic Regression | 0.63 | Balanced class weights |
| Pretrained multilingual embeddings (MiniLM) + LR | 0.53 | **Underperformed** — see below |
| **Pretrained transformer fine-tuned on Roman-Urdu** | **0.67** | Best — used in the deployed app |

### Key finding: bigger ≠ better, without domain match

A generic pretrained multilingual sentence embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) **underperformed simple TF-IDF** by 10 points of macro F1. Roman-Urdu's non-standard, transliterated spelling means this model had little relevant exposure to it during pretraining — its embeddings were closer to noise than signal for this domain.

A transformer *specifically fine-tuned on Roman-Urdu* (`Khubaib01/roman-urdu-sentiment-xlm-r`) did win, confirming the issue was domain mismatch, not the architecture itself. This is the model used in the deployed app.

### Error analysis

Manual inspection of misclassified examples showed that a meaningful share of "errors" reflect genuine **label ambiguity** in the source data (sarcasm, missing conversational context, borderline neutral/positive tone) rather than model failure — most confusion occurs at the Neutral boundary, not between clearly Positive and Negative text. See `src/error_analysis.py`.

## Project structure

```
├── data/                       # dataset (raw + cleaned; gitignored)
├── src/
│   ├── clean_data.py           # text cleaning & normalization
│   ├── train_baseline.py       # TF-IDF + Naive Bayes / Logistic Regression
│   ├── train_embeddings.py     # pretrained multilingual embeddings + LR
│   ├── eval_pretrained_transformer.py   # evaluate fine-tuned transformer
│   └── error_analysis.py       # inspect misclassified examples
├── app/
│   └── app.py                  # Streamlit demo (uses the winning model)
└── requirements.txt
```

## Running it locally

```bash
pip install -r requirements.txt

# 1. Clean the raw dataset
python3 src/clean_data.py

# 2. Train & compare baselines
python3 src/train_baseline.py

# 3. (Optional) run error analysis
python3 src/error_analysis.py

# 4. Launch the demo app
streamlit run app/app.py
```

No GPU required — everything runs on CPU, including the final transformer model (inference only, not training).

## Tech stack

Python, pandas, scikit-learn, sentence-transformers, HuggingFace `transformers`, Streamlit

## Future work

- Fine-tune the transformer directly on this dataset (rather than using an off-the-shelf fine-tuned model) — would likely close more of the remaining gap, given a GPU
- Expand the dataset with more recent social media text
- Explore a binary (Positive/Negative) variant, since most confusion concentrates around the Neutral class