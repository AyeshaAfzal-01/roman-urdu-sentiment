# Roman-Urdu Sentiment Classifier

A sentiment analysis pipeline for **Roman-Urdu** (Urdu written in Latin script) - a genuinely harder NLP problem than standard English sentiment analysis, since Roman-Urdu has **no standardized spelling** (e.g. "acha", "acha", and "achaa" all mean the same thing).

**[Live demo](#)** *(https://roman-urdu-sentiment.streamlit.app/)*

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
| Pretrained multilingual embeddings (MiniLM) + LR | 0.53 | **Underperformed** - see below |
| Off-the-shelf pretrained transformer | 0.67 | Fine-tuned by a third party on Roman-Urdu |
| **Fine-tuned on our own train split** | **0.71** | Best - used in the deployed app |

### Key finding: bigger ≠ better, without domain match

A generic pretrained multilingual sentence embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) **underperformed simple TF-IDF** by 10 points of macro F1. Roman-Urdu's non-standard, transliterated spelling means this model had little relevant exposure to it during pretraining — its embeddings were closer to noise than signal for this domain.

A transformer *specifically fine-tuned on Roman-Urdu* (`Khubaib01/roman-urdu-sentiment-xlm-r`) did win, confirming the issue was domain mismatch, not the architecture itself.

### Fine-tuning on our own data

Starting from that off-the-shelf checkpoint, I continued fine-tuning it (3 epochs, on a free Colab T4 GPU) directly on our own train split. This raised macro F1 from 0.67 to **0.71**, with per-class precision/recall becoming noticeably more balanced (previously as lopsided as 0.80 precision / 0.52 recall on Neutral; afterward, every class sits in a tight 0.68–0.75 band on both metrics). This is the model used in the deployed app. See `notebooks/finetune_colab.ipynb`.

### Error analysis

Manual inspection of misclassified examples showed that a meaningful share of "errors" reflect genuine **label ambiguity** in the source data (sarcasm, missing conversational context, borderline neutral/positive tone) rather than model failure — most confusion occurs at the Neutral boundary, not between clearly Positive and Negative text. See `src/error_analysis.py`.

## Project structure

```
├── data/                       # dataset (raw + cleaned; gitignored)
├── src/
│   ├── clean_data.py           # text cleaning & normalization
│   ├── train_baseline.py       # TF-IDF + Naive Bayes / Logistic Regression
│   ├── train_embeddings.py     # pretrained multilingual embeddings + LR
│   ├── eval_pretrained_transformer.py   # evaluate off-the-shelf fine-tuned transformer
│   └── error_analysis.py       # inspect misclassified examples
├── notebooks/
│   └── finetune_colab.ipynb    # fine-tuning on our own data (Colab, free T4 GPU)
├── app/
│   └── app.py                  # Streamlit demo (uses our fine-tuned model)
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

- Expand the dataset with more recent social media text
- Explore a binary (Positive/Negative) variant, since most confusion concentrates around the Neutral class
- Experiment with more epochs / hyperparameter tuning during fine-tuning