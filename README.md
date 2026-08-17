# Customer Churn Prediction with Scikit-Learn

An end-to-end machine learning pipeline predicting customer churn on the Telco Customer Churn dataset — covering preprocessing, class imbalance handling, model comparison, hyperparameter tuning, and interpretability.

## Overview

Customer churn directly impacts revenue metrics like CLV and ARPU. This project builds a production-style pipeline (not just a model) to predict which customers are likely to churn, and surfaces the key drivers behind those predictions.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 features (demographics, account info, services subscribed), binary churn label. ~27% positive class (churned).

Download the CSV and place it at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Approach

1. **EDA** — distribution analysis, class imbalance check, feature relationships with churn
2. **Preprocessing** — `ColumnTransformer` pipeline (scaling numeric features, one-hot encoding categoricals) to prevent data leakage
3. **Modeling** — Logistic Regression (baseline) → Random Forest → HistGradientBoosting, compared via `StratifiedKFold` cross-validation
4. **Imbalance handling** — `class_weight='balanced'`
5. **Tuning** — `GridSearchCV` across the full pipeline (preprocessing + model)
6. **Evaluation** — ROC-AUC, precision-recall curve, confusion matrix — evaluated against business cost of false negatives (missed churners)
7. **Interpretability** — permutation feature importance to identify top churn drivers

## Results

| Model | ROC-AUC |
|---|---|
| Logistic Regression | _fill in_ |
| Random Forest | _fill in_ |
| HistGradientBoosting | _fill in_ |

**Top churn drivers:** _fill in after running permutation importance (e.g. contract type, tenure, monthly charges)_

## Project Structure

```
customer-churn-prediction-sklearn/
├── data/                    # raw dataset (place CSV here)
├── notebooks/
│   └── churn_analysis.ipynb # main EDA + modeling notebook
├── src/
│   ├── preprocessing.py     # data loading + preprocessing pipeline
│   ├── train.py              # model training + tuning (CLI script)
│   └── evaluate.py           # evaluation + feature importance (CLI script)
├── models/                   # saved model + evaluation plots (generated)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/customer-churn-prediction-sklearn.git
cd customer-churn-prediction-sklearn
pip install -r requirements.txt
```

## Usage

**Option A — Notebook (recommended for exploration / Colab):**
Open `notebooks/churn_analysis.ipynb` and run top to bottom.

**Option B — CLI scripts (for a reproducible pipeline):**
```bash
cd src
python train.py --data ../data/WA_Fn-UseC_-Telco-Customer-Churn.csv --out ../models/best_model.pkl
python evaluate.py --model ../models/best_model.pkl
```

`train.py` compares Logistic Regression, Random Forest, and HistGradientBoosting via cross-validation, tunes the best one with `GridSearchCV`, and saves the fitted pipeline. `evaluate.py` loads that model, prints a classification report, and saves confusion matrix / precision-recall / feature importance plots to `models/`.

## Tech Stack

Python · scikit-learn · pandas · seaborn/matplotlib

## Future Work

- Deploy as a Streamlit app for interactive churn scoring
- Compare against a neural network baseline (TensorFlow)
- SHAP values for per-customer explanations
