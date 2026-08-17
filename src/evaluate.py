"""
Evaluate a trained churn model: classification report, confusion matrix,
precision-recall curve, and permutation feature importance.

Usage:
    python src/evaluate.py --model models/best_model.pkl
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
GRAPHS_DIR = ROOT / "graphs"


def main(model_path: str):
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(model_path)
    X_test, y_test = joblib.load(model_path.replace('.pkl', '_testset.pkl'))

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn']).plot()
    plt.title('Confusion Matrix')
    plt.tight_layout()
    out = GRAPHS_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")

    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.tight_layout()
    out = GRAPHS_DIR / "precision_recall_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")

    # Permutation feature importance (permutes original X columns)
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, scoring='roc_auc'
    )
    importances = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    importances.head(15).plot(kind='barh')
    plt.title('Top 15 Feature Importances (Permutation)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    out = GRAPHS_DIR / "feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")

    print("\nTop 10 features:")
    print(importances.head(10))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/best_model.pkl', help='Path to trained model .pkl')
    args = parser.parse_args()

    main(args.model)
