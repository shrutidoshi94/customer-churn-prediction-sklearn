"""
Evaluate a trained churn model: classification report, confusion matrix,
precision-recall curve, and permutation feature importance.

Usage:
    python src/evaluate.py --model models/best_model.pkl
"""

import argparse
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

from preprocessing import get_feature_names


def main(model_path: str):
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
    plt.savefig('models/confusion_matrix.png')
    print("Saved confusion_matrix.png")

    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.tight_layout()
    plt.savefig('models/precision_recall_curve.png')
    print("Saved precision_recall_curve.png")

    # Permutation feature importance
    preprocessor = model.named_steps['preprocessor']
    feature_names = get_feature_names(preprocessor)

    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, scoring='roc_auc'
    )
    importances = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    importances.head(15).plot(kind='barh')
    plt.title('Top 15 Feature Importances (Permutation)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')
    print("Saved feature_importance.png")

    print("\nTop 10 features:")
    print(importances.head(10))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/best_model.pkl', help='Path to trained model .pkl')
    args = parser.parse_args()

    main(args.model)
