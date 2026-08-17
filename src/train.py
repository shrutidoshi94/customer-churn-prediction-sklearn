"""
Train and tune churn prediction models.

Usage:
    python src/train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""

import argparse
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from preprocessing import load_data, split_features_target, build_preprocessor

RANDOM_STATE = 42


def build_pipeline(preprocessor, classifier):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])


def main(data_path: str, model_out: str):
    df = load_data(data_path)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        'logistic_regression': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE
        ),
        'random_forest': RandomForestClassifier(
            class_weight='balanced', random_state=RANDOM_STATE
        ),
        'hist_gradient_boosting': HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }

    print("Baseline model comparison (5-fold CV ROC-AUC):")
    results = {}
    for name, clf in candidates.items():
        pipe = build_pipeline(preprocessor, clf)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc')
        results[name] = scores.mean()
        print(f"  {name:25s} {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(results, key=results.get)
    print(f"\nBest baseline model: {best_name}")

    # Hyperparameter tuning on the best baseline
    pipe = build_pipeline(preprocessor, candidates[best_name])

    if best_name == 'random_forest':
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [5, 10, None],
            'classifier__min_samples_split': [2, 5, 10],
        }
    elif best_name == 'hist_gradient_boosting':
        param_grid = {
            'classifier__max_iter': [100, 200],
            'classifier__max_depth': [None, 5, 10],
            'classifier__learning_rate': [0.05, 0.1, 0.2],
        }
    else:
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10],
        }

    print("\nRunning GridSearchCV...")
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")

    best_model = grid.best_estimator_
    test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    print(f"Test ROC-AUC: {test_auc:.4f}")

    joblib.dump(best_model, model_out)
    joblib.dump((X_test, y_test), model_out.replace('.pkl', '_testset.pkl'))
    print(f"\nSaved model to {model_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to raw churn CSV')
    parser.add_argument('--out', default='models/best_model.pkl', help='Output path for trained model')
    args = parser.parse_args()

    main(args.data, args.out)
