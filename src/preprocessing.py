"""
Data loading and preprocessing utilities for the churn prediction project.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']


def load_data(path: str) -> pd.DataFrame:
    """Load the raw Telco churn CSV and do basic cleaning."""
    df = pd.read_csv(path)

    # TotalCharges is stored as object due to blank strings for new customers
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Drop identifier column — not predictive
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    return df


def split_features_target(df: pd.DataFrame, target_col: str = 'Churn'):
    """Split into X, y and encode target as 0/1."""
    X = df.drop(target_col, axis=1)
    y = df[target_col].map({'Yes': 1, 'No': 0})
    return X, y


def build_preprocessor(X: pd.DataFrame, numeric_features=None) -> ColumnTransformer:
    """
    Build a ColumnTransformer that scales numeric features and
    one-hot encodes everything else. Fit only on training data
    to avoid leakage.
    """
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES

    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, numeric_features=None) -> list:
    """Recover human-readable feature names after ColumnTransformer encoding."""
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES

    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(
        preprocessor.transformers_[1][2]
    )
    return list(numeric_features) + list(cat_features)
