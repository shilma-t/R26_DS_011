"""
03_train.py
Trains a Random Forest classifier on the extracted features.
Applies SMOTE for class imbalance, tunes with cross-validation,
and saves the trained model + label encoder to models/.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

FEATURES_CSV = os.path.join("..", "features.csv")
MODELS_DIR   = os.path.join("..", "models")

NON_FEATURE_COLS = {"filename", "urgency_label", "language", "transcript"}


def load_data(path: str):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].fillna(0).values
    y = df["urgency_label"].values
    return X, y, feature_cols


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading features ...")
    X, y, feature_cols = load_data(FEATURES_CSV)
    print(f"  {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=42)),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y_enc, cv=cv, scoring="f1_macro")
    print(f"\n5-Fold CV macro F1: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"  Per-fold: {[round(s,3) for s in scores]}")

    print("\nFitting on full dataset ...")
    pipeline.fit(X, y_enc)

    joblib.dump(pipeline,     os.path.join(MODELS_DIR, "rf_urgency.pkl"))
    joblib.dump(le,           os.path.join(MODELS_DIR, "label_encoder.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))

    print(f"Model saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
