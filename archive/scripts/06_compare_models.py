"""
06_compare_models.py
Trains Random Forest, SVM, and Logistic Regression side-by-side.
Outputs a comparison bar chart and summary table to reports/.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

FEATURES_CSV = os.path.join("..", "features.csv")
MODELS_DIR   = os.path.join("..", "models")
REPORTS_DIR  = os.path.join("..", "reports")
NON_FEATURE_COLS = {"filename", "urgency_label", "language", "transcript"}


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].fillna(0).values
    y = df["urgency_label"].values
    return X, y, feature_cols


def make_pipeline(clf):
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=42)),
        ("clf",    clf),
    ])


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    X, y, _ = load_data()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "SVM (RBF)": SVC(
            kernel="rbf", C=10, gamma="scale",
            class_weight="balanced", random_state=42),
        "Logistic Regression": LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=1000, random_state=42),
    }

    results = {}
    for name, clf in models.items():
        print(f"  Evaluating {name} ...")
        pipe = make_pipeline(clf)
        f1_scores  = cross_val_score(pipe, X, y_enc, cv=cv, scoring="f1_macro")
        acc_scores = cross_val_score(pipe, X, y_enc, cv=cv, scoring="accuracy")
        results[name] = {
            "Macro F1 (mean)": round(f1_scores.mean(), 3),
            "Macro F1 (std)":  round(f1_scores.std(),  3),
            "Accuracy (mean)": round(acc_scores.mean(), 3),
        }
        print(f"    F1: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

    # ── Save best model (SVM often wins on small datasets) ──────────────────
    best_name = max(results, key=lambda k: results[k]["Macro F1 (mean)"])
    print(f"\nBest model: {best_name}")
    best_pipe = make_pipeline(models[best_name])
    best_pipe.fit(X, y_enc)
    joblib.dump(best_pipe, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(le,        os.path.join(MODELS_DIR, "label_encoder.pkl"))

    # ── Comparison table ─────────────────────────────────────────────────────
    df_results = pd.DataFrame(results).T
    print(f"\n{df_results}")
    df_results.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"))

    # ── Bar chart ─────────────────────────────────────────────────────────────
    names  = list(results.keys())
    f1s    = [results[n]["Macro F1 (mean)"] for n in names]
    accs   = [results[n]["Accuracy (mean)"] for n in names]
    errors = [results[n]["Macro F1 (std)"]  for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, f1s,  width, label="Macro F1",  color="#3498db", yerr=errors, capsize=5)
    bars2 = ax.bar(x + width/2, accs, width, label="Accuracy", color="#2ecc71")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — 5-Fold Cross Validation")
    ax.legend()
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Baseline (random)")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved: reports/model_comparison.png")


if __name__ == "__main__":
    main()
