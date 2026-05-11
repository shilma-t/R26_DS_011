"""
04_evaluate.py
Generates evaluation plots: confusion matrix, per-class F1, feature importance.
Outputs saved to reports/.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

FEATURES_CSV = os.path.join("..", "features.csv")
MODELS_DIR   = os.path.join("..", "models")
REPORTS_DIR  = os.path.join("..", "reports")
NON_FEATURE_COLS = {"filename", "urgency_label", "language", "transcript"}


def load_data(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].fillna(0).values
    y = df["urgency_label"].values
    return X, y, feature_cols


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    pipeline     = joblib.load(os.path.join(MODELS_DIR, "rf_urgency.pkl"))
    le           = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))

    X, y, _ = load_data(FEATURES_CSV)
    y_enc   = le.transform(y)
    labels  = le.classes_

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipeline, X, y_enc, cv=cv)
    y_pred_labels = le.inverse_transform(y_pred)

    # Classification report
    report_dict = classification_report(y, y_pred_labels, target_names=labels, output_dict=True)
    # Print without support column
    header = f"{'':>12}  {'precision':>9}  {'recall':>6}  {'f1-score':>8}"
    print(header)
    print()
    for cls in labels:
        d = report_dict[cls]
        print(f"  {cls:>10}  {d['precision']:>9.2f}  {d['recall']:>6.2f}  {d['f1-score']:>8.2f}")
    print()
    for avg in ["accuracy", "macro avg", "weighted avg"]:
        d = report_dict[avg]
        if avg == "accuracy":
            print(f"  {'accuracy':>10}  {'':>9}  {'':>6}  {d:>8.2f}")
        else:
            print(f"  {avg:>10}  {d['precision']:>9.2f}  {d['recall']:>6.2f}  {d['f1-score']:>8.2f}")
    report = classification_report(y, y_pred_labels, target_names=labels)
    with open(os.path.join(REPORTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred_labels, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title("Urgency Classification — Confusion Matrix (5-fold CV)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Feature importance — top 20
    rf = pipeline.named_steps["clf"]
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(20), importances[indices])
    ax.set_xticks(range(20))
    ax.set_xticklabels([feature_cols[i] for i in indices], rotation=45, ha="right")
    ax.set_title("Top 20 Feature Importances")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    # Per-class F1 bar chart
    f1_per_class = f1_score(y, y_pred_labels, labels=labels, average=None)
    colors = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, f1_per_class, color=[colors.get(l, "steelblue") for l in labels])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 (5-fold CV)")
    for i, v in enumerate(f1_per_class):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "per_class_f1.png"), dpi=150)
    plt.close()

    print(f"Reports saved to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
