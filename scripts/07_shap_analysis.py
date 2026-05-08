"""
07_shap_analysis.py
Generates SHAP explainability plots for the trained Random Forest.
Shows which features drive each urgency class.
Outputs to reports/.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

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


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    pipeline     = joblib.load(os.path.join(MODELS_DIR, "rf_urgency.pkl"))
    le           = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))

    X, y, _ = load_data()

    # Transform X through the scaler only (not SMOTE — that's training-only)
    scaler = pipeline.named_steps["scaler"]
    X_scaled = scaler.transform(X)
    rf = pipeline.named_steps["clf"]

    print("Computing SHAP values (this may take ~1 min) ...")
    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_scaled)

    class_names = le.classes_   # ['High', 'Low', 'Medium']

    # ── 1. Summary beeswarm — all classes combined ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_scaled,
        feature_names=feature_cols,
        class_names=class_names,
        show=False, plot_size=None
    )
    plt.title("SHAP Feature Impact — All Urgency Classes", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: reports/shap_summary.png")

    # ── 2. Per-class top-10 bar charts ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"High": "#e74c3c", "Low": "#2ecc71", "Medium": "#f39c12"}

    for idx, cls in enumerate(class_names):
        mean_abs = np.abs(shap_values[idx]).mean(axis=0)
        top10_idx = np.argsort(mean_abs)[::-1][:10]
        top10_names  = [feature_cols[i] for i in top10_idx]
        top10_values = mean_abs[top10_idx]

        axes[idx].barh(top10_names[::-1], top10_values[::-1],
                       color=colors.get(cls, "steelblue"))
        axes[idx].set_title(f"{cls} Urgency", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Mean |SHAP value|")

    plt.suptitle("Top 10 Features per Urgency Class (SHAP)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "shap_per_class.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: reports/shap_per_class.png")

    # ── 3. Print top 5 features per class to console ─────────────────────────
    print("\nTop 5 features per class:")
    for idx, cls in enumerate(class_names):
        mean_abs = np.abs(shap_values[idx]).mean(axis=0)
        top5 = np.argsort(mean_abs)[::-1][:5]
        print(f"  {cls}: {[feature_cols[i] for i in top5]}")


if __name__ == "__main__":
    main()
