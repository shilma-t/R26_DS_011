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

    # ── 1. Summary bar chart — mean |SHAP| per feature across all classes ────
    # shap_values shape: (n_samples, n_features, n_classes)
    shap_values = np.array(shap_values)
    mean_abs_all = np.mean([np.abs(shap_values[:, :, k]).mean(axis=0)
                            for k in range(len(class_names))], axis=0)
    top15_idx    = np.argsort(mean_abs_all)[::-1][:15]
    top15_names  = [feature_cols[i] for i in top15_idx]
    top15_vals   = mean_abs_all[top15_idx]

    # Beeswarm per class
    for k, cls in enumerate(class_names):
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values[:, :, k], X_scaled,
            feature_names=feature_cols,
            max_display=15,
            show=False, plot_size=None
        )
        plt.title(f"SHAP Feature Impact — {cls} Urgency", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        fname = os.path.join(REPORTS_DIR, f"shap_beeswarm_{cls.lower()}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: reports/shap_beeswarm_{cls.lower()}.png")

    # Also keep overall bar chart as shap_summary.png
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(range(len(top15_names)), top15_vals[::-1], color="steelblue", height=0.6)
    ax.set_yticks(range(len(top15_names)))
    ax.set_yticklabels(top15_names[::-1], fontsize=10)
    ax.set_xlabel("Mean |SHAP value| (averaged across all classes)", fontsize=11)
    ax.set_title("Top 15 Features by SHAP Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: reports/shap_summary.png")

    # ── 2. Per-class top-10 bar charts ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"High": "#e74c3c", "Low": "#2ecc71", "Medium": "#f39c12"}

    for idx, cls in enumerate(class_names):
        mean_abs = np.abs(shap_values[:, :, idx]).mean(axis=0)
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
        mean_abs = np.abs(shap_values[:, :, idx]).mean(axis=0)
        top5 = np.argsort(mean_abs)[::-1][:5]
        print(f"  {cls}: {[feature_cols[i] for i in top5]}")


if __name__ == "__main__":
    main()
