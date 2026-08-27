"""
08_language_analysis.py
Breaks down classification performance by language (Sinhala / Tamil / Singlish / Tanglish / English).
Outputs a heatmap and per-language F1 bar chart to reports/.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

FEATURES_CSV = os.path.join("..", "features.csv")
MODELS_DIR   = os.path.join("..", "models")
REPORTS_DIR  = os.path.join("..", "reports")
NON_FEATURE_COLS = {"filename", "urgency_label", "language", "transcript"}

LANG_GROUPS = {
    "sinhala":  "Sinhala",
    "singlish": "Singlish",
    "tamil":    "Tamil",
    "tanglish": "Tanglish",
    "english":  "English",
}


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].fillna(0).values
    y = df["urgency_label"].values
    lang = df["language"].str.lower().str.strip().map(LANG_GROUPS).fillna("Other").values
    return X, y, lang, feature_cols


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    pipeline     = joblib.load(os.path.join(MODELS_DIR, "rf_urgency.pkl"))
    le           = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))

    X, y, lang, _ = load_data()
    y_enc = le.transform(y)
    labels = le.classes_

    # Cross-validated predictions (same as evaluate.py — unbiased)
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipeline, X, y_enc, cv=cv)
    y_pred_labels = le.inverse_transform(y_pred)

    # ── Per-language F1 table ─────────────────────────────────────────────────
    langs_present = sorted(set(lang))
    records = []
    for lg in langs_present:
        mask = lang == lg
        if mask.sum() < 3:
            continue
        f1_macro = f1_score(y[mask], y_pred_labels[mask],
                            labels=labels, average="macro", zero_division=0)
        f1_per   = f1_score(y[mask], y_pred_labels[mask],
                            labels=labels, average=None, zero_division=0)
        records.append({
            "Language":  lg,
            "N samples": int(mask.sum()),
            "Macro F1":  round(f1_macro, 3),
            **{f"F1 {l}": round(v, 3) for l, v in zip(labels, f1_per)},
        })

    df_lang = pd.DataFrame(records).set_index("Language")
    print(df_lang.to_string())
    df_lang.to_csv(os.path.join(REPORTS_DIR, "language_breakdown.csv"))

    # ── Heatmap: F1 per language × class ─────────────────────────────────────
    heat_cols = [f"F1 {l}" for l in labels]
    heat_data = df_lang[heat_cols].rename(columns=lambda c: c.replace("F1 ", ""))

    fig, ax = plt.subplots(figsize=(7, max(3, len(heat_data) * 0.7 + 1)))
    sns.heatmap(heat_data.astype(float), annot=True, fmt=".2f",
                cmap="RdYlGn", vmin=0, vmax=1, ax=ax,
                linewidths=0.5, cbar_kws={"label": "F1 Score"})
    ax.set_title("F1 Score by Language and Urgency Class", fontsize=13)
    ax.set_xlabel("Urgency Class")
    ax.set_ylabel("Language")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "language_heatmap.png"), dpi=150)
    plt.close()
    print("Saved: reports/language_heatmap.png")

    # ── Macro F1 per language bar chart ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    bar_colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    bars = ax.bar(df_lang.index, df_lang["Macro F1"],
                  color=bar_colors[:len(df_lang)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Model Performance by Language")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", fontsize=11)

    # Annotate sample counts
    for i, lg in enumerate(df_lang.index):
        n = df_lang.loc[lg, "N samples"]
        ax.text(i, 0.02, f"n={n}", ha="center", fontsize=9, color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "language_f1_bar.png"), dpi=150)
    plt.close()
    print("Saved: reports/language_f1_bar.png")


if __name__ == "__main__":
    main()
