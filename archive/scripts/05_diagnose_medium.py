import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix

FEATURES_CSV = os.path.join("..", "features.csv")
MODELS_DIR   = os.path.join("..", "models")
REPORTS_DIR  = os.path.join("..", "reports")
NON_FEATURE_COLS = {"filename", "urgency_label", "language", "transcript"}


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].fillna(0).values
    y = df["urgency_label"].values

    pipeline = joblib.load(os.path.join(MODELS_DIR, "rf_urgency.pkl"))
    le       = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    y_enc    = le.transform(y)

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipeline, X, y_enc, cv=cv)
    y_pred_labels = le.inverse_transform(y_pred)

    df["predicted"] = y_pred_labels

    # ── 1. Where do Medium calls go? ─────────────────────────────────────────
    print("=" * 60)
    print("1. MEDIUM CALL MISCLASSIFICATION BREAKDOWN")
    print("=" * 60)
    medium_df = df[df["urgency_label"] == "Medium"]
    pred_counts = medium_df["predicted"].value_counts()
    total_medium = len(medium_df)
    print(f"Total Medium calls: {total_medium}")
    for label, count in pred_counts.items():
        pct = count / total_medium * 100
        marker = " ← correct" if label == "Medium" else " ← MISCLASSIFIED"
        print(f"  Predicted {label:>6}: {count:3d}  ({pct:.1f}%){marker}")

    # ── 2. Language breakdown of Medium errors ────────────────────────────────
    print("\n" + "=" * 60)
    print("2. MEDIUM ERRORS BY LANGUAGE")
    print("=" * 60)
    if "language" in df.columns:
        lang_breakdown = medium_df.groupby("language")["predicted"].value_counts().unstack(fill_value=0)
        print(lang_breakdown.to_string())

        # Also show accuracy per language for Medium
        print("\nMedium recall per language:")
        for lang, grp in medium_df.groupby("language"):
            correct = (grp["predicted"] == "Medium").sum()
            total   = len(grp)
            print(f"  {lang:>12}: {correct}/{total}  ({correct/total*100:.1f}% recall)")

    # ── 3. Feature distribution: Medium vs High vs Low ────────────────────────
    print("\n" + "=" * 60)
    print("3. KEY FEATURE DISTRIBUTIONS (mean per class)")
    print("=" * 60)
    key_features = [
        "rms_mean", "rms_std", "rms_slope", "rms_gradient",
        "pitch_mean", "pitch_std", "pitch_slope",
        "speaking_rate", "speaking_rate_change",
        "keyword_count", "repetition_rate",
    ]
    key_features = [f for f in key_features if f in df.columns]

    stats = df.groupby("urgency_label")[key_features].mean()
    print(stats.round(4).to_string())

    # ── 4. Feature separability: how well does each feature separate Medium ───
    print("\n" + "=" * 60)
    print("4. FEATURE SEPARABILITY FOR MEDIUM")
    print("   (difference between Medium mean and nearest other class mean)")
    print("=" * 60)
    separability = {}
    for feat in feature_cols:
        means = df.groupby("urgency_label")[feat].mean()
        if "Medium" not in means.index:
            continue
        med_mean = means["Medium"]
        other_means = means.drop("Medium")
        # distance to nearest class (smaller = harder to separate)
        min_dist = (other_means - med_mean).abs().min()
        separability[feat] = min_dist

    sep_series = pd.Series(separability).sort_values()
    print("\nHARDEST to separate (Medium most confused with other classes):")
    print(sep_series.head(10).round(6).to_string())
    print("\nEASIEST to separate:")
    print(sep_series.tail(10).round(6).to_string())

    # ── 5. Plot: Medium misclassification by language ─────────────────────────
    if "language" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: where Medium calls go overall
        ax = axes[0]
        colors = {"Medium": "#f39c12", "High": "#e74c3c", "Low": "#2ecc71"}
        vals   = [pred_counts.get(l, 0) for l in ["High", "Medium", "Low"]]
        bars   = ax.bar(["→ High", "→ Medium\n(correct)", "→ Low"],
                        vals,
                        color=[colors["High"], colors["Medium"], colors["Low"]])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(val), ha="center", fontsize=11)
        ax.set_title("Where Medium Calls Are Predicted")
        ax.set_ylabel("Count")
        ax.set_ylim(0, max(vals) + 5)

        # Right: Medium recall by language
        ax = axes[1]
        lang_recalls = {}
        for lang, grp in medium_df.groupby("language"):
            correct = (grp["predicted"] == "Medium").sum()
            lang_recalls[lang] = correct / len(grp) * 100
        langs = list(lang_recalls.keys())
        recalls = [lang_recalls[l] for l in langs]
        bar_colors = ["#e74c3c" if r < 40 else "#f39c12" if r < 70 else "#2ecc71" for r in recalls]
        ax.bar(langs, recalls, color=bar_colors)
        ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="50% threshold")
        ax.set_title("Medium Recall by Language")
        ax.set_ylabel("Recall (%)")
        ax.set_ylim(0, 110)
        for i, (lang, r) in enumerate(zip(langs, recalls)):
            ax.text(i, r + 2, f"{r:.0f}%", ha="center", fontsize=10)
        ax.legend()

        plt.suptitle("Medium Class Diagnosis", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "medium_diagnosis.png"), dpi=150)
        plt.close()
        print(f"\nPlot saved to {REPORTS_DIR}/medium_diagnosis.png")

    # ── 6. Show misclassified Medium samples ─────────────────────────────────
    print("\n" + "=" * 60)
    print("5. SAMPLE MISCLASSIFIED MEDIUM CALLS")
    print("=" * 60)
    misclassified = medium_df[medium_df["predicted"] != "Medium"][
        ["filename", "language", "predicted"] +
        [f for f in ["rms_mean", "pitch_mean", "keyword_count", "rms_slope"] if f in df.columns]
    ]
    print(misclassified.to_string(index=False))


if __name__ == "__main__":
    main()
