"""
08_ablation_per_feature.py
Per-feature ablation study for temporal trajectory features.
Tests each temporal feature individually and in combinations,
all on the same fixed 5 folds for fair comparison.
Reports macro-F1 and Medium recall for each variant.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

FEATURES_CSV = os.path.join("..", "features.csv")
NON_FEATURE_COLS = {"filename", "urgency_label", "language", "transcript"}

TEMPORAL_FEATURES = [
    "rms_slope",
    "pitch_slope",
    "speaking_rate_change",
    "rms_gradient",
    "energy_change",
]

COMBINATIONS = {
    "Baseline (no temporal)":          [],
    "+ rms_slope":                     ["rms_slope"],
    "+ pitch_slope":                   ["pitch_slope"],
    "+ speaking_rate_change":          ["speaking_rate_change"],
    "+ rms_gradient":                  ["rms_gradient"],
    "+ energy_change":                 ["energy_change"],
    "+ pitch only (pitch_slope)":      ["pitch_slope"],
    "+ energy only (rms_slope+grad+ec)":["rms_slope", "rms_gradient", "energy_change"],
    "+ all 5 temporal":                ["rms_slope", "pitch_slope", "speaking_rate_change",
                                        "rms_gradient", "energy_change"],
}


def make_pipeline():
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=42)),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def run_cv(X, y_enc, le, cv_folds):
    macro_f1s   = []
    medium_recs = []
    high_recs   = []
    classes = le.classes_.tolist()
    medium_idx = classes.index("Medium")
    high_idx   = classes.index("High")

    for train_idx, test_idx in cv_folds:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        pipe = make_pipeline()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        macro_f1s.append(f1_score(y_test, y_pred, average="macro"))
        rec = recall_score(y_test, y_pred, average=None, labels=sorted(set(y_enc)))
        medium_recs.append(rec[medium_idx] if medium_idx < len(rec) else 0.0)
        high_recs.append(rec[high_idx]     if high_idx   < len(rec) else 0.0)

    return np.array(macro_f1s), np.array(medium_recs), np.array(high_recs)


def main():
    df = pd.read_csv(FEATURES_CSV)
    all_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    base_cols = [c for c in all_cols if c not in TEMPORAL_FEATURES]

    y = df["urgency_label"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Fix folds — every variant uses exactly the same splits
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fixed_folds = list(cv.split(df, y_enc))

    results = []
    baseline_f1 = None

    print("Running ablation ... (this may take a minute)\n")

    for label, extra_feats in COMBINATIONS.items():
        # Only add features that actually exist in the data
        extra_feats = [f for f in extra_feats if f in all_cols]
        cols = base_cols + extra_feats
        X = df[cols].fillna(0).values

        f1s, med_recs, high_recs = run_cv(X, y_enc, le, fixed_folds)

        if baseline_f1 is None:
            baseline_f1 = f1s.mean()

        delta = f1s.mean() - baseline_f1

        results.append({
            "Variant":        label,
            "Features added": len(extra_feats),
            "Macro F1":       f1s.mean(),
            "F1 std":         f1s.std(),
            "Medium Recall":  med_recs.mean(),
            "Med Rec std":    med_recs.std(),
            "High Recall":    high_recs.mean(),
            "Δ Macro F1":     delta,
        })
        sign = "+" if delta >= 0 else ""
        print(f"  {label}")
        print(f"    Macro F1:     {f1s.mean():.3f} ± {f1s.std():.3f}  ({sign}{delta:.3f} vs baseline)")
        print(f"    Medium Recall:{med_recs.mean():.3f} ± {med_recs.std():.3f}")
        print(f"    High Recall:  {high_recs.mean():.3f} ± {high_recs.std():.3f}")
        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print("=" * 75)
    print("ABLATION SUMMARY TABLE")
    print("=" * 75)
    print(f"{'Variant':<42} {'MacroF1':>8} {'Med Rec':>8} {'Hi Rec':>8} {'Δ F1':>7}")
    print("-" * 75)
    for r in results:
        sign = "+" if r["Δ Macro F1"] >= 0 else ""
        print(f"  {r['Variant']:<40} {r['Macro F1']:>7.3f}  "
              f"{r['Medium Recall']:>7.3f}  {r['High Recall']:>7.3f}  "
              f"{sign}{r['Δ Macro F1']:>5.3f}")

    # ── Best single feature for Medium recall ─────────────────────────────────
    single_feats = [r for r in results if r["Features added"] == 1]
    if single_feats:
        best_med = max(single_feats, key=lambda r: r["Medium Recall"])
        best_f1  = max(single_feats, key=lambda r: r["Macro F1"])
        print("\n" + "=" * 75)
        print("BEST SINGLE FEATURE")
        print("=" * 75)
        print(f"  Best for Medium Recall : {best_med['Variant']}  "
              f"→ Medium Recall {best_med['Medium Recall']:.3f}")
        print(f"  Best for Macro F1      : {best_f1['Variant']}  "
              f"→ Macro F1 {best_f1['Macro F1']:.3f}")

    # Save results to CSV
    out_path = os.path.join("..", "reports", "ablation_temporal.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
