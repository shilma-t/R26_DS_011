"""
06_ablation_temporal.py
Fold-locked comparison: baseline vs baseline + temporal features.
Both systems run on the exact same 5 folds so the comparison is fair.
Reports macro-F1 mean ± std and Medium recall mean ± std per variant.
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

TEMPORAL_FEATURES = {
    "rms_slope", "energy_change", "rms_gradient",
    "pitch_slope", "speaking_rate_change",
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
    """Run cross-validation on pre-defined folds. Returns macro-F1 and Medium recall per fold."""
    macro_f1s   = []
    medium_recs = []
    classes = le.classes_.tolist()
    medium_idx = classes.index("Medium")

    for train_idx, test_idx in cv_folds:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        pipe = make_pipeline()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        macro_f1s.append(f1_score(y_test, y_pred, average="macro"))
        rec = recall_score(y_test, y_pred, average=None)
        medium_recs.append(rec[medium_idx] if medium_idx < len(rec) else 0.0)

    return np.array(macro_f1s), np.array(medium_recs)


def main():
    df = pd.read_csv(FEATURES_CSV)
    all_feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    temporal_present = [f for f in TEMPORAL_FEATURES if f in all_feature_cols]
    baseline_cols    = [f for f in all_feature_cols if f not in TEMPORAL_FEATURES]

    print(f"Total features available : {len(all_feature_cols)}")
    print(f"Temporal features found  : {temporal_present}")
    print(f"Baseline features        : {len(baseline_cols)}")
    print(f"With temporal            : {len(all_feature_cols)}")

    y = df["urgency_label"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Fix the folds — both systems use exactly these splits
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fixed_folds = list(cv.split(df, y_enc))

    print("\nRunning fold-locked comparison ...")

    # ── Baseline (no temporal features) ──────────────────────────────────────
    X_base = df[baseline_cols].fillna(0).values
    base_f1, base_med = run_cv(X_base, y_enc, le, fixed_folds)

    # ── With temporal features ────────────────────────────────────────────────
    X_full = df[all_feature_cols].fillna(0).values
    full_f1, full_med = run_cv(X_full, y_enc, le, fixed_folds)

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FOLD-LOCKED ABLATION: Baseline vs + Temporal Features")
    print("=" * 60)
    print(f"\n{'Fold':<6} {'Base Macro-F1':>14} {'+ Temporal':>11} {'Δ':>6}")
    print("-" * 42)
    for i, (b, f) in enumerate(zip(base_f1, full_f1)):
        delta = f - b
        sign  = "+" if delta >= 0 else ""
        print(f"  {i+1:<4} {b:.3f}          {f:.3f}      {sign}{delta:.3f}")

    print("-" * 42)
    delta_mean = full_f1.mean() - base_f1.mean()
    sign = "+" if delta_mean >= 0 else ""
    print(f"  {'Mean':<4} {base_f1.mean():.3f} ± {base_f1.std():.3f}  "
          f"{full_f1.mean():.3f} ± {full_f1.std():.3f}  {sign}{delta_mean:.3f}")

    print("\n" + "=" * 60)
    print("MEDIUM RECALL per fold")
    print("=" * 60)
    print(f"\n{'Fold':<6} {'Base Medium Rec':>15} {'+ Temporal':>11} {'Δ':>6}")
    print("-" * 42)
    for i, (b, f) in enumerate(zip(base_med, full_med)):
        delta = f - b
        sign  = "+" if delta >= 0 else ""
        print(f"  {i+1:<4} {b:.3f}           {f:.3f}      {sign}{delta:.3f}")

    print("-" * 42)
    delta_med_mean = full_med.mean() - base_med.mean()
    sign = "+" if delta_med_mean >= 0 else ""
    print(f"  {'Mean':<4} {base_med.mean():.3f} ± {base_med.std():.3f}  "
          f"{full_med.mean():.3f} ± {full_med.std():.3f}  {sign}{delta_med_mean:.3f}")

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    if delta_mean > 0.01:
        print(f"  Temporal features HELP: +{delta_mean:.3f} macro-F1 consistently across folds.")
    elif delta_mean > 0:
        print(f"  Temporal features give marginal gain (+{delta_mean:.3f}). Check per-fold variance.")
    else:
        print(f"  Temporal features do NOT help on average ({delta_mean:.3f}). Investigate further.")

    if delta_med_mean > 0.02:
        print(f"  Medium recall IMPROVES: +{delta_med_mean:.3f}")
    elif delta_med_mean >= 0:
        print(f"  Medium recall marginal change: +{delta_med_mean:.3f}")
    else:
        print(f"  Medium recall gets WORSE with temporal features: {delta_med_mean:.3f}")


if __name__ == "__main__":
    main()
