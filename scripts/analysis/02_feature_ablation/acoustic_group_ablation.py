"""
acoustic_group_ablation.py
Run acoustic sub-group ablation using the same frozen setup as the main ablation.

Conditions:
  A1 — MFCC only             (80 features: mfcc_1-40 mean + std)
  A2 — Pitch + prosody only  (pitch_mean, pitch_std, jitter, speaking_rate,
                               speaking_rate_change)
  A3 — Energy only           (rms_mean, rms_std, rms_slope, rms_gradient,
                               zcr_mean)
  A4 — Spectral only         (spec_centroid_mean)
  A5 — All acoustic          (91 features — same as Condition A in main ablation)

Everything else is identical to the main ablation:
  Same folds (StratifiedKFold, seed=42)
  Same scaler (StandardScaler, fit on train only)
  Same SMOTE (fit on train only, after scaler)
  Same Random Forest (n_estimators=200, min_samples_leaf=2,
                      class_weight=balanced, seed=42)

Outputs:
  reports/pp2/acoustic_group_ablation.csv    — per-condition summary
  reports/pp2/acoustic_group_folds.csv       — per-fold per-condition results
"""

import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

_ROOT    = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
FEATURES = os.path.join(_ROOT, "data", "features.csv")
REPORTS  = os.path.join(_ROOT, "reports", "pp2")

FOLD_SEED = 42
N_FOLDS   = 5
RF_PARAMS = dict(n_estimators=200, min_samples_leaf=2,
                 class_weight="balanced", random_state=FOLD_SEED, n_jobs=-1)

# ── Feature groups ────────────────────────────────────────────────────────────
MFCC = [f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")]

PITCH_PROSODY = [
    "pitch_mean", "pitch_std", "jitter",
    "speaking_rate", "speaking_rate_change",
]

ENERGY = [
    "rms_mean", "rms_std", "rms_slope", "rms_gradient", "zcr_mean",
]

SPECTRAL = [
    "spec_centroid_mean",
]

ALL_ACOUSTIC = MFCC + PITCH_PROSODY + ENERGY + SPECTRAL

CONDITIONS = {
    "A1_mfcc":          MFCC,
    "A2_pitch_prosody": PITCH_PROSODY,
    "A3_energy":        ENERGY,
    "A4_spectral":      SPECTRAL,
    "A5_all_acoustic":  ALL_ACOUSTIC,
}


def make_pipe():
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=FOLD_SEED)),
        ("clf",    RandomForestClassifier(**RF_PARAMS)),
    ])


def run_condition(name, cols, df, y, le, skf):
    cols = [c for c in cols if c in df.columns]
    print(f"\n── {name}  ({len(cols)} features)")

    fold_rows = []
    y_true_all, y_pred_all = [], []

    for fold, (tr, te) in enumerate(skf.split(df, y), 1):
        X_tr = df.iloc[tr][cols].fillna(0).values
        X_te = df.iloc[te][cols].fillna(0).values
        pipe = make_pipe()
        pipe.fit(X_tr, y[tr])
        preds = pipe.predict(X_te)

        y_true_all.extend(y[te])
        y_pred_all.extend(preds)

        report = classification_report(
            y[te], preds, target_names=le.classes_,
            output_dict=True, zero_division=0
        )
        macro_f1 = report["macro avg"]["f1-score"]

        for cls in le.classes_:
            fold_rows.append({
                "condition": name,
                "n_features": len(cols),
                "fold": fold,
                "class": cls,
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"],    4),
                "f1":        round(report[cls]["f1-score"],  4),
                "macro_f1":  round(macro_f1, 4),
            })

    # Overall report across all folds
    overall = classification_report(
        y_true_all, y_pred_all,
        target_names=le.classes_, output_dict=True, zero_division=0
    )
    macro_f1_mean = round(
        np.mean([r["macro_f1"] for r in fold_rows
                 if r["class"] == le.classes_[0]]), 4
    )
    macro_f1_std = round(
        np.std([r["macro_f1"] for r in fold_rows
                if r["class"] == le.classes_[0]]), 4
    )

    print(f"  Macro F1 : {macro_f1_mean} ± {macro_f1_std}")
    for cls in le.classes_:
        print(f"  {cls:<8} recall={overall[cls]['recall']:.3f}  "
              f"f1={overall[cls]['f1-score']:.3f}")

    return fold_rows, {
        "condition":       name,
        "n_features":      len(cols),
        "macro_f1_mean":   macro_f1_mean,
        "macro_f1_std":    macro_f1_std,
        "High_recall":     round(overall["High"]["recall"],   4),
        "High_f1":         round(overall["High"]["f1-score"], 4),
        "Medium_recall":   round(overall["Medium"]["recall"],   4),
        "Medium_f1":       round(overall["Medium"]["f1-score"], 4),
        "Low_recall":      round(overall["Low"]["recall"],   4),
        "Low_f1":          round(overall["Low"]["f1-score"], 4),
    }


def main():
    df  = pd.read_csv(FEATURES)
    le  = LabelEncoder()
    y   = le.fit_transform(df["urgency_label"].values)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)

    print("=" * 65)
    print("ACOUSTIC GROUP ABLATION")
    print(f"Dataset: {len(df)} calls   Folds: {N_FOLDS}   Seed: {FOLD_SEED}")
    print("=" * 65)

    all_fold_rows = []
    summary_rows  = []

    for name, cols in CONDITIONS.items():
        fold_rows, summary = run_condition(name, cols, df, y, le, skf)
        all_fold_rows.extend(fold_rows)
        summary_rows.append(summary)

    # ── Print summary table ───────────────────────────────────────────────────
    sdf = pd.DataFrame(summary_rows)
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(sdf[["condition","n_features","macro_f1_mean","macro_f1_std",
               "High_recall","Medium_recall","Low_recall"]].to_string(index=False))

    print("\n" + "=" * 65)
    print("INTERPRETATION")
    print("=" * 65)
    best = sdf.loc[sdf["macro_f1_mean"].idxmax(), "condition"]
    print(f"  Best condition by macro F1: {best}")

    for _, row in sdf.iterrows():
        gap = round(sdf.loc[sdf["condition"]=="A5_all_acoustic","macro_f1_mean"].values[0]
                    - row["macro_f1_mean"], 4)
        print(f"  {row['condition']:<20}  macro_f1={row['macro_f1_mean']:.4f}  "
              f"gap_from_A5={gap:+.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    sdf.to_csv(os.path.join(REPORTS, "acoustic_group_ablation.csv"),
               index=False)
    pd.DataFrame(all_fold_rows).to_csv(
        os.path.join(REPORTS, "acoustic_group_folds.csv"), index=False)

    print(f"\nSaved → reports/pp2/acoustic_group_ablation.csv")
    print(f"Saved → reports/pp2/acoustic_group_folds.csv")


if __name__ == "__main__":
    main()
