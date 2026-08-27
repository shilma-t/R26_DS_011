"""
immediacy_removal_test.py
Test whether removing sin_immediacy_count and tam_immediacy_count
from Condition D improves or hurts performance.

Conditions:
  D_full    — 112 features (Condition D as-is)
  D_no_imm  — 110 features (Condition D minus both immediacy columns)

Same folds, scaler, SMOTE, RF as main ablation.

Output:
  reports/pp2/immediacy_removal_test.csv
"""

import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

_ROOT    = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
FEATURES = os.path.join(_ROOT, "data", "features.csv")
REPORTS  = os.path.join(_ROOT, "reports", "pp2")

FOLD_SEED = 42
N_FOLDS   = 5
RF_PARAMS = dict(n_estimators=200, min_samples_leaf=2,
                 class_weight="balanced", random_state=FOLD_SEED, n_jobs=-1)

ACOUSTIC = (
    [f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")] +
    ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
     "spec_centroid_mean", "speaking_rate", "jitter",
     "rms_slope", "rms_gradient", "speaking_rate_change"]
)
TEXT = ["asr_quality", "keyword_count", "word_count",
        "repetition_rate", "sentiment_polarity"]
CONTEXT = [
    "sin_context_score", "sin_danger_count", "sin_help_request_count",
    "sin_high_risk_phrase_count", "sin_immediacy_count",
    "sin_location_context_count", "sin_person_at_risk_count", "sin_safety_state_count",
    "tam_context_score", "tam_danger_count", "tam_help_request_count",
    "tam_high_risk_phrase_count", "tam_immediacy_count",
    "tam_location_context_count", "tam_person_at_risk_count", "tam_safety_state_count",
]
IMMEDIACY_COLS = ["sin_immediacy_count", "tam_immediacy_count"]

FULL_COLS    = ACOUSTIC + TEXT + CONTEXT
NO_IMM_COLS  = [c for c in FULL_COLS if c not in IMMEDIACY_COLS]


def make_pipe():
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=FOLD_SEED)),
        ("clf",    RandomForestClassifier(**RF_PARAMS)),
    ])


def run(name, cols, df, y, le, skf):
    cols = [c for c in cols if c in df.columns]
    fold_f1s = []
    y_true_all, y_pred_all = [], []

    for tr, te in skf.split(df, y):
        pipe = make_pipe()
        pipe.fit(df.iloc[tr][cols].fillna(0).values, y[tr])
        preds = pipe.predict(df.iloc[te][cols].fillna(0).values)
        y_true_all.extend(y[te])
        y_pred_all.extend(preds)
        rep = classification_report(y[te], preds, target_names=le.classes_,
                                    output_dict=True, zero_division=0)
        fold_f1s.append(rep["macro avg"]["f1-score"])

    overall = classification_report(y_true_all, y_pred_all,
                                    target_names=le.classes_,
                                    output_dict=True, zero_division=0)
    mf1_mean = round(np.mean(fold_f1s), 4)
    mf1_std  = round(np.std(fold_f1s),  4)

    print(f"\n── {name}  ({len(cols)} features)")
    print(f"  Macro F1 : {mf1_mean} ± {mf1_std}")
    for cls in le.classes_:
        print(f"  {cls:<8} recall={overall[cls]['recall']:.3f}  "
              f"f1={overall[cls]['f1-score']:.3f}")

    return {
        "condition":     name,
        "n_features":    len(cols),
        "macro_f1_mean": mf1_mean,
        "macro_f1_std":  mf1_std,
        "High_recall":   round(overall["High"]["recall"],   4),
        "High_f1":       round(overall["High"]["f1-score"], 4),
        "Medium_recall": round(overall["Medium"]["recall"],   4),
        "Medium_f1":     round(overall["Medium"]["f1-score"], 4),
        "Low_recall":    round(overall["Low"]["recall"],   4),
        "Low_f1":        round(overall["Low"]["f1-score"], 4),
        "fold_f1s":      [round(f, 4) for f in fold_f1s],
    }


def main():
    df  = pd.read_csv(FEATURES)
    le  = LabelEncoder()
    y   = le.fit_transform(df["urgency_label"].values)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)

    print("=" * 65)
    print("IMMEDIACY REMOVAL TEST")
    print(f"Removing: {IMMEDIACY_COLS}")
    print("=" * 65)

    r_full   = run("D_full",   FULL_COLS,   df, y, le, skf)
    r_no_imm = run("D_no_imm", NO_IMM_COLS, df, y, le, skf)

    print("\n" + "=" * 65)
    print("COMPARISON")
    print("=" * 65)
    for metric in ["macro_f1_mean", "High_recall", "Medium_recall", "Low_recall"]:
        diff = round(r_no_imm[metric] - r_full[metric], 4)
        direction = "BETTER" if diff > 0 else ("WORSE" if diff < 0 else "same")
        print(f"  {metric:<20}  full={r_full[metric]:.4f}  "
              f"no_imm={r_no_imm[metric]:.4f}  diff={diff:+.4f}  {direction}")

    print("\nFold-by-fold macro F1:")
    print(f"  {'Fold':<6} {'D_full':>8} {'D_no_imm':>10} {'diff':>8}")
    for i, (a, b) in enumerate(zip(r_full["fold_f1s"], r_no_imm["fold_f1s"]), 1):
        diff = round(b - a, 4)
        print(f"  {i:<6} {a:>8.4f} {b:>10.4f} {diff:>+8.4f}")

    print("\n" + "=" * 65)
    print("VERDICT")
    print("=" * 65)
    diff_mf1 = r_no_imm["macro_f1_mean"] - r_full["macro_f1_mean"]
    diff_med  = r_no_imm["Medium_recall"] - r_full["Medium_recall"]
    if diff_mf1 > 0.005:
        verdict = "REMOVE immediacy — improves macro F1"
    elif diff_mf1 < -0.005:
        verdict = "KEEP immediacy — removing hurts macro F1"
    elif diff_med > 0.01:
        verdict = "REMOVE immediacy — improves Medium recall"
    elif diff_med < -0.01:
        verdict = "KEEP immediacy — removing hurts Medium recall"
    else:
        verdict = "NEGLIGIBLE — either decision is defensible; document and move on"
    print(f"  {verdict}")

    # Save
    out = pd.DataFrame([
        {k: v for k, v in r_full.items()   if k != "fold_f1s"},
        {k: v for k, v in r_no_imm.items() if k != "fold_f1s"},
    ])
    out.to_csv(os.path.join(REPORTS, "immediacy_removal_test.csv"),
               index=False)
    print(f"\nSaved → reports/pp2/immediacy_removal_test.csv")


if __name__ == "__main__":
    main()
