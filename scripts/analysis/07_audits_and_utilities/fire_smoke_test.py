"""
fire_smoke_test.py
Test whether adding a single FIRE/SMOKE binary feature on top of
the updated Condition D (110 features, no immediacy) improves performance.

The FIRE/SMOKE feature fires = 1 if ANY fire or smoke keyword matches
the transcript, regardless of language.

Conditions:
  D_no_imm       — 110 features (Condition D minus immediacy)
  D_fire_smoke   — 111 features (D_no_imm + fire_smoke_match)

Same folds, scaler, SMOTE, RF as main ablation.

Output:
  reports/pp2/fire_smoke_test.csv
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

# FIRE/SMOKE keywords — all languages, all scripts
FIRE_SMOKE_KEYWORDS = [
    # English
    "fire", "smoke", "burning", "flames", "gas",
    # Sinhala script
    "ගිනිගන්නව", "ගිනිඅරගෙන", "ගින්නක්", "ගිනි", "දුමක්", "දුම",
    # Tamil script
    "தீ", "புகை",
    # Romanised Sinhala
    "gini",
]

ACOUSTIC = (
    [f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")] +
    ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
     "spec_centroid_mean", "speaking_rate", "jitter",
     "rms_slope", "rms_gradient", "speaking_rate_change"]
)
TEXT = ["asr_quality", "keyword_count", "word_count",
        "repetition_rate", "sentiment_polarity"]
CONTEXT_NO_IMM = [
    "sin_context_score", "sin_danger_count", "sin_help_request_count",
    "sin_high_risk_phrase_count",
    "sin_location_context_count", "sin_person_at_risk_count", "sin_safety_state_count",
    "tam_context_score", "tam_danger_count", "tam_help_request_count",
    "tam_high_risk_phrase_count",
    "tam_location_context_count", "tam_person_at_risk_count", "tam_safety_state_count",
]

BASE_COLS = ACOUSTIC + TEXT + CONTEXT_NO_IMM   # 110 features


def add_fire_smoke(df):
    """Add binary fire_smoke_match column to df."""
    df = df.copy()
    def match(text):
        if not isinstance(text, str):
            return 0
        t = text.lower()
        return int(any(kw.lower() in t for kw in FIRE_SMOKE_KEYWORDS))
    df["fire_smoke_match"] = df["transcript"].apply(match)
    return df


def make_pipe():
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=FOLD_SEED)),
        ("clf",    RandomForestClassifier(**RF_PARAMS)),
    ])


def run(name, df, cols, y, le, skf):
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

    overall  = classification_report(y_true_all, y_pred_all,
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
    df_raw = pd.read_csv(FEATURES)
    df_fs  = add_fire_smoke(df_raw)

    le  = LabelEncoder()
    y   = le.fit_transform(df_raw["urgency_label"].values)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)

    # Check how many calls fire_smoke_match fires in
    fs_counts = df_fs.groupby(["urgency_label"])["fire_smoke_match"].sum()
    print("=" * 65)
    print("FIRE/SMOKE TEST")
    print("=" * 65)
    print(f"\nfire_smoke_match fires in {df_fs['fire_smoke_match'].sum()} / {len(df_fs)} calls")
    print("By urgency class:")
    print(fs_counts.to_string())

    fs_lang = df_fs[df_fs["fire_smoke_match"] == 1]["language"].value_counts()
    print("\nBy language:")
    print(fs_lang.to_string())

    r_base = run("D_no_imm",     df_raw, BASE_COLS,
                 y, le, skf)
    r_fs   = run("D_fire_smoke", df_fs,  BASE_COLS + ["fire_smoke_match"],
                 y, le, skf)

    print("\n" + "=" * 65)
    print("COMPARISON")
    print("=" * 65)
    for metric in ["macro_f1_mean", "High_recall", "Medium_recall", "Low_recall"]:
        diff = round(r_fs[metric] - r_base[metric], 4)
        direction = "BETTER" if diff > 0.002 else ("WORSE" if diff < -0.002 else "same")
        print(f"  {metric:<20}  base={r_base[metric]:.4f}  "
              f"fire_smoke={r_fs[metric]:.4f}  diff={diff:+.4f}  {direction}")

    print("\nFold-by-fold macro F1:")
    print(f"  {'Fold':<6} {'D_no_imm':>10} {'D_fire_smoke':>13} {'diff':>8}")
    for i, (a, b) in enumerate(zip(r_base["fold_f1s"], r_fs["fold_f1s"]), 1):
        diff = round(b - a, 4)
        print(f"  {i:<6} {a:>10.4f} {b:>13.4f} {diff:>+8.4f}")

    print("\n" + "=" * 65)
    print("VERDICT")
    print("=" * 65)
    diff_mf1 = r_fs["macro_f1_mean"] - r_base["macro_f1_mean"]
    diff_high = r_fs["High_recall"]  - r_base["High_recall"]
    if diff_mf1 > 0.005:
        verdict = "ADD fire_smoke_match — improves macro F1"
    elif diff_high > 0.01:
        verdict = "ADD fire_smoke_match — improves High recall (operationally important)"
    elif diff_mf1 < -0.005:
        verdict = "DO NOT ADD — hurts macro F1"
    else:
        verdict = "NEGLIGIBLE — document finding, do not add"
    print(f"  {verdict}")

    # Save
    out = pd.DataFrame([
        {k: v for k, v in r_base.items() if k != "fold_f1s"},
        {k: v for k, v in r_fs.items()   if k != "fold_f1s"},
    ])
    out.to_csv(os.path.join(REPORTS, "fire_smoke_test.csv"), index=False)
    print(f"\nSaved → reports/pp2/fire_smoke_test.csv")


if __name__ == "__main__":
    main()
