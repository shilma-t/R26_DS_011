"""
medium_error_analysis.py
Save full error analysis for all misclassified calls (Condition D, frozen labels).

Root cause categories:
  ASR_GARBAGE     - asr_quality=1.0 but transcript is clearly wrong script/language
  ASR_FAILURE     - asr_quality=0.0 or near-zero / transcript empty
  ZERO_TEXT       - valid ASR but no keywords and no context score fired
  ACOUSTIC_ONLY   - model had to rely purely on acoustics (no text signal)
  LABEL_AMBIGUOUS - call appears in the 40-call kappa set with annotator disagreement
  CONTEXT_MISS    - keywords present but safety/negation context not captured
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
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
CONTEXT = [
    "sin_context_score","sin_danger_count","sin_help_request_count",
    "sin_high_risk_phrase_count","sin_immediacy_count",
    "sin_location_context_count","sin_person_at_risk_count","sin_safety_state_count",
    "tam_context_score","tam_danger_count","tam_help_request_count",
    "tam_high_risk_phrase_count","tam_immediacy_count",
    "tam_location_context_count","tam_person_at_risk_count","tam_safety_state_count",
]

# 40-call kappa subset (calls where annotator disagreed with Shilma)
KAPPA_DISAGREEMENTS = {
    "call37.ogg", "call244.ogg", "call39.ogg", "call335.ogg",
    "call48.ogg", "call73.ogg",
}

# Characters that look valid to asr_quality but are not Sinhala/Tamil/English
GARBAGE_SCRIPTS = set(range(0x0600, 0x06FF+1))  # Arabic
GARBAGE_SCRIPTS |= set(range(0x0400, 0x04FF+1)) # Cyrillic
GARBAGE_SCRIPTS |= set(range(0x3040, 0x30FF+1)) # Hiragana/Katakana
GARBAGE_SCRIPTS |= set(range(0x4E00, 0x9FFF+1)) # CJK


def has_garbage_script(text):
    if not isinstance(text, str):
        return False
    return any(ord(c) in GARBAGE_SCRIPTS for c in text)


def root_cause(row):
    asr_q  = row.get("asr_quality", 1.0)
    trans  = str(row.get("transcript", ""))
    kw     = row.get("keyword_count", 0)
    sin_c  = row.get("sin_context_score", 0)
    tam_c  = row.get("tam_context_score", 0)
    fname  = row.get("filename", "")

    if asr_q < 0.05 or trans.strip() in ("", "nan"):
        return "ASR_FAILURE"
    if has_garbage_script(trans):
        return "ASR_GARBAGE"
    if kw == 0 and sin_c == 0 and tam_c == 0:
        return "ZERO_TEXT"
    if kw > 0 and sin_c == 0 and tam_c == 0:
        return "CONTEXT_MISS"
    if fname in KAPPA_DISAGREEMENTS:
        return "LABEL_AMBIGUOUS"
    return "ACOUSTIC_ONLY"


def make_pipe():
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=FOLD_SEED)),
        ("clf",    RandomForestClassifier(**RF_PARAMS)),
    ])


def main():
    df  = pd.read_csv(FEATURES)
    le  = LabelEncoder()
    y   = le.fit_transform(df["urgency_label"].values)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)

    d_cols = (
        [c for c in ACOUSTIC if c in df.columns] +
        [c for c in ["asr_quality","keyword_count","word_count",
                     "repetition_rate","sentiment_polarity"] if c in df.columns] +
        [c for c in CONTEXT if c in df.columns]
    )

    # Collect per-call: all fold predictions + probabilities
    call_preds = {i: [] for i in range(len(df))}
    call_probs = {i: [] for i in range(len(df))}

    for tr, te in skf.split(df, y):
        X_tr = df.iloc[tr][d_cols].fillna(0).values
        X_te = df.iloc[te][d_cols].fillna(0).values
        pipe = make_pipe()
        pipe.fit(X_tr, y[tr])
        preds = pipe.predict(X_te)
        probs = pipe.predict_proba(X_te)
        for pos, idx in enumerate(te):
            call_preds[idx].append(le.classes_[preds[pos]])
            call_probs[idx].append(dict(zip(le.classes_, probs[pos].round(3))))

    def majority(lst):
        return Counter(lst).most_common(1)[0][0]

    def avg_probs(prob_list, cls):
        return round(np.mean([p.get(cls, 0) for p in prob_list]), 3)

    # Build error rows — all calls where majority prediction != true label
    rows = []
    for idx, row in df.iterrows():
        true  = row["urgency_label"]
        pred  = majority(call_preds[idx])
        if true == pred:
            continue

        prob_list = call_probs[idx]
        rows.append({
            "filename":        row["filename"],
            "language":        row.get("language", ""),
            "true_label":      true,
            "predicted_label": pred,
            "error_type":      f"{true}→{pred}",
            "prob_High":       avg_probs(prob_list, "High"),
            "prob_Medium":     avg_probs(prob_list, "Medium"),
            "prob_Low":        avg_probs(prob_list, "Low"),
            "asr_quality":     round(row.get("asr_quality", 1.0), 3),
            "keyword_count":   row.get("keyword_count", 0),
            "keyword_count_original": row.get("keyword_count_original", 0),
            "sin_context_score": row.get("sin_context_score", 0),
            "tam_context_score": row.get("tam_context_score", 0),
            "pitch_mean":      round(row.get("pitch_mean", 0), 1),
            "rms_mean":        round(row.get("rms_mean", 0), 4),
            "speaking_rate":   round(row.get("speaking_rate", 0), 2),
            "root_cause":      root_cause(row),
            "transcript":      str(row.get("transcript", ""))[:150],
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["error_type", "root_cause", "filename"]).reset_index(drop=True)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("=" * 65)
    print("MEDIUM ERROR ANALYSIS — Condition D (frozen labels)")
    print("=" * 65)

    print("\nError type counts:")
    print(out["error_type"].value_counts().to_string())

    print("\nRoot cause counts (all errors):")
    print(out["root_cause"].value_counts().to_string())

    print("\nRoot cause breakdown per error type:")
    print(out.groupby(["error_type","root_cause"]).size().to_string())

    print("\nLanguage breakdown of Medium→High errors:")
    mh = out[out["error_type"] == "Medium→High"]
    print(mh["language"].value_counts().to_string())

    print("\nLanguage breakdown of High→Medium/Low errors:")
    hm = out[out["error_type"].str.startswith("High")]
    print(hm["language"].value_counts().to_string())

    # ── Save ─────────────────────────────────────────────────────────────────
    path = os.path.join(REPORTS, "medium_error_analysis.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(out)} error rows → reports/pp2/medium_error_analysis.csv")


if __name__ == "__main__":
    main()
