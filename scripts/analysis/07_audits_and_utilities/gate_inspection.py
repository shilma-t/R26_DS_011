"""
step3_gate_inspection.py
Inspect the calls changed by the ASR gate (D vs E).

For each call where gate changes features:
  filename, true label, D prediction, E prediction,
  D probability vector, E probability vector,
  transcript, asr_quality, features changed.

Key question: does E correct genuinely bad ASR cases, or just change predictions by chance?
"""

import os
import sys
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))

_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES = os.path.join(_ROOT, "data", "features.csv")
REPORTS  = os.path.join(_ROOT, "reports", "pp2")

FOLD_SEED = 42
N_FOLDS   = 5
RF_PARAMS = dict(n_estimators=200, min_samples_leaf=2,
                 class_weight="balanced", random_state=FOLD_SEED, n_jobs=-1)

LABEL_COL    = "urgency_label"
EXCLUDE_COLS = {"filename", "urgency_label", "language", "transcript"}

ASR_GATE_QUALITY    = 0.3
ASR_GATE_REPETITION = 0.8

# D and E column lists (must match ablation_context.py exactly)
ACOUSTIC = (
    [f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")] +
    ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
     "spec_centroid_mean", "speaking_rate", "jitter",
     "rms_slope", "rms_gradient", "speaking_rate_change"]
)


def build_cols(df_cols):
    acoustic      = [c for c in ACOUSTIC if c in df_cols]
    asr_quality   = ["asr_quality"] if "asr_quality" in df_cols else []
    expanded_text = [c for c in ["keyword_count", "word_count",
                                  "repetition_rate", "sentiment_polarity"]
                     if c in df_cols]
    context       = sorted(c for c in df_cols
                           if c.startswith("sin_") or c.startswith("tam_"))
    return acoustic + asr_quality + expanded_text + context  # D and E share same cols


def apply_gate(df):
    df = df.copy()
    suppress = [c for c in df.columns
                if c not in EXCLUDE_COLS and c != "asr_quality"
                and (c in ["keyword_count", "word_count",
                           "repetition_rate", "sentiment_polarity"]
                     or c.startswith("sin_") or c.startswith("tam_"))]
    suppress += ["asr_quality"]
    suppress  = [c for c in suppress if c in df.columns]
    mask = (
        df["transcript"].isna() |
        (df["transcript"].astype(str).str.strip() == "") |
        (df.get("asr_quality",     pd.Series(1.0, index=df.index)) < ASR_GATE_QUALITY) |
        (df.get("repetition_rate", pd.Series(0.0, index=df.index)) > ASR_GATE_REPETITION)
    )
    df.loc[mask, suppress] = 0.0
    return df, mask


def make_pipeline():
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=FOLD_SEED)),
        ("clf",    RandomForestClassifier(**RF_PARAMS)),
    ])


def main():
    df_raw   = pd.read_csv(FEATURES)
    df_gated, gate_mask = apply_gate(df_raw)

    le  = LabelEncoder()
    y   = le.fit_transform(df_raw[LABEL_COL].values)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)

    cols = build_cols(df_raw.columns)
    cols = [c for c in cols if c in df_raw.columns]

    # Find which calls changed between D and E
    suppress_cols = [c for c in cols if c not in ACOUSTIC and c != "asr_quality"]
    changed_mask  = (df_raw[suppress_cols] != df_gated[suppress_cols]).any(axis=1)
    changed_idx   = df_raw[changed_mask].index.tolist()

    print(f"Calls changed by gate: {len(changed_idx)}")
    for i in changed_idx:
        print(f"  [{i}] {df_raw.loc[i,'filename']}  "
              f"label={df_raw.loc[i,LABEL_COL]}  "
              f"asr_quality={df_raw.loc[i,'asr_quality']:.4f}")

    # Find which fold each changed call lands in
    fold_of = {}
    for fold, (tr, te) in enumerate(skf.split(df_raw, y), 1):
        for i in te:
            fold_of[i] = (fold, tr, te)

    results = []

    for target_idx in changed_idx:
        fold_id, tr, te = fold_of[target_idx]
        pos_in_te = list(te).index(target_idx)

        X_d_tr = df_raw.iloc[tr][cols].fillna(0).values
        X_d_te = df_raw.iloc[te][cols].fillna(0).values
        X_e_tr = df_gated.iloc[tr][cols].fillna(0).values
        X_e_te = df_gated.iloc[te][cols].fillna(0).values
        y_tr   = y[tr]
        y_te   = y[te]

        # Train D
        pipe_d = make_pipeline()
        pipe_d.fit(X_d_tr, y_tr)
        pred_d = pipe_d.predict(X_d_te)
        prob_d = pipe_d.predict_proba(X_d_te)

        # Train E
        pipe_e = make_pipeline()
        pipe_e.fit(X_e_tr, y_tr)
        pred_e = pipe_e.predict(X_e_te)
        prob_e = pipe_e.predict_proba(X_e_te)

        true_label = le.classes_[y_te[pos_in_te]]
        pred_label_d = le.classes_[pred_d[pos_in_te]]
        pred_label_e = le.classes_[pred_e[pos_in_te]]
        proba_d = dict(zip(le.classes_, prob_d[pos_in_te].round(3)))
        proba_e = dict(zip(le.classes_, prob_e[pos_in_te].round(3)))

        row = df_raw.loc[target_idx]

        # Which features changed and by how much
        changed_features = {}
        for c in cols:
            old = df_raw.loc[target_idx, c]
            new = df_gated.loc[target_idx, c]
            if old != new:
                changed_features[c] = (round(old, 4), round(new, 4))

        results.append({
            "filename":       row["filename"],
            "true_label":     true_label,
            "fold":           fold_id,
            "language":       row.get("language", ""),
            "asr_quality":    round(row["asr_quality"], 4),
            "transcript":     str(row["transcript"])[:150],
            "D_prediction":   pred_label_d,
            "E_prediction":   pred_label_e,
            "prediction_changed": pred_label_d != pred_label_e,
            "D_correct":      pred_label_d == true_label,
            "E_correct":      pred_label_e == true_label,
            "D_prob_High":    proba_d.get("High", 0),
            "D_prob_Medium":  proba_d.get("Medium", 0),
            "D_prob_Low":     proba_d.get("Low", 0),
            "E_prob_High":    proba_e.get("High", 0),
            "E_prob_Medium":  proba_e.get("Medium", 0),
            "E_prob_Low":     proba_e.get("Low", 0),
            "features_changed": str(changed_features),
        })

    print("\n" + "=" * 70)
    print("GATE INSPECTION RESULTS")
    print("=" * 70)
    for r in results:
        print(f"\nFile       : {r['filename']}")
        print(f"Language   : {r['language']}")
        print(f"True label : {r['true_label']}")
        print(f"ASR quality: {r['asr_quality']}")
        print(f"Transcript : {r['transcript']}")
        print(f"Fold       : {r['fold']}")
        print(f"D predict  : {r['D_prediction']}  "
              f"[High={r['D_prob_High']} Med={r['D_prob_Medium']} Low={r['D_prob_Low']}]"
              f"  correct={r['D_correct']}")
        print(f"E predict  : {r['E_prediction']}  "
              f"[High={r['E_prob_High']} Med={r['E_prob_Medium']} Low={r['E_prob_Low']}]"
              f"  correct={r['E_correct']}")
        print(f"Prediction changed: {r['prediction_changed']}")
        print(f"Features changed  : {r['features_changed']}")

    out = pd.DataFrame(results)
    path = os.path.join(REPORTS, "gate_inspection.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved → reports/gate_inspection.csv")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    for r in results:
        changed = r["prediction_changed"]
        d_ok    = r["D_correct"]
        e_ok    = r["E_correct"]
        if changed and e_ok and not d_ok:
            verdict = "GATE CORRECTS a genuine ASR error"
        elif changed and d_ok and not e_ok:
            verdict = "GATE INTRODUCES an error (D was correct)"
        elif changed:
            verdict = "Prediction changed — both wrong or context unclear"
        else:
            verdict = "No prediction change"
        print(f"  {r['filename']}: {verdict}")


if __name__ == "__main__":
    main()
