"""
compare_v1_v2_cv.py
v1 vs v2 macro-F1 on the simulated corpus, same 5-fold CV protocol as the
frozen pipeline (StratifiedKFold, SMOTE+scaler fit inside each fold only).
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

SEED = 42
RF_PARAMS = dict(n_estimators=200, min_samples_leaf=2,
                 class_weight="balanced", random_state=SEED, n_jobs=-1)

ACOUSTIC = ([f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")] +
    ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
     "spec_centroid_mean", "speaking_rate", "jitter",
     "rms_slope", "rms_gradient", "speaking_rate_change"])
TEXT = ["asr_quality", "keyword_count", "word_count", "repetition_rate", "sentiment_polarity"]
CONTEXT = ["sin_context_score", "sin_danger_count", "sin_help_request_count",
    "sin_high_risk_phrase_count", "sin_location_context_count",
    "sin_person_at_risk_count", "sin_safety_state_count",
    "tam_context_score", "tam_danger_count", "tam_help_request_count",
    "tam_high_risk_phrase_count", "tam_location_context_count",
    "tam_person_at_risk_count", "tam_safety_state_count"]
FEATURE_COLS = ACOUSTIC + TEXT + CONTEXT + ["fire_smoke_match"]

def fire_smoke_match(text):
    kws = ["fire","smoke","burning","flames","gas","ගිනිගන්නව","ගිනිඅරගෙන",
           "ගින්නක්","ගිනි","දුමක්","දුම","தீ","புகை","gini"]
    if not isinstance(text, str): return 0
    t = text.lower()
    return int(any(k.lower() in t for k in kws))

def evaluate(df, label):
    df = df.copy()
    if "fire_smoke_match" not in df.columns:
        df["fire_smoke_match"] = df["transcript"].map(fire_smoke_match)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{label}: missing columns {missing}")

    X = df[FEATURE_COLS].fillna(0).values
    le = LabelEncoder().fit(["High","Low","Medium"])
    y = le.transform(df["urgency_label"].values)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_true, all_pred = [], []
    for train_idx, test_idx in skf.split(X, y):
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=SEED)),
            ("clf", RandomForestClassifier(**RF_PARAMS)),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        all_true.extend(y[test_idx]); all_pred.extend(pred)

    macro = f1_score(all_true, all_pred, average="macro")
    print(f"\n=== {label} ===")
    print(f"macro-F1: {macro:.4f}")
    print(classification_report(all_true, all_pred, target_names=le.classes_, digits=3))
    return macro

v1 = pd.read_csv(r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011\data\features.csv")
v2 = pd.read_csv(r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011\data\features_v2.csv")

f1_v1 = evaluate(v1, "v1 (original ASR)")
f1_v2 = evaluate(v2, "v2 (Component 2's ASR)")

print(f"\n{'='*40}")
print(f"v1 macro-F1: {f1_v1:.4f}")
print(f"v2 macro-F1: {f1_v2:.4f}")
print(f"delta: {f1_v2-f1_v1:+.4f}")
