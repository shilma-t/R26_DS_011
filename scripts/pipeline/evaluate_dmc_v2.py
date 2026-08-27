"""
evaluate_dmc_v2.py
Locked v2 external evaluation: same frozen 111-feature definition and RF
hyperparameters as v1, but text/context features come from Component 2's
ASR (raw, no refinement - same methodology used to build features_v2.csv)
instead of the original whisper-small-sinhala_v3.

Acoustic extraction and the denoise+trim preprocessing are UNCHANGED from
v1 - only the transcript source differs, exactly as for training.

Usage:
  python scripts/pipeline/evaluate_dmc_v2.py --limit 5   # benchmark
  python scripts/pipeline/evaluate_dmc_v2.py               # full 59 calls
"""
import os
import sys
import argparse
import importlib.util
from datetime import datetime

import numpy as np
import pandas as pd
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

_ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
BACKEND = r"C:\Users\shafr\Desktop\component 2\backend"
PIPE_DIR = os.path.join(_ROOT, "scripts", "pipeline")
CALLER_DIR = os.path.join(_ROOT, "audio", "dmc_caller")
METADATA = os.path.join(_ROOT, "data", "dmc_metadata.csv")
FEATURES_V2 = os.path.join(_ROOT, "data", "features_v2.csv")
REPORTS = os.path.join(_ROOT, "reports", "pp2")

for p in (PIPE_DIR, BACKEND):
    if p not in sys.path:
        sys.path.insert(0, p)

TARGET_SR = 16_000
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

LANG_TO_CODE = {"sinhala": "si-LK", "singlish": "si-LK",
                "tamil": "ta-IN", "tanglish": "ta-IN", "english": "en-US"}


def load_frozen():
    path = os.path.join(PIPE_DIR, "02_extract_features.py")
    spec = importlib.util.spec_from_file_location("frozen_features", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def preprocess_like_training(y, sr=TARGET_SR):
    import noisereduce as nr
    y_nr = nr.reduce_noise(y=y, sr=sr)
    if sr != TARGET_SR:
        y_nr = librosa.resample(y_nr, orig_sr=sr, target_sr=TARGET_SR)
    y_trimmed, _ = librosa.effects.trim(y_nr, top_db=20)
    return y_nr if y_trimmed.size < TARGET_SR // 2 else y_trimmed


def fire_smoke_match(text):
    kws = ["fire","smoke","burning","flames","gas","ගිනිගන්නව","ගිනිඅරගෙන",
           "ගින්නක්","ගිනි","දුමක්","දුම","தீ","புகை","gini"]
    if not isinstance(text, str): return 0
    t = text.lower()
    return int(any(k.lower() in t for k in kws))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from services.whisper_engine import transcribe_with_forced_language
    from text_features_from_transcript import text_features_from_transcript

    frozen = load_frozen()

    meta = pd.read_csv(METADATA, dtype={"recording_id": str})
    meta = meta.dropna(subset=["dmc_priority_label"])
    if args.limit:
        meta = meta.head(args.limit)
        print(f"*** SMOKE TEST: {args.limit} call(s) only ***")

    print(f"DMC calls: {len(meta)}")

    rows = []
    for i, row in enumerate(meta.itertuples(), 1):
        rec_id = os.path.splitext(str(row.filename))[0]
        path = os.path.join(CALLER_DIR, f"{rec_id}_caller.wav")
        if not os.path.exists(path):
            print(f"  [{i}] MISSING {rec_id}")
            continue

        y, sr = librosa.load(path, sr=TARGET_SR)
        y = preprocess_like_training(y, TARGET_SR)

        lang_code = LANG_TO_CODE.get(str(row.language).lower().strip(), "en-US")
        result = transcribe_with_forced_language(path, lang_code)
        # re-load+preprocess needed BEFORE acoustic extraction (audio var y above)
        acoustic = frozen.extract_acoustic(y, TARGET_SR)
        textual = text_features_from_transcript(result.text, lang_code)

        rows.append({
            "recording_id": rec_id, "true_label": row.dmc_priority_label,
            **acoustic,
            **{k: v for k, v in textual.items() if k not in ("transcript","language_normalised")},
            "fire_smoke_match": fire_smoke_match(result.text),
            "transcript": result.text,
        })
        print(f"  [{i}/{len(meta)}] {rec_id[:30]:<30} kw={textual['keyword_count']:.1f}  {result.text[:44]}")

    if args.limit:
        return

    df = pd.DataFrame(rows)
    v2 = pd.read_csv(FEATURES_V2)
    le = LabelEncoder().fit(["High","Low","Medium"])
    model = ImbPipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state=SEED)),
                         ("clf", RandomForestClassifier(**RF_PARAMS))])
    model.fit(v2[FEATURE_COLS].fillna(0).values, le.transform(v2["urgency_label"].values))

    X = df[FEATURE_COLS].fillna(0).values
    y_true = le.transform(df["true_label"].values)
    y_pred = model.predict(X)
    macro = f1_score(y_true, y_pred, average="macro")
    print(f"\nmacro-F1: {macro:.4f}")
    print(classification_report(y_true, y_pred, target_names=le.classes_, digits=3))
    df.to_csv(os.path.join(REPORTS, "dmc_evaluation_v2.csv"), index=False)


if __name__ == "__main__":
    main()
