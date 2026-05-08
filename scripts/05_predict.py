"""
05_predict.py
Demo script — classify a single audio file end-to-end.
Usage:  python 05_predict.py path/to/call.ogg
"""

import sys
import os
import tempfile
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
import joblib
import whisper
from collections import Counter

MODELS_DIR = os.path.join("..", "models")
TARGET_SR  = 16_000
N_MFCC     = 40

URGENCY_KEYWORDS = [
    "help", "emergency", "fire", "accident", "blood", "dying", "dead",
    "trapped", "attack", "knife", "gun", "hurt", "injured", "crash",
    "please", "ambulance", "police", "quickly", "fast", "now",
    "udaw", "hadisi", "ape", "maru", "wedi", "gini",
]

NON_FEATURE_COLS = {"filename", "urgency_label", "language", "transcript"}


def preprocess(src_path: str) -> np.ndarray:
    y, sr = librosa.load(src_path, sr=None, mono=True)
    y     = nr.reduce_noise(y=y, sr=sr)
    y     = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    y, _  = librosa.effects.trim(y, top_db=20)
    return y


def extract_acoustic(y: np.ndarray, sr: int) -> dict:
    feats = {}
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1}_std"]  = float(np.std(mfcc[i]))

    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
    voiced_f0 = f0[voiced] if voiced is not None and voiced.any() else np.array([0.0])
    feats["pitch_mean"] = float(np.nanmean(voiced_f0))
    feats["pitch_std"]  = float(np.nanstd(voiced_f0))

    rms = librosa.feature.rms(y=y)[0]
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"]  = float(np.std(rms))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats["zcr_mean"] = float(np.mean(zcr))

    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats["spec_centroid_mean"] = float(np.mean(sc))

    hop = 512
    total_frames  = len(y) / hop
    voiced_frames = int(np.sum(voiced)) if voiced is not None else 0
    feats["speaking_rate"] = voiced_frames / max(total_frames, 1)
    feats["jitter"] = float(np.mean(np.abs(np.diff(voiced_f0)))) if voiced_f0.size > 1 else 0.0

    return feats


def extract_textual(wav_path: str, model) -> dict:
    result = model.transcribe(wav_path, task="transcribe")
    text   = result.get("text", "").lower()
    words  = text.split()
    feats  = {}

    feats["keyword_count"] = sum(kw.lower() in text for kw in URGENCY_KEYWORDS)
    feats["word_count"]    = len(words)

    counts   = Counter(words)
    repeated = sum(1 for w, c in counts.items() if c > 1)
    feats["repetition_rate"] = repeated / max(len(counts), 1)

    positive = {"safe", "okay", "fine", "calm", "alright"}
    negative = {"help", "hurt", "dying", "trapped", "fire", "blood",
                "attack", "crash", "emergency", "pain", "dead"}
    pos = sum(1 for w in words if w in positive)
    neg = sum(1 for w in words if w in negative)
    feats["sentiment_polarity"] = (pos - neg) / max(len(words), 1)
    feats["transcript"] = result.get("text", "")
    return feats


def main(audio_path: str):
    pipeline     = joblib.load(os.path.join(MODELS_DIR, "rf_urgency.pkl"))
    le           = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))

    print("Loading Whisper ...")
    whisper_model = whisper.load_model("base")

    print(f"Preprocessing {audio_path} ...")
    y = preprocess(audio_path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, y, TARGET_SR)

    acoustic = extract_acoustic(y, TARGET_SR)
    textual  = extract_textual(tmp_path, whisper_model)
    os.unlink(tmp_path)

    all_feats = {**acoustic, **textual}
    X_row = np.array([all_feats.get(col, 0.0) for col in feature_cols]).reshape(1, -1)

    pred_enc   = pipeline.predict(X_row)[0]
    pred_label = le.inverse_transform([pred_enc])[0]
    proba      = pipeline.predict_proba(X_row)[0]
    class_proba = dict(zip(le.classes_, proba))

    print(f"\n{'='*40}")
    print(f"  Urgency Level : {pred_label.upper()}")
    print(f"  Transcript    : {textual['transcript']}")
    print(f"  Probabilities :")
    for cls in ["High", "Medium", "Low"]:
        p = class_proba.get(cls, 0.0)
        bar = "█" * int(p * 20)
        print(f"    {cls:8s} {p:.2f}  {bar}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 05_predict.py <audio_file.ogg>")
        sys.exit(1)
    main(sys.argv[1])
