"""
02_extract_features.py
Extracts acoustic + textual features from each cleaned audio file
and saves the combined feature matrix to features.csv.
"""

import os
import warnings
import numpy as np
import pandas as pd
import librosa
import whisper

warnings.filterwarnings("ignore")

CLEAN_DIR   = os.path.join("..", "audio", "clean")
LABELS      = os.path.join("..", "labels.csv")
OUTPUT_CSV  = os.path.join("..", "features.csv")
TARGET_SR   = 16_000
N_MFCC      = 40

URGENCY_KEYWORDS = [
    # English
    "help", "emergency", "fire", "accident", "blood", "dying", "dead",
    "trapped", "attack", "knife", "gun", "hurt", "injured", "crash",
    "please", "ambulance", "police", "quickly", "fast", "now",
    # Sinhala romanised
    "udaw", "hadisi", "ape", "maru", "wedi", "gini",
    # Tamil romanised
    "உதவி", "உதவுங்கள்", "தீ", "விபத்து", "இரத்தம்",
]

_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("  Loading Whisper model (base) ...")
        _whisper_model = whisper.load_model("base")
    return _whisper_model


# ── Acoustic features ────────────────────────────────────────────────────────

def extract_acoustic(y: np.ndarray, sr: int) -> dict:
    feats = {}

    # MFCCs (40 coefficients — mean + std each = 80 values)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1}_std"]  = float(np.std(mfcc[i]))

    # Pitch (F0) — mean & std of voiced frames
    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
    voiced_f0 = f0[voiced] if voiced is not None and voiced.any() else np.array([0.0])
    feats["pitch_mean"] = float(np.nanmean(voiced_f0))
    feats["pitch_std"]  = float(np.nanstd(voiced_f0))

    # Energy / RMS
    rms = librosa.feature.rms(y=y)[0]
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"]  = float(np.std(rms))

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats["zcr_mean"] = float(np.mean(zcr))

    # Spectral centroid
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats["spec_centroid_mean"] = float(np.mean(sc))

    # Speaking rate — proxy: voiced frames / total duration
    hop = 512
    total_frames = len(y) / hop
    voiced_frames = int(np.sum(voiced)) if voiced is not None else 0
    feats["speaking_rate"] = voiced_frames / max(total_frames, 1)

    # Jitter (mean absolute diff of F0 between consecutive voiced frames)
    if voiced_f0.size > 1:
        feats["jitter"] = float(np.mean(np.abs(np.diff(voiced_f0))))
    else:
        feats["jitter"] = 0.0

    return feats


# ── Textual features ─────────────────────────────────────────────────────────

def extract_textual(wav_path: str) -> dict:
    model = get_whisper_model()
    result = model.transcribe(wav_path, task="transcribe")
    text   = result.get("text", "").lower()

    words = text.split()
    feats = {}

    feats["keyword_count"] = sum(kw.lower() in text for kw in URGENCY_KEYWORDS)
    feats["word_count"]    = len(words)

    from collections import Counter
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    df_labels = pd.read_csv(LABELS)
    rows = []

    for _, row in df_labels.iterrows():
        fname = row["filename"]
        stem  = os.path.splitext(fname)[0]
        wav   = os.path.join(CLEAN_DIR, stem + ".wav")

        if not os.path.exists(wav):
            print(f"  [SKIP — no clean file] {fname}")
            continue

        print(f"  Processing {fname} ...")
        y, sr = librosa.load(wav, sr=TARGET_SR, mono=True)

        acoustic = extract_acoustic(y, sr)
        textual  = extract_textual(wav)

        combined = {
            "filename":      fname,
            "urgency_label": row["urgency_label"],
            "language":      row.get("language", ""),
            **acoustic,
            **textual,
        }
        rows.append(combined)

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
