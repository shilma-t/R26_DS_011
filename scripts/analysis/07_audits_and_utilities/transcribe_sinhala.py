"""
step3a_transcribe_sinhala.py
Runs Sinhala fine-tune ASR on all 129 Sinhala calls.
Saves reports/sinhala_transcripts.csv (UTF-8) for TF-IDF.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import librosa

warnings.filterwarnings("ignore")

_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(_ROOT, "audio", "clean")
LABELS    = os.path.join(_ROOT, "labels.csv")
OUTPUT    = os.path.join(_ROOT, "reports", "sinhala_transcripts.csv")
TARGET_SR = 16_000


def main():
    df = pd.read_csv(LABELS)
    sinhala = df[df["language"] == "sinhala"].copy()
    print(f"Sinhala calls to transcribe: {len(sinhala)}")
    print(f"Class breakdown:\n{sinhala['urgency_label'].value_counts().to_string()}\n")

    from transformers import pipeline
    print("Loading Sinhala fine-tune (Lingalingeswaran/whisper-small-sinhala_v3)...")
    asr = pipeline("automatic-speech-recognition",
                   model="Lingalingeswaran/whisper-small-sinhala_v3")
    print("Model loaded.\n")

    rows = []
    total = len(sinhala)
    for i, (_, row) in enumerate(sinhala.iterrows(), 1):
        fname = row["filename"]
        label = row["urgency_label"]
        wav   = os.path.join(CLEAN_DIR, os.path.splitext(fname)[0] + ".wav")

        if not os.path.exists(wav):
            print(f"  [{i}/{total}] SKIP {fname} — no audio")
            continue

        y, _ = librosa.load(wav, sr=TARGET_SR, mono=True)
        result = asr(y.astype(np.float32))
        text   = result.get("text", "").strip()

        print(f"  [{i}/{total}] {fname} [{label}] | {text[:70]}")
        rows.append({
            "filename":      fname,
            "urgency_label": label,
            "transcript":    text,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(out)} transcripts to {OUTPUT}")
    print(f"Class breakdown in output:\n{out['urgency_label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
