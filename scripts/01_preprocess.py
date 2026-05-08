"""
01_preprocess.py
Loads each .ogg file, applies noise reduction, resamples to 16 kHz,
trims leading/trailing silence, and saves a clean .wav to audio/clean/.
"""

import os
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
import pandas as pd

RAW_DIR   = os.path.join("..", "audio")
CLEAN_DIR = os.path.join("..", "audio", "clean")
LABELS    = os.path.join("..", "labels.csv")
TARGET_SR = 16_000


def preprocess_file(src_path: str, dst_path: str) -> None:
    y, sr = librosa.load(src_path, sr=None, mono=True)
    y_nr  = nr.reduce_noise(y=y, sr=sr)
    y_rs  = librosa.resample(y_nr, orig_sr=sr, target_sr=TARGET_SR)
    y_tr, _ = librosa.effects.trim(y_rs, top_db=20)
    sf.write(dst_path, y_tr, TARGET_SR)


def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    df = pd.read_csv(LABELS)

    skipped = []
    for fname in df["filename"]:
        src = os.path.join(RAW_DIR, fname)
        stem = os.path.splitext(fname)[0]
        dst  = os.path.join(CLEAN_DIR, stem + ".wav")

        if not os.path.exists(src):
            print(f"  [MISSING] {fname}")
            skipped.append(fname)
            continue

        preprocess_file(src, dst)
        print(f"  [OK] {fname} -> {os.path.basename(dst)}")

    if skipped:
        print(f"\n{len(skipped)} file(s) missing from audio/: {skipped}")
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
