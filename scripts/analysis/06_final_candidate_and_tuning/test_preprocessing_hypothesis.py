"""
test_preprocessing_hypothesis.py
Test whether the v1 DMC failure is a preprocessing mismatch rather than
genuine domain shift.

The 267 training calls were denoised and silence-trimmed by 01_preprocess.py
before feature extraction. The DMC caller audio was not - it went from
diarization straight into extract_acoustic(). If that mismatch is the cause,
applying the same preprocessing to DMC audio should move the most-shifted
features back toward the training distribution.

Runs on a sample of calls only: this is a diagnostic, not the fix.

Usage:
  python scripts/analysis/test_preprocessing_hypothesis.py [--n 12]
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import librosa

_ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
PIPE = os.path.join(_ROOT, "scripts", "pipeline")
if PIPE not in sys.path:
    sys.path.insert(0, PIPE)

import importlib.util

CALLER_DIR = os.path.join(_ROOT, "audio", "dmc_caller")
FEATURES = os.path.join(_ROOT, "data", "features.csv")
TARGET_SR = 16_000

# The features the diagnostic flagged as most shifted
WATCH = ["rms_mean", "rms_std", "mfcc_1_mean", "mfcc_2_mean", "mfcc_3_mean",
         "mfcc_2_std", "mfcc_4_std", "mfcc_6_std", "mfcc_8_std", "mfcc_9_std",
         "speaking_rate", "zcr_mean", "spec_centroid_mean"]


def frozen_module():
    path = os.path.join(PIPE, "02_extract_features.py")
    spec = importlib.util.spec_from_file_location("frozen_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preprocess_like_training(y, sr):
    """Exactly what 01_preprocess.py does, minus the resample (already 16k)."""
    import noisereduce as nr
    y_nr = nr.reduce_noise(y=y, sr=sr)
    if sr != TARGET_SR:
        y_nr = librosa.resample(y_nr, orig_sr=sr, target_sr=TARGET_SR)
    y_tr, _ = librosa.effects.trim(y_nr, top_db=20)
    return y_tr


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                     / max(na + nb - 2, 1))
    return (b.mean() - a.mean()) / pooled if pooled else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12,
                        help="How many DMC calls to sample")
    args = parser.parse_args()

    frozen = frozen_module()
    sim = pd.read_csv(FEATURES)

    files = sorted(f for f in os.listdir(CALLER_DIR) if f.endswith(".wav"))
    step = max(1, len(files) // args.n)
    sample = files[::step][:args.n]

    print("=" * 72)
    print("PREPROCESSING HYPOTHESIS TEST")
    print("=" * 72)
    print(f"training corpus : {len(sim)} calls (denoised + trimmed)")
    print(f"DMC sample      : {len(sample)} calls")
    print("\nextracting features raw vs preprocessed ...")

    raw_rows, prep_rows = [], []
    for i, fname in enumerate(sample, 1):
        path = os.path.join(CALLER_DIR, fname)
        y, sr = librosa.load(path, sr=TARGET_SR)
        if y.size < TARGET_SR:
            continue

        raw_rows.append(frozen.extract_acoustic(y, TARGET_SR))

        y_prep = preprocess_like_training(y, TARGET_SR)
        if y_prep.size < TARGET_SR // 2:
            y_prep = y
        prep_rows.append(frozen.extract_acoustic(y_prep, TARGET_SR))

        print(f"  [{i}/{len(sample)}] {fname[:44]:<44} "
              f"{len(y)/TARGET_SR:5.1f}s -> {len(y_prep)/TARGET_SR:5.1f}s")

    raw = pd.DataFrame(raw_rows)
    prep = pd.DataFrame(prep_rows)

    print("\n" + "-" * 72)
    print("DISTANCE FROM TRAINING DISTRIBUTION  (|Cohen's d| vs the 267 calls)")
    print("-" * 72)
    print(f"{'feature':<22} {'train':>10} {'DMC raw':>10} {'DMC prep':>10} "
          f"{'|d| raw':>8} {'|d| prep':>9}  verdict")

    improved = worsened = 0
    for name in WATCH:
        if name not in sim.columns or name not in raw.columns:
            continue
        d_raw = abs(cohens_d(sim[name], raw[name]))
        d_prep = abs(cohens_d(sim[name], prep[name]))
        if d_prep < d_raw * 0.7:
            verdict, improved = "BETTER", improved + 1
        elif d_prep > d_raw * 1.3:
            verdict, worsened = "worse", worsened + 1
        else:
            verdict = "~same"
        print(f"{name:<22} {sim[name].mean():>10.3f} {raw[name].mean():>10.3f} "
              f"{prep[name].mean():>10.3f} {d_raw:>8.2f} {d_prep:>9.2f}  {verdict}")

    print("\n" + "-" * 72)
    print(f"features materially closer to training after preprocessing: "
          f"{improved} of {len(WATCH)}")
    print(f"features materially further:                               {worsened}")
    print("-" * 72)
    if improved >= len(WATCH) // 2:
        print("\nHYPOTHESIS SUPPORTED.")
        print("The v1 DMC evaluation compared denoised, trimmed training audio")
        print("against raw test audio. That mismatch - not domain shift - is a")
        print("substantial part of the failure. The DMC pipeline must apply")
        print("01_preprocess.py's steps before feature extraction, and the")
        print("evaluation must be re-run before the 0.2524 result means anything.")
    else:
        print("\nHYPOTHESIS NOT SUPPORTED.")
        print("Preprocessing does not close the gap; the shift is genuine")
        print("(telephony bandwidth and recording conditions).")


if __name__ == "__main__":
    main()
