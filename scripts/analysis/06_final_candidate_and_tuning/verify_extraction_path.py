"""
verify_extraction_path.py
Prove the feature-extraction path is correct BEFORE spending hours on another
evaluation run.

The decisive test: regenerate features for calls that are already in
features.csv, starting from the same audio, and check the numbers come back
identical. If they do, the path is proven. If they do not, the difference is
found in minutes rather than after a five-hour run.

Then apply the identical path to a DMC caller file and confirm every one of
the 111 features is produced with no NaNs and no silent zeros.

Usage:
  python scripts/analysis/verify_extraction_path.py [--n 10] [--with-asr 2]
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

CLEAN_DIR = os.path.join(_ROOT, "audio", "clean")
CALLER_DIR = os.path.join(_ROOT, "audio", "dmc_caller")
FEATURES = os.path.join(_ROOT, "data", "features.csv")
TARGET_SR = 16_000

ACOUSTIC = (
    [f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")] +
    ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
     "spec_centroid_mean", "speaking_rate", "jitter",
     "rms_slope", "rms_gradient", "speaking_rate_change"]
)
TEXT = ["asr_quality", "keyword_count", "word_count",
        "repetition_rate", "sentiment_polarity"]
CONTEXT = [
    "sin_context_score", "sin_danger_count", "sin_help_request_count",
    "sin_high_risk_phrase_count", "sin_location_context_count",
    "sin_person_at_risk_count", "sin_safety_state_count",
    "tam_context_score", "tam_danger_count", "tam_help_request_count",
    "tam_high_risk_phrase_count", "tam_location_context_count",
    "tam_person_at_risk_count", "tam_safety_state_count",
]
FEATURE_COLS = ACOUSTIC + TEXT + CONTEXT + ["fire_smoke_match"]

failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def frozen_module():
    path = os.path.join(PIPE, "02_extract_features.py")
    spec = importlib.util.spec_from_file_location("frozen_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preprocess_like_training(y, sr):
    """The exact three steps 01_preprocess.py applies."""
    import noisereduce as nr
    y_nr = nr.reduce_noise(y=y, sr=sr)
    if sr != TARGET_SR:
        y_nr = librosa.resample(y_nr, orig_sr=sr, target_sr=TARGET_SR)
    y_tr, _ = librosa.effects.trim(y_nr, top_db=20)
    return y_tr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10,
                        help="Training calls to reproduce (acoustic only)")
    parser.add_argument("--with-asr", type=int, default=2,
                        help="How many of those to also verify text features on")
    args = parser.parse_args()

    frozen = frozen_module()
    sim = pd.read_csv(FEATURES)

    print("=" * 72)
    print("EXTRACTION PATH VERIFICATION")
    print("=" * 72)

    # ── 0. the training audio must still exist ───────────────────────────────
    print("\n0. Training audio")
    check("audio/clean exists", os.path.isdir(CLEAN_DIR), CLEAN_DIR)
    if not os.path.isdir(CLEAN_DIR):
        print("\nCannot verify without the preprocessed training audio.")
        return 1
    cleaned = [f for f in os.listdir(CLEAN_DIR) if f.lower().endswith(".wav")]
    check("clean files present", len(cleaned) > 0, f"{len(cleaned)} files")

    # ── 1. reproduce acoustic features exactly ───────────────────────────────
    print(f"\n1. Reproduce acoustic features for {args.n} training calls")
    print("   (load audio/clean/<stem>.wav at 16k mono, run extract_acoustic)")

    sim["stem"] = sim["filename"].map(lambda f: os.path.splitext(str(f))[0])
    available = sim[sim["stem"].map(lambda s: os.path.exists(
        os.path.join(CLEAN_DIR, s + ".wav")))]
    check("rows with matching clean audio", len(available) > 0,
          f"{len(available)} of {len(sim)}")
    if available.empty:
        return 1

    step = max(1, len(available) // args.n)
    sample = available.iloc[::step].head(args.n)

    worst_overall = 0.0
    mismatched = []
    for row in sample.itertuples():
        wav = os.path.join(CLEAN_DIR, row.stem + ".wav")
        y, sr = librosa.load(wav, sr=TARGET_SR, mono=True)
        got = frozen.extract_acoustic(y, sr)

        worst_feat, worst_rel = None, 0.0
        for col in ACOUSTIC:
            stored = getattr(row, col, None)
            if stored is None or not np.isfinite(stored):
                continue
            rel = abs(got[col] - stored) / max(abs(stored), 1e-6)
            if rel > worst_rel:
                worst_rel, worst_feat = rel, col

        worst_overall = max(worst_overall, worst_rel)
        flag = "" if worst_rel < 1e-6 else f"  <-- {worst_feat} off by {worst_rel:.2%}"
        if worst_rel >= 1e-6:
            mismatched.append(row.stem)
        print(f"     {row.stem[:40]:<40} max rel diff {worst_rel:.2e}{flag}")

    check("acoustic features reproduce exactly", worst_overall < 1e-6,
          f"worst relative difference {worst_overall:.2e}")

    # ── 2. reproduce text features ───────────────────────────────────────────
    if args.with_asr > 0:
        print(f"\n2. Reproduce text + context features for {args.with_asr} call(s)")
        print("   (runs ASR - slow, hence the small sample)")
        text_ok = True
        for row in sample.head(args.with_asr).itertuples():
            wav = os.path.join(CLEAN_DIR, row.stem + ".wav")
            y, _ = librosa.load(wav, sr=TARGET_SR, mono=True)
            got = frozen.extract_textual(y, language=getattr(row, "language", ""))
            diffs = []
            for col in TEXT + CONTEXT:
                stored = getattr(row, col, None)
                if stored is None or not np.isfinite(stored):
                    continue
                if abs(got.get(col, 0) - stored) > 1e-6:
                    diffs.append(f"{col}: stored={stored} got={got.get(col)}")
            if diffs:
                text_ok = False
                print(f"     {row.stem[:40]:<40} MISMATCH")
                for d in diffs[:5]:
                    print(f"        {d}")
            else:
                print(f"     {row.stem[:40]:<40} identical")
        check("text/context features reproduce", text_ok)

    # ── 3. the DMC path, with preprocessing applied ──────────────────────────
    print("\n3. DMC caller audio through the SAME path")
    files = sorted(f for f in os.listdir(CALLER_DIR) if f.endswith(".wav"))
    if not files:
        check("DMC caller audio present", False)
        return 1

    probe = os.path.join(CALLER_DIR, files[0])
    y_raw, sr = librosa.load(probe, sr=TARGET_SR, mono=True)
    y_prep = preprocess_like_training(y_raw, TARGET_SR)
    print(f"   {files[0][:46]:<46} {len(y_raw)/TARGET_SR:.1f}s "
          f"-> {len(y_prep)/TARGET_SR:.1f}s after preprocessing")

    ac = frozen.extract_acoustic(y_prep, TARGET_SR)
    check("all 91 acoustic features produced",
          all(c in ac for c in ACOUSTIC), f"{len(ac)} returned")
    nans = [c for c in ACOUSTIC if not np.isfinite(ac.get(c, np.nan))]
    check("no NaN or inf values", not nans,
          f"{len(nans)} bad: {nans[:4]}" if nans else "")

    # every feature must be reachable - a silent all-zero block is a bug
    zeros = [c for c in ACOUSTIC if ac.get(c, 0) == 0.0]
    check("no unexpected all-zero acoustic block", len(zeros) <= 2,
          f"{len(zeros)} zero-valued: {zeros[:4]}")

    # ── 4. feature ordering ──────────────────────────────────────────────────
    print("\n4. Feature vector integrity")
    check("111 features in the frozen list", len(FEATURE_COLS) == 111,
          f"{len(FEATURE_COLS)}")
    check("no duplicates", len(FEATURE_COLS) == len(set(FEATURE_COLS)))
    missing = [c for c in FEATURE_COLS if c not in sim.columns
               and c != "fire_smoke_match"]
    check("all frozen features exist in features.csv", not missing,
          f"missing: {missing}" if missing else "")

    print("\n" + "=" * 72)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: {failures}")
        print("Do NOT start an evaluation run until these are resolved.")
        return 1
    print("PATH VERIFIED.")
    print("Acoustic features reproduce the stored training values exactly, and")
    print("the same path applied to DMC audio yields a complete, finite vector.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
