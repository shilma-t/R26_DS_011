"""
augmentation_prototype.py
SMALL PROTOTYPE (25 calls, not the full 267) - before committing to a full
augmentation + re-extraction run (potentially 1-3+ hours), check whether
adding noise/telephone-style degradation to the CLEAN simulated audio
actually moves its acoustic feature distribution closer to DMC's real
distribution, using the domain-shift diagnostic (Cohen's d) as the signal.

No model training here - pure feature-distribution comparison. If this
doesn't show |d| shrinking on average, the full augmentation run isn't
worth the time investment.

Method:
  1. Take 25 calls from audio/clean/ (already denoised+trimmed - the exact
     audio 02_extract_features.py reads for the real training corpus).
  2. For each, extract acoustic features from the ORIGINAL clean audio
     (baseline) and from an AUGMENTED version (white noise added at a
     moderate SNR, using the frozen extract_acoustic() function so features
     are computed identically to the real pipeline).
  3. Compare |Cohen's d| (simulated vs DMC, all 59 DMC calls) for each of
     the 91 acoustic features, before vs after augmentation.
  4. Report mean |d| reduction - the go/no-go signal for the full run.
"""
import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import librosa

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
PIPE = os.path.join(ROOT, "scripts", "pipeline")
N_PROTOTYPE = 25
TARGET_SR = 16000
SNR_DB = 15  # moderate noise level, typical of degraded telephone audio

ACOUSTIC = ([f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")] +
    ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
     "spec_centroid_mean", "speaking_rate", "jitter",
     "rms_slope", "rms_gradient", "speaking_rate_change"])


def load_frozen_extractor():
    path = os.path.join(PIPE, "02_extract_features.py")
    spec = importlib.util.spec_from_file_location("frozen_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled_std


def add_noise_at_snr(y, snr_db, rng):
    signal_power = np.mean(y ** 2)
    noise = rng.normal(0, 1, len(y))
    noise_power = np.mean(noise ** 2)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / max(noise_power, 1e-12))
    return y + noise


def main():
    frozen = load_frozen_extractor()
    rng = np.random.default_rng(42)

    sim_meta = pd.read_csv(f"{ROOT}\\data\\features.csv")
    clean_dir = f"{ROOT}\\audio\\clean"
    all_files = sorted(os.listdir(clean_dir))
    prototype_files = all_files[:N_PROTOTYPE]
    print(f"Prototype sample: {len(prototype_files)} calls from audio/clean/")

    print("\nExtracting ORIGINAL (clean) acoustic features ...")
    orig_rows = []
    aug_rows = []
    for i, fname in enumerate(prototype_files):
        path = os.path.join(clean_dir, fname)
        y, sr = librosa.load(path, sr=TARGET_SR)

        feats_orig = frozen.extract_acoustic(y, sr)
        orig_rows.append(feats_orig)

        y_aug = add_noise_at_snr(y, SNR_DB, rng)
        feats_aug = frozen.extract_acoustic(y_aug, sr)
        aug_rows.append(feats_aug)

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(prototype_files)} done")

    orig_df = pd.DataFrame(orig_rows)
    aug_df = pd.DataFrame(aug_rows)

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")

    print("\n" + "=" * 78)
    print(f"Domain shift (|Cohen's d|, sim-prototype[{N_PROTOTYPE}] vs DMC[59]):")
    print(f"BEFORE vs AFTER adding white noise at SNR={SNR_DB}dB")
    print("=" * 78)
    rows = []
    for col in ACOUSTIC:
        d_before = abs(cohens_d(orig_df[col].values, dmc[col].fillna(0).values))
        d_after = abs(cohens_d(aug_df[col].values, dmc[col].fillna(0).values))
        rows.append({"feature": col, "d_before": d_before, "d_after": d_after,
                    "improved": d_after < d_before})
    shift_df = pd.DataFrame(rows)

    print(f"\nMean |d| before augmentation: {shift_df['d_before'].mean():.4f}")
    print(f"Mean |d| after augmentation : {shift_df['d_after'].mean():.4f}")
    print(f"Features where |d| improved (moved closer to DMC): "
          f"{shift_df['improved'].sum()}/{len(shift_df)}")

    print("\nTop 15 features by |d| improvement (d_before - d_after, positive = better):")
    shift_df["delta"] = shift_df["d_before"] - shift_df["d_after"]
    print(shift_df.sort_values("delta", ascending=False).head(15)
          [["feature", "d_before", "d_after", "delta"]].round(3).to_string(index=False))

    print("\nTop 10 features that got WORSE (delta most negative):")
    print(shift_df.sort_values("delta", ascending=True).head(10)
          [["feature", "d_before", "d_after", "delta"]].round(3).to_string(index=False))

    print("\n" + "=" * 78)
    print("GO/NO-GO SIGNAL")
    print("=" * 78)
    mean_before = shift_df["d_before"].mean()
    mean_after = shift_df["d_after"].mean()
    pct_improved = 100 * shift_df["improved"].sum() / len(shift_df)
    if mean_after < mean_before and pct_improved > 50:
        print(f"  POSITIVE SIGNAL: mean |d| dropped {mean_before:.3f} -> {mean_after:.3f} "
              f"({pct_improved:.0f}% of features improved). Worth running the full 267-call "
              f"augmentation + re-extraction + retrain.")
    else:
        print(f"  WEAK/NO SIGNAL: mean |d| {mean_before:.3f} -> {mean_after:.3f} "
              f"({pct_improved:.0f}% of features improved). Simple white-noise augmentation "
              f"alone does not clearly close the domain gap - NOT recommended to scale up "
              f"as-is. Consider a different augmentation (bandpass/telephone-channel sim, "
              f"reverb, or real DMC background noise clips) before committing the full run.")


if __name__ == "__main__":
    main()
