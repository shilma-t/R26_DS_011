"""
bandpass_prototype.py
Follow-up to augmentation_prototype.py: white noise made the domain gap
WORSE (mean |d| 0.80 -> 1.65), because it added broadband high-frequency
energy that DMC's real profile doesn't show. The failure pattern (zcr_mean,
spec_centroid_mean, and high-index MFCC stds all overshot far past DMC)
suggests the real degradation is band-limiting (telephone/VOIP channel),
not noise. This tests a 300-3400 Hz bandpass filter instead - same 25-call
prototype, same go/no-go signal (mean |Cohen's d| before vs after).
"""
import os
import importlib.util
import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, sosfiltfilt

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
PIPE = os.path.join(ROOT, "scripts", "pipeline")
N_PROTOTYPE = 25
TARGET_SR = 16000
LOWCUT, HIGHCUT = 300, 3400  # standard telephone voice band

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


def telephone_bandpass(y, sr, lowcut=LOWCUT, highcut=HIGHCUT, order=4):
    nyq = sr / 2
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, y)


def main():
    frozen = load_frozen_extractor()

    clean_dir = f"{ROOT}\\audio\\clean"
    all_files = sorted(os.listdir(clean_dir))
    prototype_files = all_files[:N_PROTOTYPE]
    print(f"Prototype sample: {len(prototype_files)} calls from audio/clean/")
    print(f"Bandpass: {LOWCUT}-{HIGHCUT} Hz (telephone voice band)")

    orig_rows, aug_rows = [], []
    for i, fname in enumerate(prototype_files):
        path = os.path.join(clean_dir, fname)
        y, sr = librosa.load(path, sr=TARGET_SR)

        feats_orig = frozen.extract_acoustic(y, sr)
        orig_rows.append(feats_orig)

        y_bp = telephone_bandpass(y, sr).astype(np.float32)
        feats_bp = frozen.extract_acoustic(y_bp, sr)
        aug_rows.append(feats_bp)

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(prototype_files)} done")

    orig_df = pd.DataFrame(orig_rows)
    aug_df = pd.DataFrame(aug_rows)
    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")

    print("\n" + "=" * 78)
    print(f"Domain shift (|Cohen's d|, sim-prototype[{N_PROTOTYPE}] vs DMC[59]):")
    print(f"BEFORE vs AFTER {LOWCUT}-{HIGHCUT}Hz bandpass")
    print("=" * 78)
    rows = []
    for col in ACOUSTIC:
        d_before = abs(cohens_d(orig_df[col].values, dmc[col].fillna(0).values))
        d_after = abs(cohens_d(aug_df[col].values, dmc[col].fillna(0).values))
        rows.append({"feature": col, "d_before": d_before, "d_after": d_after,
                    "improved": d_after < d_before})
    shift_df = pd.DataFrame(rows)
    shift_df["delta"] = shift_df["d_before"] - shift_df["d_after"]

    print(f"\nMean |d| before bandpass: {shift_df['d_before'].mean():.4f}")
    print(f"Mean |d| after bandpass : {shift_df['d_after'].mean():.4f}")
    print(f"Features improved: {shift_df['improved'].sum()}/{len(shift_df)}")

    print("\nTop 15 features by improvement:")
    print(shift_df.sort_values("delta", ascending=False).head(15)
          [["feature", "d_before", "d_after", "delta"]].round(3).to_string(index=False))

    print("\nTop 10 features that got WORSE:")
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
              f"bandpass augmentation + re-extraction + retrain.")
    else:
        print(f"  WEAK/NO SIGNAL: mean |d| {mean_before:.3f} -> {mean_after:.3f} "
              f"({pct_improved:.0f}% of features improved). Bandpass filtering alone does not "
              f"clearly close the domain gap either - NOT recommended to scale up as-is.")


if __name__ == "__main__":
    main()
