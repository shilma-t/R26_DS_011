"""
gemaps_shift_prototype.py
SMALL PROTOTYPE (20 simulated calls, not the full 267) - before extracting
eGeMAPS for the whole dataset, check whether GeMAPS features show a smaller
domain shift (simulated vs DMC) than the current custom librosa-based
acoustic features, using the same Cohen's-d diagnostic used throughout
tonight.

Uses eGeMAPSv02 (88 functionals) via the opensmile Python wrapper - the
standard, well-validated affective-speech feature set named in the PP1
proposal (openSMILE + GeMAPS).

Go/no-go:
  - PROMISING -> extract eGeMAPS for the full 267 simulated + 59 DMC calls,
    train acoustic-only, combine with Condition C, evaluate on DMC.
  - BAD -> stop; changing the acoustic representation doesn't fix the
    domain shift, and the acoustic subtraction-of-value finding stands.
"""
import os
import numpy as np
import pandas as pd
import opensmile

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
N_PROTOTYPE = 20

# The custom acoustic features already tested tonight, for a side-by-side
# "mean |d|" comparison against GeMAPS on the SAME prototype calls.
CUSTOM_ACOUSTIC = ([f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")] +
    ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
     "spec_centroid_mean", "speaking_rate", "jitter",
     "rms_slope", "rms_gradient", "speaking_rate_change"])


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if pooled_std == 0 or np.isnan(pooled_std):
        return 0.0
    return (a.mean() - b.mean()) / pooled_std


def main():
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    print(f"eGeMAPSv02 functionals: {len(smile.feature_names)} features")

    clean_dir = f"{ROOT}\\audio\\clean"
    all_sim_files = sorted(os.listdir(clean_dir))
    prototype_sim_files = all_sim_files[:N_PROTOTYPE]

    dmc_dir = os.path.join(ROOT, "audio", "dmc_caller")
    print(f"Using DMC audio dir: audio\\dmc_caller ({len(os.listdir(dmc_dir))} files)")

    dmc_meta = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")

    print(f"\nExtracting eGeMAPS for {len(prototype_sim_files)} simulated calls ...")
    sim_rows = []
    for i, fname in enumerate(prototype_sim_files):
        path = os.path.join(clean_dir, fname)
        feats = smile.process_file(path)
        sim_rows.append(feats.iloc[0].to_dict())
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(prototype_sim_files)} done")
    sim_gemaps = pd.DataFrame(sim_rows)

    print(f"\nExtracting eGeMAPS for DMC calls (matching dmc_evaluation.csv filenames) ...")
    dmc_files_available = set(os.listdir(dmc_dir))
    dmc_rows = []
    matched_ids = []
    for i, row in dmc_meta.iterrows():
        base = str(row["recording_id"])
        candidate_names = [row["filename"], base + ".wav", base + "_caller.wav"]
        found = None
        for cand in candidate_names:
            if cand in dmc_files_available:
                found = cand
                break
        if found is None:
            matches = [f for f in dmc_files_available if base in f]
            if matches:
                found = matches[0]
        if found is None:
            continue
        path = os.path.join(dmc_dir, found)
        feats = smile.process_file(path)
        dmc_rows.append(feats.iloc[0].to_dict())
        matched_ids.append(row["recording_id"])
        if len(dmc_rows) % 10 == 0:
            print(f"  {len(dmc_rows)} DMC calls done")
    dmc_gemaps = pd.DataFrame(dmc_rows)
    print(f"\nMatched {len(dmc_gemaps)}/{len(dmc_meta)} DMC calls to audio files in {dmc_dir}")

    if len(dmc_gemaps) < 10:
        print("TOO FEW DMC MATCHES - aborting comparison, check dmc_dir/filename mapping")
        return

    print("\n" + "=" * 78)
    print(f"GeMAPS domain shift (|Cohen's d|, sim-prototype[{len(sim_gemaps)}] vs "
          f"DMC[{len(dmc_gemaps)}])")
    print("=" * 78)
    gemaps_features = [c for c in sim_gemaps.columns if c in dmc_gemaps.columns]
    shift_rows = []
    for col in gemaps_features:
        d = abs(cohens_d(sim_gemaps[col].values, dmc_gemaps[col].values))
        shift_rows.append({"feature": col, "abs_d": d})
    gemaps_shift = pd.DataFrame(shift_rows).sort_values("abs_d", ascending=False)
    print(gemaps_shift.head(20).round(3).to_string(index=False))
    print(f"\nMean |d| across {len(gemaps_features)} GeMAPS features: {gemaps_shift['abs_d'].mean():.4f}")
    print(f"Median |d|: {gemaps_shift['abs_d'].median():.4f}")
    print(f"Features with |d| > 0.8 (large shift): {(gemaps_shift['abs_d'] > 0.8).sum()}/{len(gemaps_shift)}")

    print("\n" + "=" * 78)
    print(f"COMPARISON: current custom acoustic features, SAME {len(prototype_sim_files)} "
          f"sim calls vs SAME {len(dmc_gemaps)} DMC calls")
    print("=" * 78)
    sim_full = pd.read_csv(f"{ROOT}\\data\\features.csv")
    prototype_basenames = [f.replace(".wav", "") for f in prototype_sim_files]
    sim_full_matched = sim_full[sim_full["filename"].str.replace(".ogg", "", regex=False).isin(prototype_basenames)]
    dmc_full_matched = dmc_meta[dmc_meta["recording_id"].isin(matched_ids)]

    custom_shift_rows = []
    for col in CUSTOM_ACOUSTIC:
        d = abs(cohens_d(sim_full_matched[col].fillna(0).values, dmc_full_matched[col].fillna(0).values))
        custom_shift_rows.append({"feature": col, "abs_d": d})
    custom_shift = pd.DataFrame(custom_shift_rows)
    print(f"Mean |d| across {len(CUSTOM_ACOUSTIC)} CUSTOM acoustic features: {custom_shift['abs_d'].mean():.4f}")
    print(f"Median |d|: {custom_shift['abs_d'].median():.4f}")
    print(f"Features with |d| > 0.8: {(custom_shift['abs_d'] > 0.8).sum()}/{len(custom_shift)}")

    print("\n" + "=" * 78)
    print("GO/NO-GO SIGNAL")
    print("=" * 78)
    gemaps_mean = gemaps_shift["abs_d"].mean()
    custom_mean = custom_shift["abs_d"].mean()
    print(f"  GeMAPS mean |d| = {gemaps_mean:.4f}")
    print(f"  Custom mean |d| = {custom_mean:.4f}")
    if gemaps_mean < custom_mean * 0.8:
        print(f"  PROMISING: GeMAPS shows meaningfully lower domain shift "
              f"({gemaps_mean:.3f} vs {custom_mean:.3f}, {100*(1-gemaps_mean/custom_mean):.0f}% reduction). "
              f"Worth extracting for the full 267+59 dataset and testing acoustic-only + combined with Condition C.")
    elif gemaps_mean < custom_mean:
        print(f"  MARGINAL: GeMAPS is somewhat lower ({gemaps_mean:.3f} vs {custom_mean:.3f}) "
              f"but not a large reduction - borderline case, judgment call on whether to scale up.")
    else:
        print(f"  BAD: GeMAPS shows EQUAL OR WORSE domain shift ({gemaps_mean:.3f} vs {custom_mean:.3f}). "
              f"Changing the acoustic representation does not fix the domain-shift problem - STOP here.")


if __name__ == "__main__":
    main()
