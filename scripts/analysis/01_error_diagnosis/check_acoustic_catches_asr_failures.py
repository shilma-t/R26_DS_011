"""
check_acoustic_catches_asr_failures.py
Direct follow-up: does the acoustic-only model correctly call the 5
consistently-misclassified (text-only High->Low) DMC calls "High", where
Condition C (text-only) fails because their transcripts are empty or
ASR-hallucinated garbage with zero keyword/context signal?

If yes, this is concrete evidence for the supervisor's "class-specific /
fallback multimodal" idea: text-only for most High calls, acoustic as a
fallback specifically when text signal is absent/broken.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
RF_BASE = dict(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1)
SEEDS = [1, 7, 42, 99, 123]

ACOUSTIC = ([f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")] +
    ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
     "spec_centroid_mean", "speaking_rate", "jitter",
     "rms_slope", "rms_gradient", "speaking_rate_change"])

TARGET_CALLS = [
    "Call recording tamil",
    "Call recording 7251128_172437",
    "Call recording 7_251128_092626",
    "Call recording _251129_092106",
    "Call recording _251128_055921",
]


def fire_smoke_match(text):
    kws = ["fire","smoke","burning","flames","gas","ගිනිගන්නව","ගිනිඅරගෙන",
           "ගින්නක්","ගිනි","දුමක්","දුම","தீ","புகை","gini"]
    if not isinstance(text, str): return 0
    t = text.lower()
    return int(any(k.lower() in t for k in kws))


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim = sim.copy()
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    X_sim = sim[ACOUSTIC].fillna(0).values
    y_sim = le.transform(sim["urgency_label"].values)

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    X_dmc = dmc[ACOUSTIC].fillna(0).values
    dmc_reset = dmc.reset_index(drop=True)
    classes = list(le.classes_)
    high_idx = classes.index("High")

    print("Training ACOUSTIC-ONLY (91 feat) at 5 seeds, predicting on DMC ...")
    all_preds = []
    all_probas = []
    for seed in SEEDS:
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=seed)),
            ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
        ])
        pipe.fit(X_sim, y_sim)
        proba = pipe.predict_proba(X_dmc)
        all_preds.append(proba.argmax(axis=1))
        all_probas.append(proba)
    all_preds = np.array(all_preds)
    all_probas = np.array(all_probas)

    id_to_pos = {rid: i for i, rid in enumerate(dmc_reset["recording_id"])}

    print("\n" + "=" * 78)
    print("Acoustic-only predictions on the 5 text-only High->Low failure calls")
    print("=" * 78)
    caught = 0
    for rid in TARGET_CALLS:
        if rid not in id_to_pos:
            print(f"  {rid}: NOT FOUND in DMC")
            continue
        pos = id_to_pos[rid]
        preds_this_call = [classes[all_preds[s, pos]] for s in range(len(SEEDS))]
        mean_proba = all_probas[:, pos, :].mean(axis=0)
        times_high = sum(1 for p in preds_this_call if p == "High")
        if times_high >= 3:
            caught += 1
        print(f"\n  {rid}")
        print(f"    predictions across 5 seeds: {preds_this_call}  "
              f"(called High in {times_high}/5)")
        print(f"    mean_proba: High={mean_proba[high_idx]:.3f}  "
              f"Low={mean_proba[classes.index('Low')]:.3f}  "
              f"Medium={mean_proba[classes.index('Medium')]:.3f}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  {caught}/{len(TARGET_CALLS)} of the text-only ASR-failure calls are "
          f"caught as High (>=3/5 seeds) by the acoustic-only model")
    if caught > 0:
        print("  -> Supports the class-specific/fallback multimodal idea: acoustic "
              "can recover calls where text signal is absent/broken.")
    else:
        print("  -> Acoustic-only does NOT rescue these specific calls either - "
              "the fallback idea doesn't hold for this exact failure mode, at "
              "least not with the current acoustic feature set.")


if __name__ == "__main__":
    main()
