"""
acoustic_abc_variants_n11_n25.py
Extends acoustic_abc_variants.py (MFCC-only / non-MFCC-only / MFCC-minus-
top10-shifted, each x standalone/early/late fusion) from the full 59-call
DMC set to the n=11 keyword-matched fire+flood subset and the n=25
authoritative flood-only subset, using the exact same ACOUSTIC-A/B/C column
definitions already established (the top-10 shifted MFCCs to drop for
ACOUSTIC-C are NOT recomputed per subset -- reusing the same fixed list keeps
this a clean test of "does the earlier subset choice generalize", not a new
variable).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
RF_BASE = dict(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1)
SEEDS = [1, 7, 42, 99, 123]

MFCC = [f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")]
NON_MFCC = ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
    "spec_centroid_mean", "speaking_rate", "jitter",
    "rms_slope", "rms_gradient", "speaking_rate_change"]
ACOUSTIC_FULL = MFCC + NON_MFCC
CONDITION_C = ["asr_quality", "keyword_count", "repetition_rate", "sentiment_polarity",
    "sin_context_score", "sin_danger_count", "sin_help_request_count",
    "sin_high_risk_phrase_count", "sin_location_context_count",
    "sin_person_at_risk_count", "sin_safety_state_count", "tam_context_score",
    "tam_danger_count", "tam_help_request_count", "tam_high_risk_phrase_count",
    "tam_location_context_count", "tam_person_at_risk_count",
    "tam_safety_state_count", "fire_smoke_match"]

# Fixed top-10 most domain-shifted MFCCs, from the full-59-DMC comparison
# already run in acoustic_abc_variants.py -- reused as-is, not recomputed.
TOP10_SHIFTED_MFCC = ["mfcc_14_mean", "mfcc_10_std", "mfcc_4_mean", "mfcc_40_std",
    "mfcc_25_std", "mfcc_3_mean", "mfcc_17_mean", "mfcc_10_mean", "mfcc_28_std", "mfcc_26_std"]
ACOUSTIC_A = MFCC                                                        # 80
ACOUSTIC_B = NON_MFCC                                                    # 11
ACOUSTIC_C = [c for c in MFCC if c not in TOP10_SHIFTED_MFCC]            # 70

TYPE_PATTERNS = {
    "fire": ["ගිනි", "ගින්", "දුම", "எரிச்சல்", "தீ", "புகை", "fire", "smoke", "burning"],
    "flood_water": ["වතුර", "ගංවතුර", "යටවෙලා", "හිරවෙලා", "සිරවෙලා", "කොටුවෙලා",
                    "வெள்ளம்", "தண்ணீர்", "flood", "water"],
    "tree_structural": ["ගහක්", "ගස්", "කඩන්නක්", "කඩන්නාව", "වැටිලා", "පුළුගහක්", "tree"],
    "tsunami_wind": ["සුනාමි", "සුළං", "සුළි", "tsunami"],
    "road_travel": ["පාර", "රූට්", "යන්න පුළුවන්", "route", "road", "kolamba", "කොළඹ"],
    "medical_rescue": ["හොස්පිට්ල්", "රෝහල", "දොක්තර", "வைத்தியர்", "hospital", "boat",
                       "බෝට්", "රැකවරණ", "රැකගන්න"],
    "status_no_impact": ["බලපෑමක් නැහැ", "අවදානමක් නැහ", "no impact"],
}
FLOOD_FIRE_CATEGORIES = {"fire", "flood_water", "flood_water+road_travel", "fire+flood_water"}


def categorize(text):
    if not isinstance(text, str) or not text.strip():
        return "empty"
    hits = [cat for cat, patterns in TYPE_PATTERNS.items() if any(p in text for p in patterns)]
    return "+".join(hits) if hits else "unclear"


def fire_smoke_match(text):
    kws = ["fire","smoke","burning","flames","gas","ගිනිගන්නව","ගිනිඅරගෙන",
           "ගින්නක්","ගිනි","දුමක්","දුම","தீ","புகை","gini"]
    if not isinstance(text, str): return 0
    t = text.lower()
    return int(any(k.lower() in t for k in kws))


def fit_predict_proba(X_train, y_train, X_test, seed):
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=seed)),
        ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_test)


def multiseed(sim, dmc, y_sim, y_dmc, cols, high_idx):
    X_sim = sim[cols].fillna(0).values
    X_dmc = dmc[cols].fillna(0).values
    accs, macros, recs = [], [], []
    for seed in SEEDS:
        proba = fit_predict_proba(X_sim, y_sim, X_dmc, seed)
        pred = proba.argmax(axis=1)
        accs.append(accuracy_score(y_dmc, pred))
        macros.append(f1_score(y_dmc, pred, average="macro"))
        _, r, _, _ = precision_recall_fscore_support(y_dmc, pred, labels=[0, 1, 2], zero_division=0)
        recs.append(r)
    return np.mean(accs), np.mean(macros), np.mean(recs, axis=0)


def multiseed_late(sim, dmc, y_sim, y_dmc, acoustic_cols, high_idx, w=0.5):
    X_sim_ac = sim[acoustic_cols].fillna(0).values
    X_sim_tx = sim[CONDITION_C].fillna(0).values
    X_dmc_ac = dmc[acoustic_cols].fillna(0).values
    X_dmc_tx = dmc[CONDITION_C].fillna(0).values
    accs, macros, recs = [], [], []
    for seed in SEEDS:
        proba_ac = fit_predict_proba(X_sim_ac, y_sim, X_dmc_ac, seed)
        proba_tx = fit_predict_proba(X_sim_tx, y_sim, X_dmc_tx, seed)
        combined = w * proba_ac + (1 - w) * proba_tx
        pred = combined.argmax(axis=1)
        accs.append(accuracy_score(y_dmc, pred))
        macros.append(f1_score(y_dmc, pred, average="macro"))
        _, r, _, _ = precision_recall_fscore_support(y_dmc, pred, labels=[0, 1, 2], zero_division=0)
        recs.append(r)
    return np.mean(accs), np.mean(macros), np.mean(recs, axis=0)


def print_row(name, acc, macro, rec, high_idx):
    print(f"  {name:<50} acc={acc:.4f}  macroF1={macro:.4f}  "
          f"HighRec={rec[high_idx]:.4f}  LowRec={rec[1]:.4f}  MedRec={rec[2]:.4f}")


def run_for_subset(subset_name, sim, dmc, y_sim, y_dmc, le, high_idx):
    print("\n" + "=" * 95)
    print(f"{subset_name} (n={len(dmc)})")
    print("=" * 95)

    acc, macro, rec = multiseed(sim, dmc, y_sim, y_dmc, CONDITION_C, high_idx)
    print_row("Text-only (reference)", acc, macro, rec, high_idx)
    acc, macro, rec = multiseed(sim, dmc, y_sim, y_dmc, ACOUSTIC_FULL, high_idx)
    print_row("Acoustic FULL (91, reference)", acc, macro, rec, high_idx)

    print("\n  -- LATE FUSION (0.5/0.5) with each acoustic variant --")
    for name, cols in [("Acoustic-A (MFCC only, 80) + text", ACOUSTIC_A),
                        ("Acoustic-B (non-MFCC only, 11) + text", ACOUSTIC_B),
                        ("Acoustic-C (MFCC minus top10 shifted, 70) + text", ACOUSTIC_C)]:
        acc, macro, rec = multiseed_late(sim, dmc, y_sim, y_dmc, cols, high_idx)
        print_row(name, acc, macro, rec, high_idx)


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    high_idx = list(le.classes_).index("High")

    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim = sim.copy()
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    y_sim = le.transform(sim["urgency_label"].values)

    dmc59 = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")

    # n=11: keyword-matched fire+flood
    dmc_v2 = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation_v2.csv")
    dmc11_src = dmc59.reset_index(drop=True).copy()
    dmc_v2 = dmc_v2.reset_index(drop=True)
    dmc11_src["category"] = dmc_v2["transcript"].map(categorize)
    dmc11 = dmc11_src[dmc11_src["category"].isin(FLOOD_FIRE_CATEGORIES)].reset_index(drop=True)
    y_dmc11 = le.transform(dmc11["true_label"].values)

    # n=25: authoritative flood-only
    meta = pd.read_csv(f"{ROOT}\\data\\dmc_metadata.csv")
    dmc25_src = dmc59.merge(meta[["recording_id", "disaster_type"]], on="recording_id", how="inner")
    dmc25 = dmc25_src[dmc25_src["disaster_type"] == "Flood"].reset_index(drop=True)
    y_dmc25 = le.transform(dmc25["true_label"].values)

    run_for_subset("n=11 -- keyword-matched fire+flood", sim, dmc11, y_sim, y_dmc11, le, high_idx)
    run_for_subset("n=25 -- authoritative flood-only", sim, dmc25, y_sim, y_dmc25, le, high_idx)

    print("\n" + "=" * 95)
    print("REFERENCE -- same variants on full n=59 (already established):")
    print("=" * 95)
    print("  Text-only                                    : acc=0.4576 macroF1=0.4485 HighRec=0.4435")
    print("  Acoustic-A (MFCC only) + text (late fusion)  : acc=0.4339 macroF1=0.4265 HighRec=0.4174")
    print("  Acoustic-B (non-MFCC only) + text (late fusion): acc=0.4136 macroF1=0.3971 HighRec=0.5217")
    print("  Acoustic-C (MFCC minus top10) + text (late fusion): acc=0.4203 macroF1=0.4118 HighRec=0.4174")


if __name__ == "__main__":
    main()
