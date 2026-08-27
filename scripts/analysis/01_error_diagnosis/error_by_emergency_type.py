"""
error_by_emergency_type.py
Categorizes ALL 59 DMC calls (not just the earlier zero-keyword subset) by
apparent emergency type using keyword-pattern scanning of the cleaner v2
transcripts, counts how many simulated (v1) training calls fall into each
type, and cross-references with Condition C's (text-only) actual per-call
correctness on DMC - to answer: which emergency types are underrepresented
in training, and does that predict where text-only fails?

Categorization is keyword-heuristic, best-effort (not native-speaker
verified) - flagged per call so mis-categorizations can be spotted.
"""
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
RF_BASE = dict(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1)
SEEDS = [1, 7, 42, 99, 123]

CONDITION_C = ["asr_quality", "keyword_count", "repetition_rate", "sentiment_polarity",
    "sin_context_score", "sin_danger_count", "sin_help_request_count",
    "sin_high_risk_phrase_count", "sin_location_context_count",
    "sin_person_at_risk_count", "sin_safety_state_count", "tam_context_score",
    "tam_danger_count", "tam_help_request_count", "tam_high_risk_phrase_count",
    "tam_location_context_count", "tam_person_at_risk_count",
    "tam_safety_state_count", "fire_smoke_match"]

# Keyword patterns per emergency type (Sinhala/Tamil/English), best-effort,
# built from transcripts read during the earlier categorization pass.
TYPE_PATTERNS = {
    "fire": ["ගිනි", "ගින්", "දුම", "எரிச்சல்", "தீ", "புகை", "fire", "smoke", "burning"],
    "flood_water": ["වතුර", "ගංවතුර", "යටවෙලා", "හිරවෙලා", "සිරවෙලා", "කොටුවෙලා",
                    "වெள்ளம்", "தண்ணீர்", "flood", "water"],
    "tree_structural": ["ගහක්", "ගස්", "කඩන්නක්", "කඩන්නාව", "වැටිලා", "පුළුගහක්", "tree"],
    "tsunami_wind": ["සුනාමි", "සුළං", "සුළි", "tsunami"],
    "road_travel": ["පාර", "රූට්", "යන්න පුළුවන්", "route", "road", "kolamba", "කොළඹ"],
    "medical_rescue": ["හොස්පිට්ල්", "රෝහල", "දොක්තර", "வைத்தியர்", "hospital", "boat",
                       "බෝට්", "රැකවරණ", "රැකගන්න"],
    "status_no_impact": ["බලපෑමක් නැහැ", "අවදානමක් නැහ", "no impact"],
}


def categorize(text):
    if not isinstance(text, str) or not text.strip():
        return "empty"
    hits = []
    for cat, patterns in TYPE_PATTERNS.items():
        if any(p in text for p in patterns):
            hits.append(cat)
    if not hits:
        return "unclear"
    return "+".join(hits)  # multiple categories can co-occur


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
    y_sim = le.transform(sim["urgency_label"].values)
    X_sim = sim[CONDITION_C].fillna(0).values

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    dmc_v2 = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation_v2.csv")
    y_dmc = le.transform(dmc["true_label"].values)
    X_dmc = dmc[CONDITION_C].fillna(0).values

    print("=" * 78)
    print("STEP 1: categorize SIMULATED training transcripts by emergency type")
    print("=" * 78)
    sim["category"] = sim["transcript"].map(categorize)
    sim_counts = sim["category"].value_counts()
    print(sim_counts.to_string())

    print("\n" + "=" * 78)
    print("STEP 2: categorize ALL 59 DMC transcripts (v2, cleaner ASR) by type")
    print("=" * 78)
    dmc_v2["category"] = dmc_v2["transcript"].map(categorize)
    dmc_counts = dmc_v2["category"].value_counts()
    print(dmc_counts.to_string())

    print("\n" + "=" * 78)
    print("STEP 3: train Condition C (text-only) at 5 seeds, get per-call correctness")
    print("=" * 78)
    correct_matrix = np.zeros((len(SEEDS), len(y_dmc)), dtype=bool)
    for si, seed in enumerate(SEEDS):
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=seed)),
            ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
        ])
        pipe.fit(X_sim, y_sim)
        pred = pipe.predict(X_dmc)
        correct_matrix[si] = (pred == y_dmc)
    mean_correct = correct_matrix.mean(axis=0)  # fraction of 5 seeds correct, per call

    dmc_reset = dmc.reset_index(drop=True)
    dmc_v2_reset = dmc_v2.reset_index(drop=True)
    assert (dmc_reset["recording_id"] == dmc_v2_reset["recording_id"]).all(), \
        "recording_id order mismatch between v1 and v2 DMC files"

    result_df = pd.DataFrame({
        "recording_id": dmc_reset["recording_id"],
        "true_label": dmc_reset["true_label"],
        "category": dmc_v2_reset["category"],
        "text_only_accuracy": mean_correct,
    })

    print("\n" + "=" * 78)
    print("STEP 4: Condition C (text-only) accuracy BY EMERGENCY TYPE on DMC")
    print("(mean fraction of 5 seeds correct, per category)")
    print("=" * 78)
    by_cat = result_df.groupby("category").agg(
        n_calls=("recording_id", "count"),
        mean_accuracy=("text_only_accuracy", "mean"),
    ).sort_values("n_calls", ascending=False)
    print(by_cat.round(3).to_string())

    print("\n" + "=" * 78)
    print("STEP 5: side-by-side - simulated training coverage vs DMC prevalence vs "
          "text-only accuracy, per type")
    print("=" * 78)
    all_types = set(sim_counts.index) | set(dmc_counts.index)
    rows = []
    for t in sorted(all_types):
        n_sim = int(sim_counts.get(t, 0))
        n_dmc = int(dmc_counts.get(t, 0))
        acc = by_cat.loc[t, "mean_accuracy"] if t in by_cat.index else np.nan
        rows.append({"type": t, "n_simulated": n_sim, "n_dmc": n_dmc,
                    "text_only_accuracy_on_dmc": acc})
    summary = pd.DataFrame(rows).sort_values("n_dmc", ascending=False)
    print(summary.round(3).to_string(index=False))

    print("\n" + "=" * 78)
    print("Per-call detail (for inspection / native-speaker sanity check)")
    print("=" * 78)
    print(result_df.sort_values("category").to_string(index=False))


if __name__ == "__main__":
    main()
