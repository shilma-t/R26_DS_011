"""
high_to_low_error_analysis.py
Supervisor's specific ask: which True-High DMC calls get classified as Low
(the most dangerous error direction - a missed critical call, not just any
misclassification), and why?

Uses Condition C (19 text/context features, current best candidate) at all
5 validation seeds, and flags calls that are CONSISTENTLY predicted Low
across most/all seeds (a robust, real failure - not one seed's noise).
For each, pulls the transcript and the specific feature values driving the
decision to diagnose the cause.
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

CONDITION_C = ["asr_quality", "keyword_count", "repetition_rate", "sentiment_polarity",
    "sin_context_score", "sin_danger_count", "sin_help_request_count",
    "sin_high_risk_phrase_count", "sin_location_context_count",
    "sin_person_at_risk_count", "sin_safety_state_count", "tam_context_score",
    "tam_danger_count", "tam_help_request_count", "tam_high_risk_phrase_count",
    "tam_location_context_count", "tam_person_at_risk_count",
    "tam_safety_state_count", "fire_smoke_match"]


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
    X_sim = sim[CONDITION_C].fillna(0).values
    y_sim = le.transform(sim["urgency_label"].values)

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    X_dmc = dmc[CONDITION_C].fillna(0).values
    y_dmc = le.transform(dmc["true_label"].values)
    classes = list(le.classes_)  # [High, Low, Medium]

    print("Training Condition C at 5 seeds, collecting predictions on DMC ...")
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
    all_preds = np.array(all_preds)  # shape (5, 59)
    all_probas = np.array(all_probas)  # shape (5, 59, 3)

    high_idx = classes.index("High")
    low_idx = classes.index("Low")

    true_high_mask = (y_dmc == high_idx)
    n_true_high = true_high_mask.sum()
    print(f"\n{n_true_high} true-High calls in DMC")

    high_to_low_count = np.zeros(len(y_dmc), dtype=int)
    for seed_i in range(len(SEEDS)):
        for call_i in range(len(y_dmc)):
            if true_high_mask[call_i] and all_preds[seed_i, call_i] == low_idx:
                high_to_low_count[call_i] += 1

    print("\n" + "=" * 78)
    print("High->Low misclassification frequency, per true-High call (out of 5 seeds)")
    print("=" * 78)
    dmc_reset = dmc.reset_index(drop=True)
    rows = []
    for call_i in np.where(true_high_mask)[0]:
        mean_proba = all_probas[:, call_i, :].mean(axis=0)
        rows.append({
            "recording_id": dmc_reset.loc[call_i, "recording_id"],
            "times_called_Low": int(high_to_low_count[call_i]),
            "mean_prob_High": mean_proba[high_idx],
            "mean_prob_Low": mean_proba[low_idx],
            "mean_prob_Medium": mean_proba[classes.index("Medium")],
            "keyword_count": dmc_reset.loc[call_i, "keyword_count"],
            "sin_danger_count": dmc_reset.loc[call_i, "sin_danger_count"],
            "sin_context_score": dmc_reset.loc[call_i, "sin_context_score"],
            "sin_help_request_count": dmc_reset.loc[call_i, "sin_help_request_count"],
            "asr_quality": dmc_reset.loc[call_i, "asr_quality"],
            "word_count": dmc_reset.loc[call_i, "word_count"],
        })
    result_df = pd.DataFrame(rows).sort_values("times_called_Low", ascending=False)
    print(result_df.to_string(index=False))

    print("\n" + "=" * 78)
    print("CONSISTENT High->Low errors (misclassified as Low in >=4/5 seeds)")
    print("These are the real, robust failures worth diagnosing - not seed noise")
    print("=" * 78)
    consistent = result_df[result_df["times_called_Low"] >= 4]
    if consistent.empty:
        print("  None found at >=4/5 threshold - trying >=3/5")
        consistent = result_df[result_df["times_called_Low"] >= 3]

    for _, row in consistent.iterrows():
        rid = row["recording_id"]
        call_row = dmc_reset[dmc_reset["recording_id"] == rid].iloc[0]
        print(f"\n{'='*78}")
        print(f"CALL: {rid}  (misclassified as Low in {row['times_called_Low']}/5 seeds)")
        print(f"{'='*78}")
        print(f"  mean_prob: High={row['mean_prob_High']:.3f}  Low={row['mean_prob_Low']:.3f}  "
              f"Medium={row['mean_prob_Medium']:.3f}")
        print(f"  keyword_count={row['keyword_count']}  sin_danger_count={row['sin_danger_count']}  "
              f"sin_context_score={row['sin_context_score']:.3f}  "
              f"sin_help_request_count={row['sin_help_request_count']}")
        print(f"  asr_quality={row['asr_quality']}  word_count={row['word_count']}")
        print(f"  language: {call_row.get('language', 'N/A')}")
        transcript = call_row.get("transcript", "")
        transcript = "" if pd.isna(transcript) else str(transcript)
        print(f"  TRANSCRIPT: {transcript[:500] if transcript else '(EMPTY / NaN - ASR produced no transcript)'}")

    print("\n" + "=" * 78)
    print("SUMMARY: likely causes across all consistent High->Low errors")
    print("=" * 78)
    if not consistent.empty:
        avg_kw = consistent["keyword_count"].mean()
        avg_danger = consistent["sin_danger_count"].mean()
        avg_asr = consistent["asr_quality"].mean()
        avg_wc = consistent["word_count"].mean()
        print(f"  Avg keyword_count on these calls: {avg_kw:.2f} "
              f"(vs overall true-High mean: {result_df['keyword_count'].mean():.2f})")
        print(f"  Avg sin_danger_count: {avg_danger:.2f} "
              f"(vs overall true-High mean: {result_df['sin_danger_count'].mean():.2f})")
        print(f"  Avg asr_quality: {avg_asr:.3f} (vs overall true-High mean: "
              f"{result_df['asr_quality'].mean():.3f})")
        print(f"  Avg word_count: {avg_wc:.1f} (vs overall true-High mean: "
              f"{result_df['word_count'].mean():.1f})")


if __name__ == "__main__":
    main()
