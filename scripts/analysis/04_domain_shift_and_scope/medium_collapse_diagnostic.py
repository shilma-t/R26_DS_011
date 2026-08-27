"""
medium_collapse_diagnostic.py
Why does the text/context-only (20-feature) model collapse toward
predicting "Medium" on DMC? Three checks:

1. Feature importances of the text-only RF - which features drive the
   High/Low/Medium split, and are they mostly count-based (keyword_count,
   sin/tam_*_count) that would be suppressed by real-world noisy ASR?
2. Per-feature class profile in SIMULATED training data (mean by class) vs
   the actual DMC feature values - is DMC's overall profile closer to the
   simulated "Medium" class than to High or Low?
3. Domain shift (Cohen's d, simulated vs DMC) for each of the 20 features,
   to see which ones moved the most and in which direction.

Diagnostic only - no new model saved, no frozen artifacts touched.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
RF_BASE = dict(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1)
SEED = 42

TEXT_CONTEXT = ["asr_quality", "keyword_count", "word_count", "repetition_rate",
    "sentiment_polarity", "sin_context_score", "sin_danger_count",
    "sin_help_request_count", "sin_high_risk_phrase_count",
    "sin_location_context_count", "sin_person_at_risk_count",
    "sin_safety_state_count", "tam_context_score", "tam_danger_count",
    "tam_help_request_count", "tam_high_risk_phrase_count",
    "tam_location_context_count", "tam_person_at_risk_count",
    "tam_safety_state_count", "fire_smoke_match"]


def fire_smoke_match(text):
    kws = ["fire","smoke","burning","flames","gas","ගිනිගන්නව","ගිනිඅරගෙන",
           "ගින්නක්","ගිනි","දුමක්","දුම","தீ","புகை","gini"]
    if not isinstance(text, str): return 0
    t = text.lower()
    return int(any(k.lower() in t for k in kws))


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled_std


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])

    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim = sim.copy()
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    y_sim = le.transform(sim["urgency_label"].values)
    X_sim = sim[TEXT_CONTEXT].fillna(0).values

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    X_dmc_df = dmc[TEXT_CONTEXT].fillna(0)

    print("=" * 78)
    print("CHECK 1: feature importances of the text-only RF (seed=42)")
    print("=" * 78)
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=SEED)),
        ("clf", RandomForestClassifier(random_state=SEED, **RF_BASE)),
    ])
    pipe.fit(X_sim, y_sim)
    importances = pipe.named_steps["clf"].feature_importances_
    imp_df = pd.DataFrame({"feature": TEXT_CONTEXT, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)
    print(imp_df.to_string(index=False))

    print("\n" + "=" * 78)
    print("CHECK 2: per-feature mean by class (SIMULATED) vs actual DMC mean")
    print("(is DMC's profile closer to simulated Medium than High or Low?)")
    print("=" * 78)
    sim_df = sim[TEXT_CONTEXT + ["urgency_label"]].fillna(0)
    class_means = sim_df.groupby("urgency_label")[TEXT_CONTEXT].mean().T
    dmc_mean = X_dmc_df.mean()
    class_means["DMC_actual"] = dmc_mean
    class_means["closest_class"] = class_means[["High", "Low", "Medium"]].apply(
        lambda row: (row - class_means.loc[row.name, "DMC_actual"]).abs().idxmin(), axis=1)
    print(class_means.round(3).to_string())

    closest_counts = class_means["closest_class"].value_counts()
    print(f"\n  DMC's actual mean is closest to each simulated class' mean, by feature:")
    print(f"  {closest_counts.to_dict()}")

    print("\n" + "=" * 78)
    print("CHECK 3: domain shift (Cohen's d, simulated ALL vs DMC ALL) per feature")
    print("(|d|>0.8 = large shift, 0.5-0.8 = moderate, <0.2 = negligible)")
    print("=" * 78)
    rows = []
    for col in TEXT_CONTEXT:
        d = cohens_d(sim[col].fillna(0).values, X_dmc_df[col].values)
        rows.append({"feature": col, "sim_mean": sim[col].fillna(0).mean(),
                    "dmc_mean": X_dmc_df[col].mean(), "cohens_d": d})
    shift_df = pd.DataFrame(rows).sort_values("cohens_d", key=lambda s: s.abs(), ascending=False)
    print(shift_df.round(3).to_string(index=False))

    print("\n" + "=" * 78)
    print("CHECK 4: for calls PREDICTED Medium on DMC, what do their top-importance")
    print("features look like vs calls predicted High or Low?")
    print("=" * 78)
    pred_dmc = pipe.predict(X_dmc_df.values)
    pred_labels = le.inverse_transform(pred_dmc)
    dmc_with_pred = X_dmc_df.copy()
    dmc_with_pred["predicted"] = pred_labels
    top_feats = imp_df["feature"].head(5).tolist()
    print(f"\nTop 5 important features: {top_feats}")
    print(dmc_with_pred.groupby("predicted")[top_feats].mean().round(3).to_string())
    print(f"\nPredicted label counts: {dmc_with_pred['predicted'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
