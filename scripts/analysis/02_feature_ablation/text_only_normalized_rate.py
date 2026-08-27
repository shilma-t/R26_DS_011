"""
text_only_normalized_rate.py
Follow-up to text_only_minus_spurious.py: dropping word_count/repetition_rate
entirely fixed the Medium-collapse but made the model nearly blind to the
Medium class (1/70 recall). This tests whether a NORMALIZED version of
word_count recovers some Medium signal without reintroducing the collapse.

No duration column exists in the feature set to compute a true "words per
second" rate, so the practical normalization tested here is log1p(word_count)
- this compresses the raw scale gap (sim mean ~10, DMC mean ~47, ratio ~4.7x)
into log-space (log1p(10)=2.4, log1p(47)=3.9, ratio ~1.6x), which is the
standard fix for a count feature with a heavy-tailed domain shift.

repetition_rate is already a bounded proportion (0-1), so log-transforming it
doesn't meaningfully change its scale - it's tested back in raw, unchanged,
as a separate condition to isolate whether IT (not word_count) was actually
driving the collapse.

Four conditions, all vs the 18-feature trimmed baseline and DMC ground truth,
5-seed rigor throughout.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
RF_BASE = dict(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1)
SEEDS = [1, 7, 42, 99, 123]

TEXT_CONTEXT_TRIMMED = ["asr_quality", "keyword_count", "sentiment_polarity",
    "sin_context_score", "sin_danger_count", "sin_help_request_count",
    "sin_high_risk_phrase_count", "sin_location_context_count",
    "sin_person_at_risk_count", "sin_safety_state_count", "tam_context_score",
    "tam_danger_count", "tam_help_request_count", "tam_high_risk_phrase_count",
    "tam_location_context_count", "tam_person_at_risk_count",
    "tam_safety_state_count", "fire_smoke_match"]
assert len(TEXT_CONTEXT_TRIMMED) == 18


def fire_smoke_match(text):
    kws = ["fire","smoke","burning","flames","gas","ගිනිගන්නව","ගිනිඅරගෙන",
           "ගින්නක්","ගිනි","දුමක්","දුම","தீ","புகை","gini"]
    if not isinstance(text, str): return 0
    t = text.lower()
    return int(any(k.lower() in t for k in kws))


def fit_on_all_predict(X_train, y_train, X_test, seed):
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=seed)),
        ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict(X_test), pipe.predict_proba(X_test)


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])

    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim = sim.copy()
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    sim["word_count_log"] = np.log1p(sim["word_count"].fillna(0))
    y_sim = le.transform(sim["urgency_label"].values)

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    dmc["word_count_log"] = np.log1p(dmc["word_count"].fillna(0))
    y_dmc = le.transform(dmc["true_label"].values)

    print("Domain shift check - log1p(word_count):")
    print(f"  sim mean={sim['word_count_log'].mean():.3f}  dmc mean={dmc['word_count_log'].mean():.3f}  "
          f"(raw word_count was sim=10.1 vs dmc=47.2)")

    conditions = [
        ("A: trimmed (18, baseline)", TEXT_CONTEXT_TRIMMED),
        ("B: trimmed + log1p(word_count) (19)", TEXT_CONTEXT_TRIMMED + ["word_count_log"]),
        ("C: trimmed + repetition_rate raw (19)", TEXT_CONTEXT_TRIMMED + ["repetition_rate"]),
        ("D: trimmed + log1p(word_count) + repetition_rate (20)",
         TEXT_CONTEXT_TRIMMED + ["word_count_log", "repetition_rate"]),
    ]

    print("\n" + "=" * 78)
    print("DMC (59 calls), production RF retrained per seed, mean over 5 seeds")
    print("=" * 78)

    results = {}
    for name, cols in conditions:
        X_sim = sim[cols].fillna(0).values
        X_dmc = dmc[cols].fillna(0).values

        accs, macros, hrs = [], [], []
        preds = []
        for seed in SEEDS:
            pred, _ = fit_on_all_predict(X_sim, y_sim, X_dmc, seed)
            preds.append(pred)
            accs.append(accuracy_score(y_dmc, pred))
            macros.append(f1_score(y_dmc, pred, average="macro"))
            hrs.append((pred[y_dmc == 0] == 0).mean())
        results[name] = {"accs": accs, "macros": macros, "hrs": hrs, "preds": preds}

        print(f"\n{name}:")
        print(f"  mean acc={np.mean(accs):.4f} (+/-{np.std(accs):.4f})  "
              f"mean macroF1={np.mean(macros):.4f} (+/-{np.std(macros):.4f})  "
              f"mean HighRecall={np.mean(hrs):.4f} (+/-{np.std(hrs):.4f})")

        all_preds_flat = np.concatenate(preds)
        labels_flat = le.inverse_transform(all_preds_flat)
        print(f"  predicted-label totals (5 seeds): {pd.Series(labels_flat).value_counts().to_dict()}")

    print("\n" + "=" * 78)
    print("Medium-class recall specifically, per condition (summed confusion matrix)")
    print("=" * 78)
    for name, cols in conditions:
        cm_sum = np.zeros((3, 3), dtype=int)
        for pred in results[name]["preds"]:
            cm_sum += confusion_matrix(y_dmc, pred, labels=le.transform(le.classes_))
        medium_idx = list(le.classes_).index("Medium")
        medium_total = cm_sum[medium_idx].sum()
        medium_correct = cm_sum[medium_idx, medium_idx]
        print(f"  {name}: Medium recall = {medium_correct}/{medium_total} = "
              f"{medium_correct/medium_total:.3f}")


if __name__ == "__main__":
    main()
