"""
text_only_minus_spurious.py
Follow-up to medium_collapse_diagnostic.py: word_count and repetition_rate
were found to be top-3 important features in the text-only model despite
being near-noise in simulated training data (9.3-11.6 words across all three
classes) and massively domain-shifted on DMC (word_count: 10.1 sim vs 47.2
DMC, Cohen's d=-1.39) - causing a systematic collapse toward "Medium".

This tests text/context-only WITHOUT those two features (18 features),
same 5-seed rigor as text_only_candidate.py, compared against both the
full 20-feature text-only model and v1 (111 features) on DMC.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
RF_BASE = dict(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1)
SEEDS = [1, 7, 42, 99, 123]

ACOUSTIC = ([f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")] +
    ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
     "spec_centroid_mean", "speaking_rate", "jitter",
     "rms_slope", "rms_gradient", "speaking_rate_change"])
TEXT_CONTEXT = ["asr_quality", "keyword_count", "word_count", "repetition_rate",
    "sentiment_polarity", "sin_context_score", "sin_danger_count",
    "sin_help_request_count", "sin_high_risk_phrase_count",
    "sin_location_context_count", "sin_person_at_risk_count",
    "sin_safety_state_count", "tam_context_score", "tam_danger_count",
    "tam_help_request_count", "tam_high_risk_phrase_count",
    "tam_location_context_count", "tam_person_at_risk_count",
    "tam_safety_state_count", "fire_smoke_match"]
FULL = ACOUSTIC + TEXT_CONTEXT
SPURIOUS = ["word_count", "repetition_rate"]
TEXT_CONTEXT_TRIMMED = [c for c in TEXT_CONTEXT if c not in SPURIOUS]
assert len(TEXT_CONTEXT_TRIMMED) == 18


def fire_smoke_match(text):
    kws = ["fire","smoke","burning","flames","gas","ගිනිගන්නව","ගිනිඅරගෙන",
           "ගින්නක්","ගිනි","දුමක්","දුම","தீ","புகை","gini"]
    if not isinstance(text, str): return 0
    t = text.lower()
    return int(any(k.lower() in t for k in kws))


def cv_predict_proba(X, y, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out = np.zeros((len(y), 3))
    for train_idx, test_idx in skf.split(X, y):
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=seed)),
            ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        out[test_idx] = pipe.predict_proba(X[test_idx])
    return out


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
    y_sim = le.transform(sim["urgency_label"].values)

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    y_dmc = le.transform(dmc["true_label"].values)

    conditions = [
        ("TEXT-only trimmed (18, no word_count/repetition_rate)", TEXT_CONTEXT_TRIMMED),
        ("TEXT-only full (20)", TEXT_CONTEXT),
        ("FULL (111, v1)", FULL),
    ]

    print("=" * 78)
    print("STEP 1: simulated 5-fold CV (in-domain capability check)")
    print("=" * 78)
    for name, cols in conditions:
        X = sim[cols].fillna(0).values
        accs, macros, hrs = [], [], []
        for seed in SEEDS:
            proba = cv_predict_proba(X, y_sim, seed)
            pred = proba.argmax(axis=1)
            accs.append(accuracy_score(y_sim, pred))
            macros.append(f1_score(y_sim, pred, average="macro"))
            hrs.append((pred[y_sim == 0] == 0).mean())
        print(f"\n{name}:")
        print(f"  acc={np.mean(accs):.4f}  macroF1={np.mean(macros):.4f}  HighRecall={np.mean(hrs):.4f}")

    print("\n" + "=" * 78)
    print("STEP 2: DMC (59 calls), production RF retrained per seed, mean over 5 seeds")
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
        for seed, a, h in zip(SEEDS, accs, hrs):
            print(f"    seed={seed:<4} acc={a:.4f}  HighRecall={h:.4f}")

    print("\n" + "=" * 78)
    print("STEP 3: trimmed text-only vs v1, win counts across 5 seeds")
    print("=" * 78)
    trimmed = results["TEXT-only trimmed (18, no word_count/repetition_rate)"]
    full_text = results["TEXT-only full (20)"]
    v1 = results["FULL (111, v1)"]

    for label, cand in [("trimmed text-only vs v1", trimmed), ("trimmed text-only vs full text-only", full_text)]:
        other = v1 if "v1" in label else full_text
        acc_wins = sum(1 for a, b in zip(cand["accs"], other["accs"]) if a >= b) if cand is not other else None
    acc_wins_v1 = sum(1 for a, b in zip(trimmed["accs"], v1["accs"]) if a >= b)
    hr_wins_v1 = sum(1 for a, b in zip(trimmed["hrs"], v1["hrs"]) if a >= b)
    acc_wins_fulltext = sum(1 for a, b in zip(trimmed["accs"], full_text["accs"]) if a >= b)
    hr_wins_fulltext = sum(1 for a, b in zip(trimmed["hrs"], full_text["hrs"]) if a >= b)

    print(f"  trimmed accuracy   mean={np.mean(trimmed['accs']):.4f}  vs v1={np.mean(v1['accs']):.4f}  "
          f"vs full-text={np.mean(full_text['accs']):.4f}")
    print(f"  trimmed HighRecall mean={np.mean(trimmed['hrs']):.4f}  vs v1={np.mean(v1['hrs']):.4f}  "
          f"vs full-text={np.mean(full_text['hrs']):.4f}")
    print(f"  trimmed >= v1 on accuracy   : {acc_wins_v1}/5 seeds")
    print(f"  trimmed >= v1 on HighRecall : {hr_wins_v1}/5 seeds")
    print(f"  trimmed >= full-text on accuracy   : {acc_wins_fulltext}/5 seeds")
    print(f"  trimmed >= full-text on HighRecall : {hr_wins_fulltext}/5 seeds")

    print("\n" + "=" * 78)
    print("Aggregated confusion matrix, TRIMMED text-only, summed over 5 seeds")
    print("=" * 78)
    cm_sum = np.zeros((3, 3), dtype=int)
    for pred in trimmed["preds"]:
        cm_sum += confusion_matrix(y_dmc, pred, labels=le.transform(le.classes_))
    print("classes:", list(le.classes_), " (rows=true, cols=predicted)")
    print(cm_sum)

    print("\nPredicted-label totals across 5 seeds (trimmed text-only):")
    all_preds_flat = np.concatenate(trimmed["preds"])
    labels_flat = le.inverse_transform(all_preds_flat)
    print(pd.Series(labels_flat).value_counts().to_dict())

    print("\nFor comparison, FULL text-only (20-feat) predicted-label totals:")
    all_preds_flat_full = np.concatenate(full_text["preds"])
    labels_flat_full = le.inverse_transform(all_preds_flat_full)
    print(pd.Series(labels_flat_full).value_counts().to_dict())


if __name__ == "__main__":
    main()
