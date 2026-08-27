"""
text_only_candidate.py
Follow-up to feature_source_ablation.py's Part 1 finding: a text/context-only
(20-feature) model beat the full 111-feature model (v1) on the 59-call DMC
set, on every metric, including High recall. This script re-runs that
comparison with the same multi-seed rigor used for the decision-threshold
work (5 seeds, not 3), so the result can be argued confidently either as a
primary candidate or as a documented diagnostic finding.

Frozen v1 artifacts are untouched - this trains disposable, in-memory models
on already-saved data only (data/features.csv, reports/pp2/dmc_evaluation.csv).
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                              classification_report, precision_recall_fscore_support)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
RF_BASE = dict(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1)
SEEDS = [1, 7, 42, 99, 123]  # same 5-seed set used for the decision-threshold work

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
assert len(ACOUSTIC) == 91 and len(TEXT_CONTEXT) == 20 and len(FULL) == 111


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

    print("=" * 78)
    print("STEP 1: text/context-only (20 feat) vs FULL (111 feat, v1), simulated CV")
    print(f"5-fold StratifiedKFold, seeds={SEEDS}")
    print("=" * 78)

    for name, cols in [("TEXT/CONTEXT-only (20)", TEXT_CONTEXT), ("FULL (111, v1)", FULL)]:
        X = sim[cols].fillna(0).values
        accs, macros, hrs = [], [], []
        for seed in SEEDS:
            proba = cv_predict_proba(X, y_sim, seed)
            pred = proba.argmax(axis=1)
            accs.append(accuracy_score(y_sim, pred))
            macros.append(f1_score(y_sim, pred, average="macro"))
            hrs.append((pred[y_sim == 0] == 0).mean())
        print(f"\n{name} - simulated 5-fold CV, mean over {len(SEEDS)} seeds:")
        print(f"  acc={np.mean(accs):.4f} (+/-{np.std(accs):.4f})  "
              f"macroF1={np.mean(macros):.4f} (+/-{np.std(macros):.4f})  "
              f"HighRecall={np.mean(hrs):.4f} (+/-{np.std(hrs):.4f})")

    print("\n" + "=" * 78)
    print("STEP 2+3: apply to the 59-call DMC set, production RF retrained per seed")
    print("(DMC feature vectors are the same already-saved ones used throughout)")
    print("=" * 78)

    results = {}
    for name, cols in [("TEXT/CONTEXT-only (20)", TEXT_CONTEXT), ("FULL (111, v1)", FULL)]:
        X_sim = sim[cols].fillna(0).values
        X_dmc = dmc[cols].fillna(0).values

        accs, macros, hrs = [], [], []
        all_preds = []
        for seed in SEEDS:
            pred, _ = fit_on_all_predict(X_sim, y_sim, X_dmc, seed)
            all_preds.append(pred)
            accs.append(accuracy_score(y_dmc, pred))
            macros.append(f1_score(y_dmc, pred, average="macro"))
            hrs.append((pred[y_dmc == 0] == 0).mean())
        results[name] = {"accs": accs, "macros": macros, "hrs": hrs, "preds": all_preds}

        print(f"\n{name} on DMC (59 calls), mean over {len(SEEDS)} seeds:")
        print(f"  acc={np.mean(accs):.4f} (+/-{np.std(accs):.4f})  "
              f"macroF1={np.mean(macros):.4f} (+/-{np.std(macros):.4f})  "
              f"HighRecall={np.mean(hrs):.4f} (+/-{np.std(hrs):.4f})")
        for seed, a, m, h in zip(SEEDS, accs, macros, hrs):
            print(f"    seed={seed:<4} acc={a:.4f}  macroF1={m:.4f}  HighRecall={h:.4f}")

    print("\n" + "=" * 78)
    print("STEP 3: direct comparison, text-only vs FULL (v1), on DMC")
    print("=" * 78)
    t = results["TEXT/CONTEXT-only (20)"]
    f = results["FULL (111, v1)"]
    print(f"  mean accuracy   : text-only={np.mean(t['accs']):.4f}  vs  v1={np.mean(f['accs']):.4f}  "
          f"delta={np.mean(t['accs'])-np.mean(f['accs']):+.4f}")
    print(f"  mean macro-F1   : text-only={np.mean(t['macros']):.4f}  vs  v1={np.mean(f['macros']):.4f}  "
          f"delta={np.mean(t['macros'])-np.mean(f['macros']):+.4f}")
    print(f"  mean HighRecall : text-only={np.mean(t['hrs']):.4f}  vs  v1={np.mean(f['hrs']):.4f}  "
          f"delta={np.mean(t['hrs'])-np.mean(f['hrs']):+.4f}")
    acc_wins = sum(1 for a, b in zip(t["accs"], f["accs"]) if a >= b)
    hr_wins = sum(1 for a, b in zip(t["hrs"], f["hrs"]) if a >= b)
    print(f"  text-only >= v1 on accuracy   in {acc_wins}/{len(SEEDS)} seeds")
    print(f"  text-only >= v1 on HighRecall in {hr_wins}/{len(SEEDS)} seeds")

    print("\n" + "=" * 78)
    print("Full confusion matrices - one representative seed (42) - both models")
    print("=" * 78)
    seed_idx = SEEDS.index(42)
    for name in ["TEXT/CONTEXT-only (20)", "FULL (111, v1)"]:
        pred = results[name]["preds"][seed_idx]
        cm = confusion_matrix(y_dmc, pred, labels=le.transform(le.classes_))
        print(f"\n{name}  (seed=42)")
        print("classes:", list(le.classes_), " (rows=true, cols=predicted)")
        print(cm)
        print(classification_report(y_dmc, pred, target_names=le.classes_, digits=3))

    print("\n" + "=" * 78)
    print("Aggregated confusion matrix, text-only, SUMMED across all 5 seeds")
    print("(shows the dominant/stable error directions, not one seed's noise)")
    print("=" * 78)
    cm_sum_text = np.zeros((3, 3), dtype=int)
    cm_sum_full = np.zeros((3, 3), dtype=int)
    for pred in results["TEXT/CONTEXT-only (20)"]["preds"]:
        cm_sum_text += confusion_matrix(y_dmc, pred, labels=le.transform(le.classes_))
    for pred in results["FULL (111, v1)"]["preds"]:
        cm_sum_full += confusion_matrix(y_dmc, pred, labels=le.transform(le.classes_))
    print("\nTEXT/CONTEXT-only, summed over 5 seeds (rows=true, cols=predicted):")
    print("classes:", list(le.classes_))
    print(cm_sum_text)
    print("\nFULL (v1), summed over 5 seeds (rows=true, cols=predicted):")
    print("classes:", list(le.classes_))
    print(cm_sum_full)

    print("\n" + "=" * 78)
    print("Where it fails: per-class error breakdown, text-only, summed over 5 seeds")
    print("=" * 78)
    classes = list(le.classes_)
    n_true = cm_sum_text.sum(axis=1)
    for i, cls in enumerate(classes):
        total = n_true[i]
        correct = cm_sum_text[i, i]
        row_str = "  ".join(f"{classes[j]}={cm_sum_text[i,j]}" for j in range(3) if j != i)
        print(f"  true={cls:<7} n={total:<3} (x5 seeds)  correct={correct}  "
              f"misclassified as: {row_str}")

    dominant_confusions = []
    for i in range(3):
        for j in range(3):
            if i != j and cm_sum_text[i, j] > 0:
                dominant_confusions.append((cm_sum_text[i, j], classes[i], classes[j]))
    dominant_confusions.sort(reverse=True)
    print("\n  Top confusion pairs (true -> predicted), text-only, summed over 5 seeds:")
    for count, true_c, pred_c in dominant_confusions[:5]:
        print(f"    {true_c} -> {pred_c}: {count} (out of 5 seeds x class count)")


if __name__ == "__main__":
    main()
