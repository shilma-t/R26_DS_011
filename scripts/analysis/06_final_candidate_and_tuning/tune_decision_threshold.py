"""
tune_decision_threshold.py
Cost-sensitive decision-rule tuning, isolated from feature/model changes.

Model and features stay exactly as frozen (v1: 111 features, RF 200 trees).
The only thing that changes is HOW predict_proba's three numbers get turned
into a label: instead of plain argmax(P_High, P_Low, P_Medium), we search
a small grid of per-class weight multipliers and take
argmax(w_High*P_High, w_Low*P_Low, w_Medium*P_Medium).

Protocol (agreed before running):
  1. Tune ONLY on the 267-call SIMULATED corpus, via the same 5-fold
     StratifiedKFold(n_splits=5, seed=42) used everywhere else tonight -
     never on DMC, to keep DMC a genuine held-out external test.
  2. Select two candidate weight vectors from the simulated CV results:
       - best macro-F1 (balanced choice)
       - best High recall subject to accuracy not dropping more than 3
         points below the plain-argmax baseline (guards against a
         degenerate "always predict High" solution)
  3. FREEZE both. Apply them exactly once to the EXISTING 59-call DMC
     predict_proba already saved in dmc_evaluation.csv - no reruns, no
     retuning on DMC.
  4. Report accuracy, macro-F1, full confusion matrix, and per-class
     precision/recall for baseline vs both tuned rules, side by side.
"""
import os
import sys
import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

_PIPE = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011\scripts\pipeline"
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)
from text_features_from_transcript import fire_smoke_match  # noqa: E402 - frozen, faithful list

SEED = 42
RF_PARAMS = dict(n_estimators=200, min_samples_leaf=2,
                 class_weight="balanced", random_state=SEED, n_jobs=-1)

ACOUSTIC = ([f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")] +
    ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
     "spec_centroid_mean", "speaking_rate", "jitter",
     "rms_slope", "rms_gradient", "speaking_rate_change"])
TEXT = ["asr_quality", "keyword_count", "word_count", "repetition_rate", "sentiment_polarity"]
CONTEXT = ["sin_context_score", "sin_danger_count", "sin_help_request_count",
    "sin_high_risk_phrase_count", "sin_location_context_count",
    "sin_person_at_risk_count", "sin_safety_state_count",
    "tam_context_score", "tam_danger_count", "tam_help_request_count",
    "tam_high_risk_phrase_count", "tam_location_context_count",
    "tam_person_at_risk_count", "tam_safety_state_count"]
FEATURE_COLS = ACOUSTIC + TEXT + CONTEXT + ["fire_smoke_match"]

WEIGHT_GRID_HIGH = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.3, 2.6]
WEIGHT_GRID_LOW = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # below 1 = require more p(Low) to call Low


def apply_weights(proba, w_high, w_low, w_medium=1.0):
    """proba columns are in CLASSES order [High, Low, Medium] (alphabetical)."""
    weighted = proba * np.array([w_high, w_low, w_medium])
    return weighted.argmax(axis=1)


def cv_predict_proba(X, y, seed=SEED):
    """5-fold CV predict_proba for every row, using a model that never saw it."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out = np.zeros((len(y), 3))
    for train_idx, test_idx in skf.split(X, y):
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=seed)),
            ("clf", RandomForestClassifier(**RF_PARAMS)),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        out[test_idx] = pipe.predict_proba(X[test_idx])
    return out


def report(y_true, y_pred, label, le):
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    print(f"\n--- {label} ---")
    print(f"accuracy={acc:.4f}  macro-F1={macro:.4f}")
    print(classification_report(y_true, y_pred, target_names=le.classes_, digits=3))
    cm = confusion_matrix(y_true, y_pred, labels=le.transform(le.classes_))
    print("Confusion matrix (rows=true, cols=pred), classes:", list(le.classes_))
    print(cm)
    return acc, macro


def main():
    print("=" * 78)
    print("STEP 1-2: tune weights on SIMULATED corpus, 5-fold CV")
    print("=" * 78)

    sim = pd.read_csv(r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011\data\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)

    le = LabelEncoder().fit(["High", "Low", "Medium"])
    X = sim[FEATURE_COLS].fillna(0).values
    y = le.transform(sim["urgency_label"].values)

    proba_cv = cv_predict_proba(X, y)

    baseline_pred = proba_cv.argmax(axis=1)
    base_acc, base_macro = report(y, baseline_pred, "BASELINE (plain argmax), simulated CV", le)
    base_high_recall = (baseline_pred[y == 0] == 0).mean()  # High is index 0 alphabetically
    print(f"High recall (baseline): {base_high_recall:.4f}")

    print("\nGrid search over weight multipliers ...")
    results = []
    for w_high, w_low in product(WEIGHT_GRID_HIGH, WEIGHT_GRID_LOW):
        pred = apply_weights(proba_cv, w_high, w_low)
        acc = accuracy_score(y, pred)
        macro = f1_score(y, pred, average="macro")
        high_recall = (pred[y == 0] == 0).mean()
        results.append({"w_high": w_high, "w_low": w_low, "accuracy": acc,
                        "macro_f1": macro, "high_recall": high_recall})
    grid = pd.DataFrame(results)

    best_macro = grid.loc[grid["macro_f1"].idxmax()]
    guarded = grid[grid["accuracy"] >= base_acc - 0.03]
    best_recall_guarded = guarded.loc[guarded["high_recall"].idxmax()]

    print(f"\nBest macro-F1 config      : w_high={best_macro.w_high} w_low={best_macro.w_low}  "
          f"acc={best_macro.accuracy:.4f} macro_f1={best_macro.macro_f1:.4f} "
          f"high_recall={best_macro.high_recall:.4f}")
    print(f"Best High-recall (acc guard >= {base_acc-0.03:.3f}): "
          f"w_high={best_recall_guarded.w_high} w_low={best_recall_guarded.w_low}  "
          f"acc={best_recall_guarded.accuracy:.4f} macro_f1={best_recall_guarded.macro_f1:.4f} "
          f"high_recall={best_recall_guarded.high_recall:.4f}")

    grid.to_csv(r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011\reports\pp2\threshold_grid_search.csv", index=False)

    print("\n" + "=" * 78)
    print("STEP 3-4: apply FROZEN weights ONCE to existing 59-call DMC results")
    print("=" * 78)

    dmc = pd.read_csv(r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011\reports\pp2\dmc_evaluation.csv")
    dmc_proba = dmc[["prob_High", "prob_Low", "prob_Medium"]].values
    y_true_dmc = le.transform(dmc["true_label"].values)

    baseline_dmc_pred = dmc_proba.argmax(axis=1)
    report(y_true_dmc, baseline_dmc_pred, "DMC BASELINE (plain argmax, already known)", le)

    for name, w_high, w_low in [
        ("DMC + best-macro-F1 weights", best_macro.w_high, best_macro.w_low),
        ("DMC + best-High-recall weights", best_recall_guarded.w_high, best_recall_guarded.w_low),
    ]:
        pred = apply_weights(dmc_proba, w_high, w_low)
        report(y_true_dmc, pred, f"{name} (w_high={w_high}, w_low={w_low})", le)


if __name__ == "__main__":
    main()
