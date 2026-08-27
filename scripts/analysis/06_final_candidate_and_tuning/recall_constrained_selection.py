"""
recall_constrained_selection.py
Reframe of the existing Condition C decision-rule tuning using the
"recall-constrained operating point" selection rule from the emergency-triage
literature (Okazaki & Watanuki 2026, "Recall-constrained acoustic decision
support..."): instead of picking the weight pair with the single best
macro-F1, scan the grid and pick the HIGHEST-ACCURACY pair subject to
High recall >= R0, for a few candidate R0 thresholds.

This also doubles as the "expected-cost decision rule" framing requested:
argmax(w_high*P_High, w_low*P_Low, w_medium*P_Medium) is mathematically the
Bayes-optimal decision under an asymmetric 0-1-ish cost matrix where the cost
of missing class c is proportional to w_c. So the weight search already IS a
cost-sensitive decision rule search - this script makes that framing explicit
in the tuning objective (constrained by recall, not just averaged macro-F1)
rather than introducing new machinery.

Tuned entirely on simulated CV (5 seeds). Applied once to DMC (production
RF retrained at 5 validation seeds, same as every other frozen-weight
experiment tonight).
"""
import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
RF_BASE = dict(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1)

CONDITION_C = ["asr_quality", "keyword_count", "repetition_rate", "sentiment_polarity",
    "sin_context_score", "sin_danger_count", "sin_help_request_count",
    "sin_high_risk_phrase_count", "sin_location_context_count",
    "sin_person_at_risk_count", "sin_safety_state_count", "tam_context_score",
    "tam_danger_count", "tam_help_request_count", "tam_high_risk_phrase_count",
    "tam_location_context_count", "tam_person_at_risk_count",
    "tam_safety_state_count", "fire_smoke_match"]

WEIGHT_GRID_HIGH = [1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.3, 2.6]
WEIGHT_GRID_LOW = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TUNING_SEEDS = [1, 7, 42, 99, 123]
R0_CANDIDATES = [0.55, 0.60, 0.65, 0.70]


def fire_smoke_match(text):
    kws = ["fire","smoke","burning","flames","gas","ගිනිගන්නව","ගිනිඅරගෙන",
           "ගින්නක්","ගිනි","දුමක්","දුම","தீ","புகை","gini"]
    if not isinstance(text, str): return 0
    t = text.lower()
    return int(any(k.lower() in t for k in kws))


def apply_weights(proba, w_high, w_low, w_medium=1.0):
    return (proba * np.array([w_high, w_low, w_medium])).argmax(axis=1)


def cv_proba(X, y, seed):
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


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim = sim.copy()
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    X_sim = sim[CONDITION_C].fillna(0).values
    y_sim = le.transform(sim["urgency_label"].values)

    print("=" * 78)
    print("STEP 1: build weight grid on simulated CV, averaged over 5 tuning seeds")
    print("Feature set: Condition C (19 feat, text/context, word_count dropped)")
    print("=" * 78)
    proba_by_seed = {s: cv_proba(X_sim, y_sim, s) for s in TUNING_SEEDS}

    rows = []
    for w_h, w_l in product(WEIGHT_GRID_HIGH, WEIGHT_GRID_LOW):
        accs, macros, hrs = [], [], []
        for seed in TUNING_SEEDS:
            pred = apply_weights(proba_by_seed[seed], w_h, w_l)
            accs.append(accuracy_score(y_sim, pred))
            macros.append(f1_score(y_sim, pred, average="macro"))
            hrs.append((pred[y_sim == 0] == 0).mean())
        rows.append({"w_high": w_h, "w_low": w_l, "mean_acc": np.mean(accs),
                    "mean_macro_f1": np.mean(macros), "mean_high_recall": np.mean(hrs)})
    grid = pd.DataFrame(rows)

    baseline = grid[(grid.w_high == 1.0) & (grid.w_low == 1.0)].iloc[0]
    print(f"Baseline (w=1,1): acc={baseline.mean_acc:.4f}  HR={baseline.mean_high_recall:.4f}")

    print("\n" + "=" * 78)
    print("STEP 2: recall-constrained selection - highest accuracy s.t. HighRecall >= R0")
    print("(cost-sensitive framing: this is equivalent to minimizing expected cost")
    print(" C_FN*P(missed High) + C_FP*P(false High) under the implied weight ratio)")
    print("=" * 78)
    chosen = {}
    for r0 in R0_CANDIDATES:
        feasible = grid[grid["mean_high_recall"] >= r0]
        if feasible.empty:
            print(f"  R0={r0}: NO weight pair reaches this recall on average - infeasible")
            continue
        best = feasible.loc[feasible["mean_acc"].idxmax()]
        chosen[r0] = best
        print(f"  R0={r0}: w_high={best.w_high} w_low={best.w_low}  "
              f"acc={best.mean_acc:.4f}  macro_f1={best.mean_macro_f1:.4f}  "
              f"HR={best.mean_high_recall:.4f}")

    print("\n" + "=" * 78)
    print("STEP 3: apply each frozen R0-selected weight pair to DMC ONCE")
    print("(production RF retrained per seed, DMC untouched during selection)")
    print("=" * 78)

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    X_dmc = dmc[CONDITION_C].fillna(0).values
    y_dmc = le.transform(dmc["true_label"].values)
    medium_idx = list(le.classes_).index("Medium")

    for r0, best in chosen.items():
        w_h, w_l = float(best.w_high), float(best.w_low)
        accs, macros, hrs, mrs = [], [], [], []
        last_pred = None
        for seed in TUNING_SEEDS:
            pipe = ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=seed)),
                ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
            ])
            pipe.fit(X_sim, y_sim)
            proba_dmc = pipe.predict_proba(X_dmc)
            pred = apply_weights(proba_dmc, w_h, w_l)
            last_pred = pred
            accs.append(accuracy_score(y_dmc, pred))
            macros.append(f1_score(y_dmc, pred, average="macro"))
            hrs.append((pred[y_dmc == 0] == 0).mean())
            mrs.append((pred[y_dmc == medium_idx] == medium_idx).mean())

        print(f"\n  R0={r0} (w_high={w_h}, w_low={w_l}) on DMC, mean over 5 seeds:")
        print(f"    acc={np.mean(accs):.4f}  macroF1={np.mean(macros):.4f}  "
              f"HR={np.mean(hrs):.4f}  MedR={np.mean(mrs):.4f}")

    print("\n" + "=" * 78)
    print("Cost-function framing (for write-up, no new computation)")
    print("=" * 78)
    print("""
  argmax(w_High*P_High, w_Low*P_Low, w_Medium*P_Medium) is the Bayes decision
  rule under a cost matrix where the cost of failing to call class c is
  proportional to w_c. Concretely, choosing w_High=1.1, w_Low=0.8 (Condition C's
  best-macro-F1 pair from earlier) is equivalent to saying: a missed High-
  urgency call is costed ~1.375x (=1.1/0.8) more heavily than a missed Low
  call. This lets the PP2 report state the tuning objective in explicit
  cost-sensitive language, matching how the cited triage papers frame their
  decision rules, without any new experiments - it is the same weighted-
  argmax rule already validated on DMC in condition_c_threshold_tuning.py.
""")


if __name__ == "__main__":
    main()
