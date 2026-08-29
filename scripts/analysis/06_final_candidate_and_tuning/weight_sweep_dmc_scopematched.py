"""
weight_sweep_dmc_scopematched.py
Same fusion-weight tuning protocol as 03_fusion_experiments/late_fusion_experiment.py
(grid search w_acoustic on SIMULATED CV only, frozen weight never sees DMC during
tuning), but evaluated against the DMC flood/fire scope-matched subset (n=11) from
04_domain_shift_and_scope/full_metrics_sim_and_dmc_scopematched.py, instead of the
full 59-call DMC set. That combination has not been run before -- the existing
weight search was only ever checked against full DMC (worse there, 35.9% at w=0.7).

Protocol (unchanged from late_fusion_experiment.py, so results are comparable):
  1. Train acoustic-only RF and Condition-C text RF separately on simulated data.
  2. Grid-search combined = w*P_acoustic + (1-w)*P_text over w in [0.0, 1.0] step 0.1,
     evaluated via 5-fold CV on SIMULATED data only, averaged over 5 seeds. DMC is
     never touched at this stage.
  3. For every weight in the grid (not just the "best" one), retrain production
     acoustic-RF and text-RF on ALL simulated data (5 seeds), predict on the n=11
     scope-matched DMC subset, and report full metrics there.
  4. Compare every weight against the existing 0.5/0.5 baseline on the same subset.

Nothing here modifies urgency_bridge.py or any deployed model file. Purely an
evaluation script, same as its two parent scripts.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
RF_BASE = dict(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1)
SEEDS = [1, 7, 42, 99, 123]
WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # weight on ACOUSTIC

ACOUSTIC = ([f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")] +
    ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
     "spec_centroid_mean", "speaking_rate", "jitter",
     "rms_slope", "rms_gradient", "speaking_rate_change"])
CONDITION_C = ["asr_quality", "keyword_count", "repetition_rate", "sentiment_polarity",
    "sin_context_score", "sin_danger_count", "sin_help_request_count",
    "sin_high_risk_phrase_count", "sin_location_context_count",
    "sin_person_at_risk_count", "sin_safety_state_count", "tam_context_score",
    "tam_danger_count", "tam_help_request_count", "tam_high_risk_phrase_count",
    "tam_location_context_count", "tam_person_at_risk_count",
    "tam_safety_state_count", "fire_smoke_match"]
assert len(ACOUSTIC) == 91 and len(CONDITION_C) == 19

# Identical scope-matching heuristic to full_metrics_sim_and_dmc_scopematched.py,
# copied rather than imported so this script stays independently runnable.
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


def fit_predict_proba(X_train, y_train, X_test, seed):
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=seed)),
        ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_test)


def full_metrics(y_true, preds_by_seed, classes, high_idx):
    accs, macros, hrs = [], [], []
    prec_rows, rec_rows = [], []
    for pred in preds_by_seed:
        accs.append(accuracy_score(y_true, pred))
        macros.append(f1_score(y_true, pred, average="macro"))
        hrs.append((pred[y_true == high_idx] == high_idx).mean() if (y_true == high_idx).any() else float("nan"))
        p, r, _, _ = precision_recall_fscore_support(y_true, pred, labels=range(len(classes)), zero_division=0)
        prec_rows.append(p); rec_rows.append(r)
    return {
        "acc": np.mean(accs), "macro_f1": np.mean(macros), "high_recall": np.mean(hrs),
        "high_prec": np.mean([p[high_idx] for p in prec_rows]),
        "prec": np.mean(prec_rows, axis=0), "rec": np.mean(rec_rows, axis=0),
    }


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    classes = list(le.classes_)
    high_idx = classes.index("High")

    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim = sim.copy()
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    y_sim = le.transform(sim["urgency_label"].values)
    X_sim_ac = sim[ACOUSTIC].fillna(0).values
    X_sim_tx = sim[CONDITION_C].fillna(0).values

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    dmc_v2 = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation_v2.csv")
    dmc = dmc.reset_index(drop=True); dmc_v2 = dmc_v2.reset_index(drop=True)
    dmc_v2 = dmc_v2.copy()
    dmc_v2["category"] = dmc_v2["transcript"].map(categorize)
    dmc = dmc.copy()
    dmc["category"] = dmc_v2["category"]
    dmc_ff = dmc[dmc["category"].isin(FLOOD_FIRE_CATEGORIES)].reset_index(drop=True)
    y_dmc = le.transform(dmc_ff["true_label"].values)
    X_dmc_ac = dmc_ff[ACOUSTIC].fillna(0).values
    X_dmc_tx = dmc_ff[CONDITION_C].fillna(0).values
    print(f"DMC flood/fire-only scope-matched subset: n={len(dmc_ff)}")
    print(f"True label counts: {dmc_ff['true_label'].value_counts().to_dict()}\n")

    # ---- Step 1: tune w on SIMULATED CV only (identical to late_fusion_experiment.py) ----
    print("=" * 90)
    print("STEP 1: grid-search w_acoustic on SIMULATED 5-fold CV only (DMC untouched)")
    print("combined = w*P_acoustic + (1-w)*P_text ; averaged over 5 seeds")
    print("=" * 90)
    proba_ac_by_seed = {s: cv_proba(X_sim_ac, y_sim, s) for s in SEEDS}
    proba_tx_by_seed = {s: cv_proba(X_sim_tx, y_sim, s) for s in SEEDS}

    grid_rows = []
    for w in WEIGHT_GRID:
        accs, macros, hrs = [], [], []
        for seed in SEEDS:
            combined = w * proba_ac_by_seed[seed] + (1 - w) * proba_tx_by_seed[seed]
            pred = combined.argmax(axis=1)
            accs.append(accuracy_score(y_sim, pred))
            macros.append(f1_score(y_sim, pred, average="macro"))
            hrs.append((pred[y_sim == high_idx] == high_idx).mean())
        grid_rows.append({"w_acoustic": w, "sim_acc": np.mean(accs),
                          "sim_macro_f1": np.mean(macros), "sim_high_recall": np.mean(hrs)})
    grid = pd.DataFrame(grid_rows)
    print(grid.round(4).to_string(index=False))
    best_by_macro = grid.loc[grid["sim_macro_f1"].idxmax(), "w_acoustic"]
    best_by_acc = grid.loc[grid["sim_acc"].idxmax(), "w_acoustic"]
    print(f"\nBest by simulated macro-F1: w_acoustic={best_by_macro}")
    print(f"Best by simulated accuracy: w_acoustic={best_by_acc}")

    # ---- Step 2: apply EVERY weight in the grid to the scope-matched DMC subset ----
    print("\n" + "=" * 90)
    print(f"STEP 2: apply each w_acoustic to DMC scope-matched subset (n={len(dmc_ff)})")
    print("(production acoustic-RF / text-RF retrained on ALL simulated data per seed,")
    print(" fused AFTER training -- subset never touched during Step 1 tuning)")
    print("=" * 90)

    proba_ac_dmc_by_seed = {s: fit_predict_proba(X_sim_ac, y_sim, X_dmc_ac, s) for s in SEEDS}
    proba_tx_dmc_by_seed = {s: fit_predict_proba(X_sim_tx, y_sim, X_dmc_tx, s) for s in SEEDS}

    rows = []
    for w in WEIGHT_GRID:
        preds = []
        for s in SEEDS:
            combined = w * proba_ac_dmc_by_seed[s] + (1 - w) * proba_tx_dmc_by_seed[s]
            preds.append(combined.argmax(axis=1))
        m = full_metrics(y_dmc, preds, classes, high_idx)
        rows.append({"w_acoustic": w, "dmc_scopematched_acc": m["acc"],
                     "dmc_scopematched_macro_f1": m["macro_f1"],
                     "dmc_scopematched_high_recall": m["high_recall"],
                     "dmc_scopematched_high_precision": m["high_prec"]})
        marker = "  <- current deployed (0.5/0.5)" if w == 0.5 else ""
        print(f"  w_acoustic={w:.1f}  acc={m['acc']:.4f}  macroF1={m['macro_f1']:.4f}  "
              f"HighRec={m['high_recall']:.4f}  HighPrec={m['high_prec']:.4f}{marker}")

    results = pd.DataFrame(rows)
    baseline = results[results["w_acoustic"] == 0.5].iloc[0]

    print("\n" + "=" * 90)
    print("SUMMARY: does any weight beat the current deployed 0.5/0.5 on the scope-matched subset?")
    print("=" * 90)
    print(f"  Baseline (w=0.5): acc={baseline.dmc_scopematched_acc:.4f}  "
          f"macroF1={baseline.dmc_scopematched_macro_f1:.4f}  "
          f"HighRec={baseline.dmc_scopematched_high_recall:.4f}")
    best_acc_row = results.loc[results["dmc_scopematched_acc"].idxmax()]
    best_macro_row = results.loc[results["dmc_scopematched_macro_f1"].idxmax()]
    print(f"  Best by scope-matched accuracy : w_acoustic={best_acc_row.w_acoustic:.1f} "
          f"-> acc={best_acc_row.dmc_scopematched_acc:.4f}")
    print(f"  Best by scope-matched macro-F1 : w_acoustic={best_macro_row.w_acoustic:.1f} "
          f"-> macroF1={best_macro_row.dmc_scopematched_macro_f1:.4f}")
    print(f"\n  n={len(dmc_ff)} -- each call = {100/len(dmc_ff):.1f} accuracy points. "
          f"Differences smaller than one call's worth are noise, not signal.")

    print(f"\nSimulated-CV weight recommendation (w={best_by_macro}) applied to scope-matched DMC:")
    sim_recommended_row = results[results["w_acoustic"] == best_by_macro].iloc[0]
    print(f"  acc={sim_recommended_row.dmc_scopematched_acc:.4f}  "
          f"macroF1={sim_recommended_row.dmc_scopematched_macro_f1:.4f}  "
          f"HighRec={sim_recommended_row.dmc_scopematched_high_recall:.4f}")


if __name__ == "__main__":
    main()
