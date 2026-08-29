"""
late_fusion_threshold_tuning_scopematched.py
Applies the established decision-threshold tuning protocol (see
condition_c_threshold_tuning.py) to the FINAL model -- late fusion (0.5/0.5,
default RF params) -- instead of Condition-C text-only, and evaluates against
the DMC flood/fire scope-matched subset (n=11) instead of full DMC. Neither
combination (threshold tuning x late-fusion x scope-matched) has been run
before; this is the last cheap check in the "improve via tuning" search.

Protocol (identical to condition_c_threshold_tuning.py):
  1. Compute late-fusion probabilities (0.5*P_acoustic + 0.5*P_text) via 5-fold
     CV on SIMULATED data, for every (w_high, w_low) pair in the grid.
  2. Pick the pair with the best AVERAGE macro-F1 across 5 tuning seeds.
     Freeze it. DMC is never touched during this step.
  3. Apply the frozen weights exactly once to the scope-matched DMC subset,
     production acoustic-RF and text-RF retrained (default params) at 5
     validation seeds, late-fused, THEN threshold-weighted.
  4. Report accuracy / macro-F1 / High recall / High precision, baseline
     (w=1,1, i.e. plain argmax on the fused proba) vs tuned, side by side,
     against the already-established 0.5/0.5 default baseline (67.3% acc).
"""
import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
RF_BASE = dict(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1)
SEEDS = [1, 7, 42, 99, 123]

WEIGHT_GRID_HIGH = [1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0]
WEIGHT_GRID_LOW = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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


def fit_predict_proba(X_train, y_train, X_test, seed):
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=seed)),
        ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_test)


def metrics(y_true, pred, high_idx):
    acc = accuracy_score(y_true, pred)
    macro = f1_score(y_true, pred, average="macro")
    p, r, _, _ = precision_recall_fscore_support(y_true, pred, labels=[0, 1, 2], zero_division=0)
    return acc, macro, r[high_idx], p[high_idx]


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
    print(f"DMC flood/fire-only scope-matched subset: n={len(dmc_ff)}\n")

    # ---- Step 1: tune (w_high, w_low) on late-fusion SIMULATED CV proba only ----
    print("=" * 90)
    print("STEP 1: grid-search (w_high, w_low) on late-fusion SIMULATED CV proba (DMC untouched)")
    print("=" * 90)
    proba_ac_by_seed = {s: cv_proba(X_sim_ac, y_sim, s) for s in SEEDS}
    proba_tx_by_seed = {s: cv_proba(X_sim_tx, y_sim, s) for s in SEEDS}
    fused_by_seed = {s: 0.5 * proba_ac_by_seed[s] + 0.5 * proba_tx_by_seed[s] for s in SEEDS}

    rows = []
    for w_h, w_l in product(WEIGHT_GRID_HIGH, WEIGHT_GRID_LOW):
        accs, macros, hrs = [], [], []
        for seed in SEEDS:
            pred = apply_weights(fused_by_seed[seed], w_h, w_l)
            accs.append(accuracy_score(y_sim, pred))
            macros.append(f1_score(y_sim, pred, average="macro"))
            hrs.append((pred[y_sim == high_idx] == high_idx).mean())
        rows.append({"w_high": w_h, "w_low": w_l, "sim_acc": np.mean(accs),
                    "sim_macro_f1": np.mean(macros), "sim_high_recall": np.mean(hrs)})
    grid = pd.DataFrame(rows)
    baseline_row = grid[(grid.w_high == 1.0) & (grid.w_low == 1.0)].iloc[0]
    best = grid.loc[grid["sim_macro_f1"].idxmax()]
    print(f"Baseline (w=1,1, plain argmax): sim_acc={baseline_row.sim_acc:.4f}  "
          f"sim_macroF1={baseline_row.sim_macro_f1:.4f}")
    print(f"Best by SIMULATED macro-F1: w_high={best.w_high}  w_low={best.w_low}  "
          f"sim_acc={best.sim_acc:.4f}  sim_macroF1={best.sim_macro_f1:.4f}")
    print("\nTop 5 pairs on simulated data:")
    print(grid.sort_values("sim_macro_f1", ascending=False).head(5).to_string(index=False))

    FROZEN_W_HIGH, FROZEN_W_LOW = float(best.w_high), float(best.w_low)

    # ---- Step 2: apply frozen weights once to scope-matched DMC ----
    print("\n" + "=" * 90)
    print(f"STEP 2: apply FROZEN (w_high={FROZEN_W_HIGH}, w_low={FROZEN_W_LOW}) "
          f"to scope-matched DMC (n={len(dmc_ff)})")
    print("=" * 90)

    base_accs, tuned_accs = [], []
    base_macros, tuned_macros = [], []
    base_hrs, tuned_hrs = [], []
    base_hps, tuned_hps = [], []
    for seed in SEEDS:
        proba_ac_dmc = fit_predict_proba(X_sim_ac, y_sim, X_dmc_ac, seed)
        proba_tx_dmc = fit_predict_proba(X_sim_tx, y_sim, X_dmc_tx, seed)
        fused_dmc = 0.5 * proba_ac_dmc + 0.5 * proba_tx_dmc

        base_pred = fused_dmc.argmax(axis=1)
        tuned_pred = apply_weights(fused_dmc, FROZEN_W_HIGH, FROZEN_W_LOW)

        b_acc, b_macro, b_hr, b_hp = metrics(y_dmc, base_pred, high_idx)
        t_acc, t_macro, t_hr, t_hp = metrics(y_dmc, tuned_pred, high_idx)
        base_accs.append(b_acc); tuned_accs.append(t_acc)
        base_macros.append(b_macro); tuned_macros.append(t_macro)
        base_hrs.append(b_hr); tuned_hrs.append(t_hr)
        base_hps.append(b_hp); tuned_hps.append(t_hp)

        print(f"  seed={seed:<4} baseline(0.5/0.5, argmax): acc={b_acc:.4f} macroF1={b_macro:.4f} "
              f"HR={b_hr:.4f} HP={b_hp:.4f}   ->  tuned(w_h={FROZEN_W_HIGH},w_l={FROZEN_W_LOW}): "
              f"acc={t_acc:.4f} macroF1={t_macro:.4f} HR={t_hr:.4f} HP={t_hp:.4f}")

    print(f"\n  MEAN baseline (0.5/0.5 default): acc={np.mean(base_accs):.4f}  "
          f"macroF1={np.mean(base_macros):.4f}  HighRec={np.mean(base_hrs):.4f}  "
          f"HighPrec={np.mean(base_hps):.4f}")
    print(f"  MEAN tuned                     : acc={np.mean(tuned_accs):.4f}  "
          f"macroF1={np.mean(tuned_macros):.4f}  HighRec={np.mean(tuned_hrs):.4f}  "
          f"HighPrec={np.mean(tuned_hps):.4f}")
    print(f"\n  (reference, already established: default late fusion on this exact subset "
          f"= acc 0.6727, macroF1 0.6303, HighRec 0.6571, HighPrec 1.0000)")
    print(f"  n={len(dmc_ff)} -- each call = {100/len(dmc_ff):.1f} accuracy points.")


if __name__ == "__main__":
    main()
