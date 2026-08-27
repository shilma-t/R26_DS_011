"""
late_fusion_experiment.py
Tests whether the FUSION STRATEGY (early concatenation into one RF) is the
structural cause of DMC underperformance, not just which acoustic features
are included - the key remaining hypothesis after Condition C (text-only)
beat both acoustic-only and full early-fusion (v1) on DMC.

Order, per the user/research-assistant protocol:
  1. Reference: text-only (Condition C, 19 feat), acoustic-only (91 feat),
     early-fusion v1 (111 feat, current deployed feature set), and an
     apples-to-apples early-fusion (91+19=110 feat, same columns as the
     late-fusion inputs below, to isolate fusion STRATEGY from feature
     SELECTION differences).
  2. Late fusion: train the acoustic-only RF and the Condition-C text RF
     SEPARATELY on simulated data. Combine their predict_proba on DMC via
     a weighted average: combined = w*P_acoustic + (1-w)*P_text.
     Fusion weight w is tuned ONLY on simulated validation (5-fold CV proba
     from each modality-specific model, averaged over 5 tuning seeds),
     frozen, then applied once to DMC (production models retrained at 5
     validation seeds - DMC never touched during tuning).
  3. (Only reached if late fusion doesn't help) acoustic feature ablations
     A/B/C - not run in this script; separate follow-up if needed.

All disposable/in-memory - frozen v1 artifacts untouched.
"""
import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
RF_BASE = dict(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1)
SEEDS = [1, 7, 42, 99, 123]
TUNING_SEEDS = [1, 7, 42, 99, 123]
WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # w = weight on ACOUSTIC

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
FULL_V1 = ACOUSTIC + ["asr_quality", "keyword_count", "word_count", "repetition_rate",
    "sentiment_polarity"] + [c for c in CONDITION_C if c not in
    ("asr_quality", "keyword_count", "repetition_rate", "sentiment_polarity")]
EARLY_FUSION_APPLES = ACOUSTIC + CONDITION_C  # 110 feat, same columns as late fusion inputs
assert len(ACOUSTIC) == 91 and len(CONDITION_C) == 19 and len(EARLY_FUSION_APPLES) == 110


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


def multiseed_dmc(sim, dmc, y_sim, y_dmc, cols, le):
    X_sim = sim[cols].fillna(0).values
    X_dmc = dmc[cols].fillna(0).values
    accs, macros, hrs, mrs = [], [], [], []
    medium_idx = list(le.classes_).index("Medium")
    for seed in SEEDS:
        proba = fit_predict_proba(X_sim, y_sim, X_dmc, seed)
        pred = proba.argmax(axis=1)
        accs.append(accuracy_score(y_dmc, pred))
        macros.append(f1_score(y_dmc, pred, average="macro"))
        hrs.append((pred[y_dmc == 0] == 0).mean())
        mrs.append((pred[y_dmc == medium_idx] == medium_idx).mean())
    return {"accs": accs, "macros": macros, "hrs": hrs, "mrs": mrs}


def report(name, res):
    print(f"  {name:<42} acc={np.mean(res['accs']):.4f}  macroF1={np.mean(res['macros']):.4f}  "
          f"HighRec={np.mean(res['hrs']):.4f}  MedRec={np.mean(res['mrs']):.4f}")


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim = sim.copy()
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    y_sim = le.transform(sim["urgency_label"].values)

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    y_dmc = le.transform(dmc["true_label"].values)
    medium_idx = list(le.classes_).index("Medium")

    print("=" * 78)
    print("STEP 1: references - text-only, acoustic-only, early-fusion v1, "
          "early-fusion apples-to-apples (91+19=110)")
    print("=" * 78)
    res_text = multiseed_dmc(sim, dmc, y_sim, y_dmc, CONDITION_C, le)
    res_acoustic = multiseed_dmc(sim, dmc, y_sim, y_dmc, ACOUSTIC, le)
    res_v1 = multiseed_dmc(sim, dmc, y_sim, y_dmc, FULL_V1, le)
    res_early_apples = multiseed_dmc(sim, dmc, y_sim, y_dmc, EARLY_FUSION_APPLES, le)

    report("Text-only (Condition C, 19)", res_text)
    report("Acoustic-only (91)", res_acoustic)
    report("Early-fusion v1 (111, current)", res_v1)
    report("Early-fusion apples-to-apples (110)", res_early_apples)

    print("\n" + "=" * 78)
    print("STEP 2a: tune fusion weight w (on ACOUSTIC proba) using SIMULATED CV only")
    print("combined = w*P_acoustic + (1-w)*P_text ; averaged over 5 tuning seeds")
    print("=" * 78)
    X_sim_ac = sim[ACOUSTIC].fillna(0).values
    X_sim_tx = sim[CONDITION_C].fillna(0).values

    proba_ac_by_seed = {s: cv_proba(X_sim_ac, y_sim, s) for s in TUNING_SEEDS}
    proba_tx_by_seed = {s: cv_proba(X_sim_tx, y_sim, s) for s in TUNING_SEEDS}

    grid_rows = []
    for w in WEIGHT_GRID:
        accs, macros, hrs = [], [], []
        for seed in TUNING_SEEDS:
            combined = w * proba_ac_by_seed[seed] + (1 - w) * proba_tx_by_seed[seed]
            pred = combined.argmax(axis=1)
            accs.append(accuracy_score(y_sim, pred))
            macros.append(f1_score(y_sim, pred, average="macro"))
            hrs.append((pred[y_sim == 0] == 0).mean())
        grid_rows.append({"w_acoustic": w, "mean_acc": np.mean(accs),
                          "mean_macro_f1": np.mean(macros), "mean_high_recall": np.mean(hrs)})
    grid = pd.DataFrame(grid_rows)
    print(grid.round(4).to_string(index=False))

    best = grid.loc[grid["mean_macro_f1"].idxmax()]
    FROZEN_W = float(best.w_acoustic)
    print(f"\nBest fusion weight (by simulated CV macro-F1): w_acoustic={FROZEN_W}  "
          f"(1-w={1-FROZEN_W:.1f} on text)")
    print(f"  acc={best.mean_acc:.4f}  macro_f1={best.mean_macro_f1:.4f}  "
          f"high_recall={best.mean_high_recall:.4f}")

    print("\n" + "=" * 78)
    print(f"STEP 2b: apply FROZEN w_acoustic={FROZEN_W} to DMC")
    print("(production acoustic-RF and text-RF retrained separately at 5 seeds each,")
    print(" probabilities combined AFTER both are trained - DMC untouched during tuning)")
    print("=" * 78)

    X_dmc_ac = dmc[ACOUSTIC].fillna(0).values
    X_dmc_tx = dmc[CONDITION_C].fillna(0).values

    unweighted_accs, tuned_accs = [], []
    unweighted_hrs, tuned_hrs = [], []
    unweighted_mrs, tuned_mrs = [], []
    unweighted_macros, tuned_macros = [], []
    last_tuned_pred = None
    for seed in SEEDS:
        proba_ac_dmc = fit_predict_proba(X_sim_ac, y_sim, X_dmc_ac, seed)
        proba_tx_dmc = fit_predict_proba(X_sim_tx, y_sim, X_dmc_tx, seed)

        unweighted = 0.5 * proba_ac_dmc + 0.5 * proba_tx_dmc
        tuned = FROZEN_W * proba_ac_dmc + (1 - FROZEN_W) * proba_tx_dmc

        pred_uw = unweighted.argmax(axis=1)
        pred_tn = tuned.argmax(axis=1)
        last_tuned_pred = pred_tn

        unweighted_accs.append(accuracy_score(y_dmc, pred_uw))
        tuned_accs.append(accuracy_score(y_dmc, pred_tn))
        unweighted_hrs.append((pred_uw[y_dmc == 0] == 0).mean())
        tuned_hrs.append((pred_tn[y_dmc == 0] == 0).mean())
        unweighted_mrs.append((pred_uw[y_dmc == medium_idx] == medium_idx).mean())
        tuned_mrs.append((pred_tn[y_dmc == medium_idx] == medium_idx).mean())
        unweighted_macros.append(f1_score(y_dmc, pred_uw, average="macro"))
        tuned_macros.append(f1_score(y_dmc, pred_tn, average="macro"))

        print(f"  seed={seed:<4} unweighted(0.5/0.5): acc={unweighted_accs[-1]:.4f} "
              f"macroF1={unweighted_macros[-1]:.4f} HR={unweighted_hrs[-1]:.4f}   "
              f"tuned(w={FROZEN_W}): acc={tuned_accs[-1]:.4f} macroF1={tuned_macros[-1]:.4f} "
              f"HR={tuned_hrs[-1]:.4f}")

    print(f"\n  MEAN unweighted (0.5/0.5): acc={np.mean(unweighted_accs):.4f}  "
          f"macroF1={np.mean(unweighted_macros):.4f}  HR={np.mean(unweighted_hrs):.4f}  "
          f"MedR={np.mean(unweighted_mrs):.4f}")
    print(f"  MEAN tuned (w={FROZEN_W})    : acc={np.mean(tuned_accs):.4f}  "
          f"macroF1={np.mean(tuned_macros):.4f}  HR={np.mean(tuned_hrs):.4f}  "
          f"MedR={np.mean(tuned_mrs):.4f}")

    print(f"\nConfusion matrix, tuned late fusion, seed={SEEDS[-1]}:")
    cm = confusion_matrix(y_dmc, last_tuned_pred, labels=le.transform(le.classes_))
    print("classes:", list(le.classes_))
    print(cm)
    print(classification_report(y_dmc, last_tuned_pred, target_names=le.classes_, digits=3))

    print("\n" + "=" * 78)
    print("FINAL SUMMARY - all conditions, DMC 59 calls, mean over 5 seeds")
    print("=" * 78)
    print(f"  {'condition':<42}{'acc':>8}{'macroF1':>10}{'HighRec':>10}{'MedRec':>10}")
    def line(name, accs, macros, hrs, mrs):
        print(f"  {name:<42}{np.mean(accs):>8.4f}{np.mean(macros):>10.4f}"
              f"{np.mean(hrs):>10.4f}{np.mean(mrs):>10.4f}")
    line("Text-only (Condition C, 19)", res_text["accs"], res_text["macros"], res_text["hrs"], res_text["mrs"])
    line("Acoustic-only (91)", res_acoustic["accs"], res_acoustic["macros"], res_acoustic["hrs"], res_acoustic["mrs"])
    line("Early-fusion v1 (111, current)", res_v1["accs"], res_v1["macros"], res_v1["hrs"], res_v1["mrs"])
    line("Early-fusion apples-to-apples (110)", res_early_apples["accs"], res_early_apples["macros"], res_early_apples["hrs"], res_early_apples["mrs"])
    line("Late fusion unweighted (0.5/0.5)", unweighted_accs, unweighted_macros, unweighted_hrs, unweighted_mrs)
    line(f"Late fusion TUNED (w_acoustic={FROZEN_W})", tuned_accs, tuned_macros, tuned_hrs, tuned_mrs)

    print("\nDoes late fusion beat text-only alone? (win counts across 5 seeds)")
    acc_w = sum(1 for a, b in zip(tuned_accs, res_text["accs"]) if a >= b)
    hr_w = sum(1 for a, b in zip(tuned_hrs, res_text["hrs"]) if a >= b)
    macro_w = sum(1 for a, b in zip(tuned_macros, res_text["macros"]) if a >= b)
    print(f"  tuned late fusion >= text-only on acc    : {acc_w}/5")
    print(f"  tuned late fusion >= text-only on macroF1: {macro_w}/5")
    print(f"  tuned late fusion >= text-only on HighRec: {hr_w}/5")


if __name__ == "__main__":
    main()
