"""
stability_and_override_check.py
Two checks requested before treating the (w_high=1.2, w_low=0.9) decision
rule as final:

  1. STABILITY: is that weight pair a robust choice, or did it just win on
     one lucky seed? Re-runs the simulated-CV weight search across 5 seeds,
     AND retrains the production RF (fit on all 267 simulated calls) across
     those same 5 seeds, applying the frozen weights to the ALREADY-SAVED
     DMC feature vectors each time (no ASR re-run - that part is untouched
     and unchanged).

  2. MODEL_OVERRIDE cases: of the 10 calls where the earlier error analysis
     found good signal but a wrong prediction anyway, how many does the new
     weighting actually fix?
"""
import sys
import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

_PIPE = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011\scripts\pipeline"
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)
from text_features_from_transcript import fire_smoke_match

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
RF_BASE = dict(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1)

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
WEIGHT_GRID_LOW = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEEDS = [1, 7, 42, 99, 123]
FROZEN_W_HIGH, FROZEN_W_LOW = 1.2, 0.9


def apply_weights(proba, w_high, w_low, w_medium=1.0):
    return (proba * np.array([w_high, w_low, w_medium])).argmax(axis=1)


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


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])

    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    X_sim = sim[FEATURE_COLS].fillna(0).values
    y_sim = le.transform(sim["urgency_label"].values)

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    X_dmc = dmc[FEATURE_COLS].fillna(0).values
    y_dmc = le.transform(dmc["true_label"].values)

    # ---- CHECK 1a: does (1.2, 0.9) keep winning on simulated CV across seeds? ----
    print("=" * 78)
    print("CHECK 1a: best-macro-F1 weight pair across 5 tuning seeds (simulated CV)")
    print("=" * 78)
    winners = []
    for seed in SEEDS:
        proba_cv = cv_predict_proba(X_sim, y_sim, seed)
        rows = []
        for w_h, w_l in product(WEIGHT_GRID_HIGH, WEIGHT_GRID_LOW):
            pred = apply_weights(proba_cv, w_h, w_l)
            rows.append({"w_high": w_h, "w_low": w_l,
                        "macro_f1": f1_score(y_sim, pred, average="macro")})
        grid = pd.DataFrame(rows)
        best = grid.loc[grid["macro_f1"].idxmax()]
        winners.append((seed, best.w_high, best.w_low, best.macro_f1))
        print(f"  seed={seed:<4} winner: w_high={best.w_high}  w_low={best.w_low}  "
              f"macro_f1={best.macro_f1:.4f}")

    # ---- CHECK 1b: apply FROZEN (1.2, 0.9) to DMC using RF retrained at 5 seeds ----
    print("\n" + "=" * 78)
    print("CHECK 1b: frozen (w_high=1.2, w_low=0.9) applied to DMC, RF retrained per seed")
    print("(DMC features/ASR untouched - only the RF's random_state varies)")
    print("=" * 78)
    baseline_accs, tuned_accs = [], []
    baseline_hr, tuned_hr = [], []
    for seed in SEEDS:
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=seed)),
            ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
        ])
        pipe.fit(X_sim, y_sim)
        proba_dmc = pipe.predict_proba(X_dmc)

        base_pred = proba_dmc.argmax(axis=1)
        tuned_pred = apply_weights(proba_dmc, FROZEN_W_HIGH, FROZEN_W_LOW)

        b_acc = accuracy_score(y_dmc, base_pred)
        t_acc = accuracy_score(y_dmc, tuned_pred)
        b_hr = (base_pred[y_dmc == 0] == 0).mean()
        t_hr = (tuned_pred[y_dmc == 0] == 0).mean()
        baseline_accs.append(b_acc); tuned_accs.append(t_acc)
        baseline_hr.append(b_hr); tuned_hr.append(t_hr)

        print(f"  seed={seed:<4} baseline acc={b_acc:.4f} HR={b_hr:.4f}   "
              f"-> tuned acc={t_acc:.4f} HR={t_hr:.4f}   "
              f"delta_acc={t_acc-b_acc:+.4f}  delta_HR={t_hr-b_hr:+.4f}")

    print(f"\n  mean baseline accuracy : {np.mean(baseline_accs):.4f} (+/- {np.std(baseline_accs):.4f})")
    print(f"  mean tuned accuracy    : {np.mean(tuned_accs):.4f} (+/- {np.std(tuned_accs):.4f})")
    print(f"  mean baseline HighRecall: {np.mean(baseline_hr):.4f} (+/- {np.std(baseline_hr):.4f})")
    print(f"  mean tuned HighRecall   : {np.mean(tuned_hr):.4f} (+/- {np.std(tuned_hr):.4f})")
    improved_count = sum(1 for b, t in zip(baseline_accs, tuned_accs) if t >= b)
    print(f"  tuned >= baseline accuracy in {improved_count}/{len(SEEDS)} seeds")

    # ---- CHECK 2: the 10 MODEL_OVERRIDE cases ----
    print("\n" + "=" * 78)
    print("CHECK 2: the 10 MODEL_OVERRIDE cases from the error analysis")
    print("=" * 78)
    err = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_error_analysis.csv")
    overrides = err[err["category"] == "MODEL_OVERRIDE"]["recording_id"].tolist()
    print(f"  {len(overrides)} MODEL_OVERRIDE cases: {overrides}\n")

    dmc_idx = dmc.set_index("recording_id")
    fixed, still_wrong = 0, 0
    for rec_id in overrides:
        if rec_id not in dmc_idx.index:
            print(f"  SKIP {rec_id}: not found"); continue
        row = dmc_idx.loc[rec_id]
        proba = row[["prob_High", "prob_Low", "prob_Medium"]].values.astype(float)
        old_pred = row["predicted_label"]
        true_label = row["true_label"]

        new_idx = int((proba * np.array([FROZEN_W_HIGH, FROZEN_W_LOW, 1.0])).argmax())
        new_pred = ["High", "Low", "Medium"][new_idx]

        status = "FIXED" if new_pred == true_label else "still wrong"
        if status == "FIXED":
            fixed += 1
        else:
            still_wrong += 1
        print(f"  {rec_id[:32]:<32} true={true_label:<7} old={old_pred:<7} new={new_pred:<7} [{status}]")

    print(f"\n  {fixed} / {len(overrides)} MODEL_OVERRIDE cases fixed by the new weighting")
    print(f"  {still_wrong} / {len(overrides)} remain wrong (a genuine model/feature weakness, "
          f"not fixable by decision-rule tuning alone)")


if __name__ == "__main__":
    main()
