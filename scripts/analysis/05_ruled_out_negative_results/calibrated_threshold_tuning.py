"""
calibrated_threshold_tuning.py
Step 1: does calibrating predict_proba (Platt/sigmoid) stabilize the
weight-selection noise found in stability_and_override_check.py?

Step 2: re-tune the decision rule on CALIBRATED probabilities, using an
explicit constrained objective (maximize accuracy subject to High recall
>= r) instead of "best macro-F1 on one seed", and select by AVERAGING
across multiple tuning seeds rather than trusting a single CV split.

Everything here runs on the SIMULATED corpus only. DMC is not touched.
Only if this shows a genuine, stable improvement does a DMC check happen,
as a separate, deliberate next step - not automatically here.
"""
import sys
import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss
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

WEIGHT_GRID_HIGH = [1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0]
WEIGHT_GRID_LOW = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TUNING_SEEDS = [1, 7, 42, 99, 123]
MIN_HIGH_RECALL = 0.60  # the explicit constraint: don't accept below this


def apply_weights(proba, w_high, w_low, w_medium=1.0):
    return (proba * np.array([w_high, w_low, w_medium])).argmax(axis=1)


def cv_proba(X, y, seed, calibrate):
    """5-fold CV predict_proba, optionally with Platt/sigmoid calibration
    fit on the training fold only (never on the held-out fold)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out = np.zeros((len(y), 3))
    for train_idx, test_idx in skf.split(X, y):
        base = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=seed)),
            ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
        ])
        if calibrate:
            # Sigmoid (Platt), not isotonic - isotonic's flexible fit needs
            # more samples per bin than 267 calls comfortably provides.
            model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        else:
            model = base
        model.fit(X[train_idx], y[train_idx])
        out[test_idx] = model.predict_proba(X[test_idx])
    return out


def brier_multiclass(y_true_onehot, proba):
    return np.mean(np.sum((proba - y_true_onehot) ** 2, axis=1))


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    X = sim[FEATURE_COLS].fillna(0).values
    y = le.transform(sim["urgency_label"].values)
    y_onehot = np.eye(3)[y]

    print("=" * 78)
    print("STEP 1: does calibration change reliability of predict_proba?")
    print("=" * 78)
    for seed in TUNING_SEEDS:
        proba_raw = cv_proba(X, y, seed, calibrate=False)
        proba_cal = cv_proba(X, y, seed, calibrate=True)
        b_raw = brier_multiclass(y_onehot, proba_raw)
        b_cal = brier_multiclass(y_onehot, proba_cal)
        acc_raw = accuracy_score(y, proba_raw.argmax(axis=1))
        acc_cal = accuracy_score(y, proba_cal.argmax(axis=1))
        print(f"  seed={seed:<4} Brier raw={b_raw:.4f} cal={b_cal:.4f} "
              f"(lower=better calibrated)   argmax acc raw={acc_raw:.4f} cal={acc_cal:.4f}")

    print("\n" + "=" * 78)
    print(f"STEP 2: weight search on CALIBRATED proba, constraint High recall >= {MIN_HIGH_RECALL}")
    print("Selecting by AVERAGE performance across all 5 tuning seeds, not one seed")
    print("=" * 78)

    # Collect calibrated CV proba for every seed once
    proba_by_seed = {seed: cv_proba(X, y, seed, calibrate=True) for seed in TUNING_SEEDS}

    grid_rows = []
    for w_h, w_l in product(WEIGHT_GRID_HIGH, WEIGHT_GRID_LOW):
        accs, macros, hrs = [], [], []
        for seed in TUNING_SEEDS:
            pred = apply_weights(proba_by_seed[seed], w_h, w_l)
            accs.append(accuracy_score(y, pred))
            macros.append(f1_score(y, pred, average="macro"))
            hrs.append((pred[y == 0] == 0).mean())
        grid_rows.append({
            "w_high": w_h, "w_low": w_l,
            "mean_acc": np.mean(accs), "std_acc": np.std(accs),
            "mean_macro_f1": np.mean(macros),
            "mean_high_recall": np.mean(hrs), "std_high_recall": np.std(hrs),
        })
    grid = pd.DataFrame(grid_rows)

    baseline = grid[(grid.w_high == 1.0) & (grid.w_low == 1.0)].iloc[0]
    print(f"\nBaseline (w=1,1) averaged over {len(TUNING_SEEDS)} seeds: "
          f"acc={baseline.mean_acc:.4f}  macro_f1={baseline.mean_macro_f1:.4f}  "
          f"high_recall={baseline.mean_high_recall:.4f}")

    feasible = grid[grid["mean_high_recall"] >= MIN_HIGH_RECALL]
    if feasible.empty:
        print(f"\nNo weight pair reaches mean High recall >= {MIN_HIGH_RECALL} "
              f"consistently. Lower the constraint or reconsider.")
        return

    best = feasible.loc[feasible["mean_acc"].idxmax()]
    print(f"\nBest under constraint: w_high={best.w_high} w_low={best.w_low}")
    print(f"  mean_acc={best.mean_acc:.4f} (+/-{best.std_acc:.4f})  "
          f"mean_macro_f1={best.mean_macro_f1:.4f}  "
          f"mean_high_recall={best.mean_high_recall:.4f} (+/-{best.std_high_recall:.4f})")
    print(f"  vs baseline: delta_acc={best.mean_acc-baseline.mean_acc:+.4f}  "
          f"delta_HR={best.mean_high_recall-baseline.mean_high_recall:+.4f}")

    grid.to_csv(f"{ROOT}\\reports\\pp2\\calibrated_weight_grid.csv", index=False)
    print(f"\nFull grid saved -> reports/pp2/calibrated_weight_grid.csv")
    print("\nDMC has NOT been touched. This is the simulated-only result.")


if __name__ == "__main__":
    main()
