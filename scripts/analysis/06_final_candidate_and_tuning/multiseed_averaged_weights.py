"""
multiseed_averaged_weights.py
Fix for the instability found earlier: select the weight pair by AVERAGING
macro-F1 across 5 tuning seeds (not trusting a single seed's winner), on
UNCALIBRATED probabilities (calibration was tested and made things worse
at this sample size - see calibrated_threshold_tuning.py results).

Then validate the frozen pair against DMC using RF models retrained at
5 different seeds on the full simulated corpus, applied to the ALREADY-
SAVED DMC feature vectors (no ASR re-run).
"""
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
VALIDATION_SEEDS = [1, 7, 42, 99, 123]  # separate use: retrain production model per seed


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
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    X_sim = sim[FEATURE_COLS].fillna(0).values
    y_sim = le.transform(sim["urgency_label"].values)

    print("=" * 78)
    print("STEP 1: select weight pair by AVERAGE macro-F1 across 5 tuning seeds")
    print("(uncalibrated - calibration tested worse at this sample size)")
    print("=" * 78)

    proba_by_seed = {seed: cv_proba(X_sim, y_sim, seed) for seed in TUNING_SEEDS}

    rows = []
    for w_h, w_l in product(WEIGHT_GRID_HIGH, WEIGHT_GRID_LOW):
        accs, macros, hrs = [], [], []
        for seed in TUNING_SEEDS:
            pred = apply_weights(proba_by_seed[seed], w_h, w_l)
            accs.append(accuracy_score(y_sim, pred))
            macros.append(f1_score(y_sim, pred, average="macro"))
            hrs.append((pred[y_sim == 0] == 0).mean())
        rows.append({"w_high": w_h, "w_low": w_l,
                    "mean_acc": np.mean(accs), "mean_macro_f1": np.mean(macros),
                    "std_macro_f1": np.std(macros),
                    "mean_high_recall": np.mean(hrs)})
    grid = pd.DataFrame(rows)

    baseline = grid[(grid.w_high == 1.0) & (grid.w_low == 1.0)].iloc[0]
    best = grid.loc[grid["mean_macro_f1"].idxmax()]

    print(f"\nBaseline (w=1,1), averaged over {len(TUNING_SEEDS)} seeds:")
    print(f"  acc={baseline.mean_acc:.4f}  macro_f1={baseline.mean_macro_f1:.4f}  "
          f"high_recall={baseline.mean_high_recall:.4f}")
    print(f"\nBest AVERAGED weight pair: w_high={best.w_high}  w_low={best.w_low}")
    print(f"  acc={best.mean_acc:.4f}  macro_f1={best.mean_macro_f1:.4f} "
          f"(+/-{best.std_macro_f1:.4f} across seeds)  high_recall={best.mean_high_recall:.4f}")
    print(f"  delta vs baseline: acc={best.mean_acc-baseline.mean_acc:+.4f}  "
          f"macro_f1={best.mean_macro_f1-baseline.mean_macro_f1:+.4f}  "
          f"HR={best.mean_high_recall-baseline.mean_high_recall:+.4f}")

    # Show the top 5 by average macro-F1, so we can see how flat/peaked the
    # landscape is - a robust choice should have close competitors nearby,
    # not one isolated spike.
    print("\nTop 5 weight pairs by average macro-F1:")
    print(grid.sort_values("mean_macro_f1", ascending=False).head(5).to_string(index=False))

    FROZEN_W_HIGH, FROZEN_W_LOW = float(best.w_high), float(best.w_low)
    grid.to_csv(f"{ROOT}\\reports\\pp2\\multiseed_weight_grid.csv", index=False)

    # ---- STEP 2: validate the frozen pair against DMC across multiple ------
    # production-model training seeds (DMC features/ASR untouched) ----------
    print("\n" + "=" * 78)
    print(f"STEP 2: apply FROZEN (w_high={FROZEN_W_HIGH}, w_low={FROZEN_W_LOW}) to DMC")
    print("RF retrained at 5 seeds on full simulated corpus; DMC features unchanged")
    print("=" * 78)

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    X_dmc = dmc[FEATURE_COLS].fillna(0).values
    y_dmc = le.transform(dmc["true_label"].values)

    base_accs, tuned_accs, base_hrs, tuned_hrs, base_macros, tuned_macros = [], [], [], [], [], []
    last_tuned_pred = None
    for seed in VALIDATION_SEEDS:
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=seed)),
            ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
        ])
        pipe.fit(X_sim, y_sim)
        proba_dmc = pipe.predict_proba(X_dmc)

        base_pred = proba_dmc.argmax(axis=1)
        tuned_pred = apply_weights(proba_dmc, FROZEN_W_HIGH, FROZEN_W_LOW)
        last_tuned_pred = tuned_pred

        b_acc, t_acc = accuracy_score(y_dmc, base_pred), accuracy_score(y_dmc, tuned_pred)
        b_hr = (base_pred[y_dmc == 0] == 0).mean()
        t_hr = (tuned_pred[y_dmc == 0] == 0).mean()
        b_macro = f1_score(y_dmc, base_pred, average="macro")
        t_macro = f1_score(y_dmc, tuned_pred, average="macro")

        base_accs.append(b_acc); tuned_accs.append(t_acc)
        base_hrs.append(b_hr); tuned_hrs.append(t_hr)
        base_macros.append(b_macro); tuned_macros.append(t_macro)

        print(f"  seed={seed:<4} baseline: acc={b_acc:.4f} macroF1={b_macro:.4f} HR={b_hr:.4f}"
              f"   ->  tuned: acc={t_acc:.4f} macroF1={t_macro:.4f} HR={t_hr:.4f}")

    print(f"\n  MEAN baseline: acc={np.mean(base_accs):.4f} macroF1={np.mean(base_macros):.4f} "
          f"HR={np.mean(base_hrs):.4f}")
    print(f"  MEAN tuned   : acc={np.mean(tuned_accs):.4f} macroF1={np.mean(tuned_macros):.4f} "
          f"HR={np.mean(tuned_hrs):.4f}")
    print(f"  acc improved in {sum(t>=b for t,b in zip(tuned_accs,base_accs))}/{len(VALIDATION_SEEDS)} seeds")
    print(f"  macroF1 improved in {sum(t>=b for t,b in zip(tuned_macros,base_macros))}/{len(VALIDATION_SEEDS)} seeds")
    print(f"  HR improved in {sum(t>=b for t,b in zip(tuned_hrs,base_hrs))}/{len(VALIDATION_SEEDS)} seeds")

    print(f"\nConfusion matrix, tuned rule, LAST seed ({VALIDATION_SEEDS[-1]}) as example:")
    cm = confusion_matrix(y_dmc, last_tuned_pred, labels=le.transform(le.classes_))
    print("classes:", list(le.classes_))
    print(cm)
    print(classification_report(y_dmc, last_tuned_pred, target_names=le.classes_, digits=3))


if __name__ == "__main__":
    main()
