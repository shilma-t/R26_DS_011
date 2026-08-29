"""
tuned_late_fusion_simulated_cv.py
Closes a gap: the HP-tuned late fusion config (tuned text-RF + tuned
acoustic-RF, fused 0.5/0.5) was only ever evaluated directly on DMC n=11
(tuned_models_dmc_scopematched.py). This runs the SAME tuned hyperparameters
through 5-fold CV on the SIMULATED corpus, so the simulated-side numbers
exist for comparison -- same methodology (StratifiedKFold, SMOTE, 5 seeds)
as every other simulated-CV report in this project.

Hyperparameters are copied verbatim from tuned_models_dmc_scopematched.py
(TUNED_TEXT, TUNED_ACOUSTIC) -- not re-tuned here, just re-evaluated on a
different data split to fill in the missing comparison point.
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
SEEDS = [1, 7, 42, 99, 123]

TUNED_TEXT = dict(n_estimators=300, max_depth=None, min_samples_leaf=2, max_features="log2")
TUNED_ACOUSTIC = dict(n_estimators=500, max_depth=30, min_samples_leaf=1, max_features="sqrt")
DEFAULT_PARAMS = dict(n_estimators=200, max_depth=None, min_samples_leaf=2, max_features="sqrt")

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


def fire_smoke_match(text):
    kws = ["fire","smoke","burning","flames","gas","ගිනිගන්නව","ගිනිඅරගෙන",
           "ගින්නක්","ගිනි","දුමක්","දුම","தீ","புகை","gini"]
    if not isinstance(text, str): return 0
    t = text.lower()
    return int(any(k.lower() in t for k in kws))


def fit_pipe(seed, params):
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=seed)),
        ("clf", RandomForestClassifier(random_state=seed, class_weight="balanced",
                                       n_jobs=-1, **params)),
    ])


def cv_proba(X, y, seed, params):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out = np.zeros((len(y), 3))
    for train_idx, test_idx in skf.split(X, y):
        pipe = fit_pipe(seed, params)
        pipe.fit(X[train_idx], y[train_idx])
        out[test_idx] = pipe.predict_proba(X[test_idx])
    return out


def report(name, y_true, preds_by_seed, classes, high_idx):
    accs, macros = [], []
    rec_rows = []
    for pred in preds_by_seed:
        accs.append(accuracy_score(y_true, pred))
        macros.append(f1_score(y_true, pred, average="macro"))
        _, r, _, _ = precision_recall_fscore_support(y_true, pred, labels=[0, 1, 2], zero_division=0)
        rec_rows.append(r)
    rec = np.mean(rec_rows, axis=0)
    print(f"  {name:<42} acc={np.mean(accs):.4f}  macroF1={np.mean(macros):.4f}  "
          f"HighRec={rec[high_idx]:.4f}  LowRec={rec[1]:.4f}  MedRec={rec[2]:.4f}")
    return np.mean(accs), np.mean(macros), rec


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    classes = list(le.classes_)
    high_idx = classes.index("High")

    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim = sim.copy()
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    y_sim = le.transform(sim["urgency_label"].values)
    X_sim_tx = sim[CONDITION_C].fillna(0).values
    X_sim_ac = sim[ACOUSTIC].fillna(0).values

    print("=" * 90)
    print("SIMULATED 5-fold CV -- late fusion (0.5/0.5), default vs HP-tuned")
    print("=" * 90)

    default_preds, tuned_preds = [], []
    for seed in SEEDS:
        proba_tx_def = cv_proba(X_sim_tx, y_sim, seed, DEFAULT_PARAMS)
        proba_ac_def = cv_proba(X_sim_ac, y_sim, seed, DEFAULT_PARAMS)
        default_preds.append((0.5 * proba_tx_def + 0.5 * proba_ac_def).argmax(axis=1))

        proba_tx_tuned = cv_proba(X_sim_tx, y_sim, seed, TUNED_TEXT)
        proba_ac_tuned = cv_proba(X_sim_ac, y_sim, seed, TUNED_ACOUSTIC)
        tuned_preds.append((0.5 * proba_tx_tuned + 0.5 * proba_ac_tuned).argmax(axis=1))

    report("Default params (baseline)", y_sim, default_preds, classes, high_idx)
    report("HP-tuned (text n=300/log2, acoustic n=500/depth30)", y_sim, tuned_preds, classes, high_idx)

    print(f"\n  Reference -- same tuned config on DMC n=11 (already established):")
    print(f"    acc=0.6364  macroF1=0.6076  HighRec=0.6000  (vs default n=11: acc=0.6727 macroF1=0.6303 HighRec=0.6571)")


if __name__ == "__main__":
    main()
