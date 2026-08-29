"""
asr_source_comparison_simulated.py
Extends asr_source_comparison_late_fusion.py (own vs friend's ASR, DMC n=11
and n=25) to the SIMULATED corpus, using data/features.csv (own ASR,
Lingalingeswaran/whisper-small-sinhala_v3) vs data/features_v2.csv (friend's
ASR, Component 2's fine-tuned faster-whisper) -- same 267 calls, same audio,
verified acoustic-feature-identical (max abs diff 0.0) and label-identical
(265/267 rows; kept the standard features.csv labels throughout, matching
every other simulated-CV report this session).

Only the TEXT branch's input changes between the two conditions -- acoustic
branch, model, and fusion recipe (0.5/0.5, default params) held identical,
same as the DMC-side version of this experiment.
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


def report(name, y_true, preds_by_seed, high_idx):
    accs, macros, rec_rows = [], [], []
    for pred in preds_by_seed:
        accs.append(accuracy_score(y_true, pred))
        macros.append(f1_score(y_true, pred, average="macro"))
        _, r, _, _ = precision_recall_fscore_support(y_true, pred, labels=[0, 1, 2], zero_division=0)
        rec_rows.append(r)
    rec = np.mean(rec_rows, axis=0)
    print(f"  {name:<42} acc={np.mean(accs):.4f}  macroF1={np.mean(macros):.4f}  "
          f"HighRec={rec[high_idx]:.4f}  LowRec={rec[1]:.4f}  MedRec={rec[2]:.4f}")


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    high_idx = list(le.classes_).index("High")

    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")               # own ASR
    if "fire_smoke_match" not in sim.columns:
        sim = sim.copy()
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)

    sim_v2 = pd.read_csv(f"{ROOT}\\data\\features_v2.csv")          # friend's ASR
    if "fire_smoke_match" not in sim_v2.columns:
        sim_v2 = sim_v2.copy()
        sim_v2["fire_smoke_match"] = sim_v2["transcript"].map(fire_smoke_match)

    assert len(sim) == len(sim_v2), "row count mismatch!"
    y = le.transform(sim["urgency_label"].values)  # standard labels, used throughout this session

    X_ac = sim[ACOUSTIC].fillna(0).values           # identical either way (verified: max abs diff 0.0)
    X_tx_own = sim[CONDITION_C].fillna(0).values
    X_tx_friend = sim_v2[CONDITION_C].fillna(0).values

    print("=" * 90)
    print("SIMULATED 5-fold CV -- late fusion (0.5/0.5), own ASR vs friend's ASR text branch")
    print("=" * 90)

    preds_own, preds_friend = [], []
    for seed in SEEDS:
        proba_ac = cv_proba(X_ac, y, seed)
        proba_tx_own = cv_proba(X_tx_own, y, seed)
        proba_tx_friend = cv_proba(X_tx_friend, y, seed)
        preds_own.append((0.5 * proba_ac + 0.5 * proba_tx_own).argmax(axis=1))
        preds_friend.append((0.5 * proba_ac + 0.5 * proba_tx_friend).argmax(axis=1))

    report("OWN ASR (trained/evaluated on this)", y, preds_own, high_idx)
    report("FRIEND'S ASR (what live deployment sends)", y, preds_friend, high_idx)

    print(f"\n  Reference -- DMC n=11: own=0.6727/0.6303  friend=0.8000/0.7528 (own trusted less, small n)")
    print(f"  Reference -- DMC n=25: own=0.5440/0.5162  friend=0.3840/0.2836 (more trustworthy view)")


if __name__ == "__main__":
    main()
