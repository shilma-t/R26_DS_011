"""
empty_transcript_acoustic_fallback.py
Tests a reliability-aware fallback rule with NO tuned/fit parameters (so
there is nothing to overfit to DMC): if the transcript is empty, use
acoustic-only probabilities for that call; otherwise use standard late
fusion (0.5/0.5). This is different from the earlier-rejected rms_cv
confidence gate (SUMMARY.md Sec 6) -- that used an acoustic-side signal
tuned on simulated data; this uses a hard, common-sense text-side signal
(transcript presence) with no free parameters at all.

Motivation: SUMMARY.md Sec 5 already documents that on the one real DMC call
with an empty transcript, acoustic-only correctly called it High in 5/5
seeds while the text/fusion path called it Low -- this tests whether wiring
that observation into the fusion logic actually helps, on both a sanity-check
pass over simulated data and the DMC flood-authoritative n=25 subset.

Evaluated on:
  1. SIMULATED (5-fold CV) -- sanity check only, expect near-zero change
     since empty transcripts are rare there (5/267) and this isn't where
     the problem was ever observed.
  2. DMC flood subset (data/dmc_metadata.csv disaster_type == "Flood", n=25,
     authoritative labels) -- exactly 1 empty-transcript call in this set.
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


def is_empty(transcript):
    return not isinstance(transcript, str) or not transcript.strip()


def apply_fallback(proba_text, proba_acoustic, transcripts):
    """Empty transcript -> acoustic-only proba. Otherwise -> 0.5/0.5 late fusion."""
    out = 0.5 * proba_text + 0.5 * proba_acoustic
    for i, t in enumerate(transcripts):
        if is_empty(t):
            out[i] = proba_acoustic[i]
    return out


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
    sim_transcripts = sim["transcript"].tolist()
    print(f"Simulated corpus: n={len(sim)}, empty transcripts={sum(is_empty(t) for t in sim_transcripts)}")

    meta = pd.read_csv(f"{ROOT}\\data\\dmc_metadata.csv")
    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    dmc = dmc.merge(meta[["recording_id", "disaster_type"]], on="recording_id", how="inner")
    dmc_flood = dmc[dmc["disaster_type"] == "Flood"].reset_index(drop=True)
    y_dmc = le.transform(dmc_flood["true_label"].values)
    X_dmc_ac = dmc_flood[ACOUSTIC].fillna(0).values
    X_dmc_tx = dmc_flood[CONDITION_C].fillna(0).values
    dmc_transcripts = dmc_flood["transcript"].tolist()
    n_empty_dmc = sum(is_empty(t) for t in dmc_transcripts)
    print(f"DMC flood subset: n={len(dmc_flood)}, empty transcripts={n_empty_dmc}\n")

    # ---- 1. SIMULATED sanity check ----
    print("=" * 90)
    print("STEP 1: sanity check on SIMULATED 5-fold CV")
    print("=" * 90)
    base_accs, fb_accs, base_macros, fb_macros, base_hrs, fb_hrs = [], [], [], [], [], []
    for seed in SEEDS:
        proba_tx = cv_proba(X_sim_tx, y_sim, seed)
        proba_ac = cv_proba(X_sim_ac, y_sim, seed)
        base_pred = (0.5 * proba_tx + 0.5 * proba_ac).argmax(axis=1)
        fb_pred = apply_fallback(proba_tx, proba_ac, sim_transcripts).argmax(axis=1)

        b_acc, b_macro, b_hr, _ = metrics(y_sim, base_pred, high_idx)
        f_acc, f_macro, f_hr, _ = metrics(y_sim, fb_pred, high_idx)
        base_accs.append(b_acc); fb_accs.append(f_acc)
        base_macros.append(b_macro); fb_macros.append(f_macro)
        base_hrs.append(b_hr); fb_hrs.append(f_hr)
        n_changed = (base_pred != fb_pred).sum()
        print(f"  seed={seed:<4} baseline: acc={b_acc:.4f} macroF1={b_macro:.4f}   "
              f"fallback: acc={f_acc:.4f} macroF1={f_macro:.4f}   predictions changed={n_changed}")

    print(f"\n  MEAN baseline (0.5/0.5)    : acc={np.mean(base_accs):.4f}  macroF1={np.mean(base_macros):.4f}  HighRec={np.mean(base_hrs):.4f}")
    print(f"  MEAN with empty-fallback   : acc={np.mean(fb_accs):.4f}  macroF1={np.mean(fb_macros):.4f}  HighRec={np.mean(fb_hrs):.4f}")

    # ---- 2. DMC flood subset ----
    print("\n" + "=" * 90)
    print(f"STEP 2: DMC flood subset (n={len(dmc_flood)}, {n_empty_dmc} empty transcript call(s))")
    print("=" * 90)
    d_base_accs, d_fb_accs = [], []
    d_base_macros, d_fb_macros = [], []
    d_base_hrs, d_fb_hrs = [], []
    d_base_hps, d_fb_hps = [], []
    for seed in SEEDS:
        proba_tx = fit_predict_proba(X_sim_tx, y_sim, X_dmc_tx, seed)
        proba_ac = fit_predict_proba(X_sim_ac, y_sim, X_dmc_ac, seed)
        base_pred = (0.5 * proba_tx + 0.5 * proba_ac).argmax(axis=1)
        fb_pred = apply_fallback(proba_tx, proba_ac, dmc_transcripts).argmax(axis=1)

        b_acc, b_macro, b_hr, b_hp = metrics(y_dmc, base_pred, high_idx)
        f_acc, f_macro, f_hr, f_hp = metrics(y_dmc, fb_pred, high_idx)
        d_base_accs.append(b_acc); d_fb_accs.append(f_acc)
        d_base_macros.append(b_macro); d_fb_macros.append(f_macro)
        d_base_hrs.append(b_hr); d_fb_hrs.append(f_hr)
        d_base_hps.append(b_hp); d_fb_hps.append(f_hp)

        changed_idx = np.where(base_pred != fb_pred)[0]
        changed_desc = [(dmc_flood.iloc[i]["recording_id"], classes[base_pred[i]], "->", classes[fb_pred[i]],
                        "true=" + classes[y_dmc[i]]) for i in changed_idx]
        print(f"  seed={seed:<4} baseline: acc={b_acc:.4f} macroF1={b_macro:.4f} HR={b_hr:.4f}   "
              f"fallback: acc={f_acc:.4f} macroF1={f_macro:.4f} HR={f_hr:.4f}   changed={changed_desc}")

    print(f"\n  MEAN baseline (0.5/0.5)    : acc={np.mean(d_base_accs):.4f}  macroF1={np.mean(d_base_macros):.4f}  "
          f"HighRec={np.mean(d_base_hrs):.4f}  HighPrec={np.mean(d_base_hps):.4f}")
    print(f"  MEAN with empty-fallback   : acc={np.mean(d_fb_accs):.4f}  macroF1={np.mean(d_fb_macros):.4f}  "
          f"HighRec={np.mean(d_fb_hrs):.4f}  HighPrec={np.mean(d_fb_hps):.4f}")
    print(f"\n  n={len(dmc_flood)} -- each call = {100/len(dmc_flood):.1f} accuracy points.")


if __name__ == "__main__":
    main()
