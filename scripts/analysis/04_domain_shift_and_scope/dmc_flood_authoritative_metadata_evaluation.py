"""
dmc_flood_authoritative_metadata_evaluation.py
Same evaluation as full_metrics_sim_and_dmc_scopematched.py (text-only,
acoustic-only, early fusion, late fusion 0.5/0.5 -- simulated CV and DMC),
but the DMC subset is selected from data/dmc_metadata.csv's `disaster_type`
column (authoritative, manually labeled) instead of the keyword-heuristic
categorize() function used before. disaster_type == "Flood" gives n=25,
more than double the earlier keyword-matched n=11, with no guesswork.

Note: dmc_metadata.csv has no "Fire" value in disaster_type at all (only
Flood / Landslide / Tree fallen / Current / Rain / NaN) -- so this subset is
flood-only, not fire+flood like the earlier scope-matched evaluation. That's
a narrower but more precisely correct match to what the label actually says,
rather than a keyword guess.
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
FUSED = ACOUSTIC + CONDITION_C


def fire_smoke_match(text):
    kws = ["fire","smoke","burning","flames","gas","ගිනිගන්නව","ගිනිඅරගෙන",
           "ගින්නක්","ගිනි","දුමක්","දුම","தீ","புகை","gini"]
    if not isinstance(text, str): return 0
    t = text.lower()
    return int(any(k.lower() in t for k in kws))


def fit_pipe(seed):
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=seed)),
        ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
    ])


def print_full_metrics(name, y_true, preds_by_seed, classes, high_idx):
    accs, macros, hrs = [], [], []
    prec_rows, rec_rows, f1_rows = [], [], []
    for pred in preds_by_seed:
        accs.append(accuracy_score(y_true, pred))
        macros.append(f1_score(y_true, pred, average="macro"))
        hrs.append((pred[y_true == high_idx] == high_idx).mean())
        p, r, f, _ = precision_recall_fscore_support(y_true, pred, labels=range(len(classes)), zero_division=0)
        prec_rows.append(p); rec_rows.append(r); f1_rows.append(f)
    prec = np.mean(prec_rows, axis=0); rec = np.mean(rec_rows, axis=0); f1v = np.mean(f1_rows, axis=0)
    print(f"\n  {name}")
    print(f"    {'class':<8}{'precision':>12}{'recall':>10}{'f1':>10}")
    for i, c in enumerate(classes):
        print(f"    {c:<8}{prec[i]:>12.3f}{rec[i]:>10.3f}{f1v[i]:>10.3f}")
    print(f"    overall accuracy={np.mean(accs):.4f}   macro-F1={np.mean(macros):.4f}   "
          f"High recall={np.mean(hrs):.4f}")


def fit_predict_proba(X_train, y_train, X_test, seed):
    pipe = fit_pipe(seed)
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_test)


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
    X_sim_fused = sim[FUSED].fillna(0).values

    meta = pd.read_csv(f"{ROOT}\\data\\dmc_metadata.csv")
    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    dmc = dmc.merge(meta[["recording_id", "disaster_type", "dmc_priority_label", "notes"]],
                    on="recording_id", how="inner")
    dmc_flood = dmc[dmc["disaster_type"] == "Flood"].reset_index(drop=True)
    print(f"DMC flood subset (authoritative dmc_metadata.csv): n={len(dmc_flood)}")
    print(f"True label counts: {dmc_flood['true_label'].value_counts().to_dict()}\n")

    y_dmc = le.transform(dmc_flood["true_label"].values)
    X_dmc_tx = dmc_flood[CONDITION_C].fillna(0).values
    X_dmc_ac = dmc_flood[ACOUSTIC].fillna(0).values
    X_dmc_fused = dmc_flood[FUSED].fillna(0).values

    dmc_preds = {"text": [], "acoustic": [], "early": [], "late": []}
    for seed in SEEDS:
        proba_tx = fit_predict_proba(X_sim_tx, y_sim, X_dmc_tx, seed)
        proba_ac = fit_predict_proba(X_sim_ac, y_sim, X_dmc_ac, seed)
        proba_early = fit_predict_proba(X_sim_fused, y_sim, X_dmc_fused, seed)
        dmc_preds["text"].append(proba_tx.argmax(axis=1))
        dmc_preds["acoustic"].append(proba_ac.argmax(axis=1))
        dmc_preds["early"].append(proba_early.argmax(axis=1))
        dmc_preds["late"].append((0.5 * proba_tx + 0.5 * proba_ac).argmax(axis=1))

    print("=" * 78)
    print(f"DMC FLOOD ONLY, authoritative dmc_metadata.csv (n={len(dmc_flood)})")
    print("=" * 78)
    print_full_metrics("TEXT-ONLY", y_dmc, dmc_preds["text"], classes, high_idx)
    print_full_metrics("EARLY FUSION (both together, 110 feat)", y_dmc, dmc_preds["early"], classes, high_idx)
    print_full_metrics("ACOUSTIC-ONLY", y_dmc, dmc_preds["acoustic"], classes, high_idx)
    print_full_metrics("LATE FUSION (0.5/0.5)", y_dmc, dmc_preds["late"], classes, high_idx)

    print("\n" + "=" * 78)
    print("REFERENCE -- earlier keyword-heuristic fire+flood scope-matched (n=11):")
    print("  Late fusion (0.5/0.5): acc=0.6727  macroF1=0.6303  HighRec=0.6571  HighPrec=1.0000")
    print(f"  n={len(dmc_flood)} here -- each call = {100/len(dmc_flood):.1f} accuracy points "
          f"(vs {100/11:.1f} for the n=11 set)")
    print("=" * 78)


if __name__ == "__main__":
    main()
