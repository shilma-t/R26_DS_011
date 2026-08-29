"""
per_corpus_standardization_fusion_experiment.py
Extends per_corpus_standardization_experiment.py (acoustic-only self-
standardization) to early fusion (110-feat concatenated model) and late
fusion (0.5/0.5). Same unsupervised technique, same caveats, same "no
labels used, no accuracy-feedback tuning loop" distinction as before.

Only the ACOUSTIC portion of each model's input is re-standardized using
DMC's own mean/std -- the TEXT/context portion keeps the normal sim-fit
scaler throughout, because the domain-shift investigation that motivated
this experiment specifically found substantial shift in acoustic features,
not text/context ones. This is not tested here for text.

For early fusion: StandardScaler is a per-column (elementwise) transform
with no cross-column coupling, so "acoustic columns self-standardized to
DMC + text columns sim-standardized" is assembled directly into the same
110-column order the frozen early-fusion recipe uses -- mathematically
identical to what a single joint scaler would produce per column, just
computed from two different sources for the two feature groups.

For late fusion: the text-only branch is completely untouched (always
sim-fit scaler, as in every other late-fusion test this session); only the
acoustic branch's standardization varies across A/B/C, then combined 0.5/0.5
as always.

Does NOT retrain, modify, or replace any frozen model file. Every RF fit
here is a fresh, throwaway, in-memory copy using the exact same recipe
(RF_BASE, SMOTE, 5 seeds) as every other diagnostic run this session.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
assert len(ACOUSTIC) == 91 and len(CONDITION_C) == 19 and len(FUSED) == 110


def full_metrics(y_true, pred, high_idx):
    acc = accuracy_score(y_true, pred)
    macro = f1_score(y_true, pred, average="macro")
    p, r, _, _ = precision_recall_fscore_support(y_true, pred, labels=[0, 1, 2], zero_division=0)
    return acc, macro, r, p


def report(name, y_true, preds_by_seed, classes, high_idx):
    accs, macros = [], []
    rec_rows, prec_rows = [], []
    for pred in preds_by_seed:
        acc, macro, r, p = full_metrics(y_true, pred, high_idx)
        accs.append(acc); macros.append(macro)
        rec_rows.append(r); prec_rows.append(p)
    rec = np.mean(rec_rows, axis=0)
    print(f"  {name:<48} acc={np.mean(accs):.4f}  macroF1={np.mean(macros):.4f}  "
          f"HighRec={rec[high_idx]:.4f}  MedRec={rec[2]:.4f}  LowRec={rec[1]:.4f}")
    return np.mean(accs), np.mean(macros), rec[high_idx]


def fit_fresh(X, y, seed):
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=seed)),
        ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
    ])
    pipe.fit(X, y)
    return pipe


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    classes = list(le.classes_)
    high_idx = classes.index("High")

    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        kws = ["fire","smoke","burning","flames","gas","ගිනිගන්නව","ගිනිඅරගෙන",
               "ගින්නක්","ගිනි","දුමක්","දුම","தீ","புகை","gini"]
        sim = sim.copy()
        sim["fire_smoke_match"] = sim["transcript"].map(
            lambda t: int(any(k.lower() in str(t).lower() for k in kws)) if isinstance(t, str) else 0)
    y_sim = le.transform(sim["urgency_label"].values)
    X_sim_ac = sim[ACOUSTIC].fillna(0).values
    X_sim_tx = sim[CONDITION_C].fillna(0).values
    X_sim_fused = sim[FUSED].fillna(0).values

    meta = pd.read_csv(f"{ROOT}\\data\\dmc_metadata.csv")
    dmc_all = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    dmc_all = dmc_all.merge(meta[["recording_id", "disaster_type"]], on="recording_id", how="inner")
    dmc_flood = dmc_all[dmc_all["disaster_type"] == "Flood"].reset_index(drop=True)
    y_dmc = le.transform(dmc_flood["true_label"].values)
    X_dmc_ac = dmc_flood[ACOUSTIC].fillna(0).values
    X_dmc_tx = dmc_flood[CONDITION_C].fillna(0).values
    X_dmc_all_ac = dmc_all[ACOUSTIC].fillna(0).values

    print(f"Simulated n={len(sim)}, DMC flood-25 n={len(dmc_flood)}, full DMC n={len(dmc_all)}\n")

    # Three acoustic-standardization variants, applied to the DMC ACOUSTIC block only.
    ac_scaler_25 = StandardScaler().fit(X_dmc_ac)
    ac_scaler_59 = StandardScaler().fit(X_dmc_all_ac)
    variants_ac_dmc = {
        "A_baseline_simscaler": None,   # handled specially: use pipeline's own sim-fit scaler
        "B_selfstd_n25": ac_scaler_25.transform(X_dmc_ac),
        "C_selfstd_n59stats": ac_scaler_59.transform(X_dmc_ac),
    }

    results = {}

    # ================= EARLY FUSION =================
    print("=" * 95)
    print("EARLY FUSION (110 feat, one joint RF) -- DMC flood-25")
    print("=" * 95)
    for vname, ac_dmc_std in variants_ac_dmc.items():
        preds = []
        for seed in SEEDS:
            pipe = fit_fresh(X_sim_fused, y_sim, seed)
            clf = pipe.named_steps["clf"]
            if vname == "A_baseline_simscaler":
                X_eval = pipe.named_steps["scaler"].transform(
                    np.concatenate([X_dmc_ac, X_dmc_tx], axis=1))
            else:
                # text columns: sim-fit scaler (per-column stats only, no cross-coupling)
                sim_scaler = pipe.named_steps["scaler"]
                tx_mean = sim_scaler.mean_[91:]
                tx_scale = sim_scaler.scale_[91:]
                tx_std = (X_dmc_tx - tx_mean) / tx_scale
                X_eval = np.concatenate([ac_dmc_std, tx_std], axis=1)
            preds.append(clf.predict(X_eval))
        results[("early", vname)] = report(f"Early fusion [{vname}]", y_dmc, preds, classes, high_idx)

    # ================= LATE FUSION =================
    print("\n" + "=" * 95)
    print("LATE FUSION (0.5/0.5) -- text branch always sim-scaled; acoustic branch varies -- DMC flood-25")
    print("=" * 95)
    for vname, ac_dmc_std in variants_ac_dmc.items():
        preds = []
        for seed in SEEDS:
            pipe_tx = fit_fresh(X_sim_tx, y_sim, seed)
            pipe_ac = fit_fresh(X_sim_ac, y_sim, seed)

            proba_tx = pipe_tx.predict_proba(X_dmc_tx)  # normal, sim-fit scaler throughout
            clf_ac = pipe_ac.named_steps["clf"]
            if vname == "A_baseline_simscaler":
                X_ac_eval = pipe_ac.named_steps["scaler"].transform(X_dmc_ac)
            else:
                X_ac_eval = ac_dmc_std
            proba_ac = clf_ac.predict_proba(X_ac_eval)

            preds.append((0.5 * proba_tx + 0.5 * proba_ac).argmax(axis=1))
        results[("late", vname)] = report(f"Late fusion [{vname}]", y_dmc, preds, classes, high_idx)

    # ================= SUMMARY =================
    print("\n" + "=" * 95)
    print("SUMMARY -- DMC flood-25 (n=25, each call = 4.0 accuracy points)")
    print("=" * 95)
    print(f"  {'Model':<48}{'Accuracy':>10}{'MacroF1':>10}{'HighRec':>10}")
    for (model, vname), (acc, macro, hr) in results.items():
        print(f"  {model + ' [' + vname + ']':<48}{acc:>10.4f}{macro:>10.4f}{hr:>10.4f}")

    print(f"\n  Reference -- acoustic-only [A baseline]: acc=0.2960 macroF1=0.2444 HighRec=0.2400")
    print(f"  Reference -- acoustic-only [B, n=25 stats]: acc=0.3200 macroF1=0.2638 HighRec=0.3467")
    print(f"  Reference -- acoustic-only [C, n=59 stats]: acc=0.2960 macroF1=0.2362 HighRec=0.3600")
    print(f"  Reference -- text-only: acc=0.6080 macroF1=0.5907 HighRec=0.5067")


if __name__ == "__main__":
    main()
