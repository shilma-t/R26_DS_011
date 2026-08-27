"""
low_shift_acoustic_selection.py
Untried angle: instead of block ablation (all MFCCs / non-MFCCs / top-10
shifted removed from a large block), POSITIVELY select only the individual
acoustic features with LOW domain shift (|Cohen's d| < threshold), regardless
of whether they're MFCC or non-MFCC. Earlier subsets (ACOUSTIC-A/B/C) still
contained plenty of moderately-shifted features even after trimming the
worst ones - this targets shift directly, feature-by-feature.

Tests the resulting low-shift acoustic subset:
  - standalone
  - early-fused with Condition C (19 text/context features)
  - late-fused (unweighted 0.5/0.5) with Condition C

against Condition C alone and v1, same 5-seed rigor as everything else.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
RF_BASE = dict(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1)
SEEDS = [1, 7, 42, 99, 123]
SHIFT_THRESHOLDS = [0.2, 0.3, 0.5]  # negligible / negligible-small / small shift cutoffs

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


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled_std


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


def multiseed_dmc_late_fusion(sim, dmc, y_sim, y_dmc, acoustic_cols, le, w=0.5):
    X_sim_ac = sim[acoustic_cols].fillna(0).values
    X_sim_tx = sim[CONDITION_C].fillna(0).values
    X_dmc_ac = dmc[acoustic_cols].fillna(0).values
    X_dmc_tx = dmc[CONDITION_C].fillna(0).values
    accs, macros, hrs, mrs = [], [], [], []
    medium_idx = list(le.classes_).index("Medium")
    for seed in SEEDS:
        proba_ac = fit_predict_proba(X_sim_ac, y_sim, X_dmc_ac, seed)
        proba_tx = fit_predict_proba(X_sim_tx, y_sim, X_dmc_tx, seed)
        combined = w * proba_ac + (1 - w) * proba_tx
        pred = combined.argmax(axis=1)
        accs.append(accuracy_score(y_dmc, pred))
        macros.append(f1_score(y_dmc, pred, average="macro"))
        hrs.append((pred[y_dmc == 0] == 0).mean())
        mrs.append((pred[y_dmc == medium_idx] == medium_idx).mean())
    return {"accs": accs, "macros": macros, "hrs": hrs, "mrs": mrs}


def line(name, res):
    print(f"  {name:<52}acc={np.mean(res['accs']):.4f}  macroF1={np.mean(res['macros']):.4f}  "
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

    print("=" * 78)
    print("STEP 1: compute domain shift for all 91 acoustic features")
    print("=" * 78)
    shift_rows = []
    for col in ACOUSTIC:
        d = cohens_d(sim[col].fillna(0).values, dmc[col].fillna(0).values)
        shift_rows.append({"feature": col, "abs_d": abs(d)})
    shift_df = pd.DataFrame(shift_rows).sort_values("abs_d")
    print(shift_df.head(30).round(3).to_string(index=False))

    print("\n" + "=" * 78)
    print("STEP 2: build low-shift acoustic subsets at a few thresholds")
    print("=" * 78)
    subsets = {}
    for t in SHIFT_THRESHOLDS:
        cols = shift_df[shift_df["abs_d"] < t]["feature"].tolist()
        subsets[t] = cols
        print(f"  threshold |d|<{t}: {len(cols)} features")

    print("\n" + "=" * 78)
    print("References")
    print("=" * 78)
    res_text = multiseed_dmc(sim, dmc, y_sim, y_dmc, CONDITION_C, le)
    line("Text-only (Condition C, 19) [current best]", res_text)

    for t, cols in subsets.items():
        if len(cols) < 3:
            print(f"\n  threshold {t}: too few features ({len(cols)}), skipping")
            continue
        print(f"\n" + "=" * 78)
        print(f"LOW-SHIFT ACOUSTIC (|d|<{t}, {len(cols)} features)")
        print("=" * 78)
        res_standalone = multiseed_dmc(sim, dmc, y_sim, y_dmc, cols, le)
        line(f"standalone", res_standalone)

        res_early = multiseed_dmc(sim, dmc, y_sim, y_dmc, cols + CONDITION_C, le)
        line(f"early fusion + Condition C", res_early)

        res_late = multiseed_dmc_late_fusion(sim, dmc, y_sim, y_dmc, cols, le, w=0.5)
        line(f"late fusion (0.5/0.5) + Condition C", res_late)

        text_macro = np.mean(res_text["macros"])
        for name, res in [("standalone", res_standalone), ("early fusion", res_early), ("late fusion", res_late)]:
            m = np.mean(res["macros"])
            beat = "BEATS text-only!" if m > text_macro else "still below text-only"
            print(f"    {name}: macroF1={m:.4f} vs text-only {text_macro:.4f} -> {beat}")


if __name__ == "__main__":
    main()
