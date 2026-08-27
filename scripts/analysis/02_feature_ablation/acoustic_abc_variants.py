"""
acoustic_abc_variants.py
Step 3 of the protocol (reached because late fusion did NOT beat text-only -
see late_fusion_experiment.py): tests whether a specific SUBSET of the 91
acoustic features transfers better to DMC than the full acoustic set.

  ACOUSTIC-A: MFCC mean/std only (80 feat) - drop the 11 non-MFCC descriptors
              (pitch/rms/zcr/spectral centroid/speaking rate/jitter/slopes).
  ACOUSTIC-B: non-MFCC only (11 feat) - drop all 80 MFCCs.
  ACOUSTIC-C: ACOUSTIC-A minus the top-10 most domain-shifted MFCC
              coefficients (by |Cohen's d|, simulated vs DMC), i.e. 70 feat.

Each variant is evaluated three ways, multi-seed (5 seeds), on DMC:
  - standalone (acoustic variant alone)
  - early fusion (concat with Condition C's 19 text/context features)
  - late fusion, unweighted 0.5/0.5 (the strategy that came closest to
    matching text-only alone in the previous experiment)

All disposable/in-memory - frozen v1 artifacts untouched.
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

MFCC = [f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")]
NON_MFCC = ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
    "spec_centroid_mean", "speaking_rate", "jitter",
    "rms_slope", "rms_gradient", "speaking_rate_change"]
ACOUSTIC_FULL = MFCC + NON_MFCC
CONDITION_C = ["asr_quality", "keyword_count", "repetition_rate", "sentiment_polarity",
    "sin_context_score", "sin_danger_count", "sin_help_request_count",
    "sin_high_risk_phrase_count", "sin_location_context_count",
    "sin_person_at_risk_count", "sin_safety_state_count", "tam_context_score",
    "tam_danger_count", "tam_help_request_count", "tam_high_risk_phrase_count",
    "tam_location_context_count", "tam_person_at_risk_count",
    "tam_safety_state_count", "fire_smoke_match"]
assert len(MFCC) == 80 and len(NON_MFCC) == 11 and len(ACOUSTIC_FULL) == 91


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


def multiseed_dmc_standalone(sim, dmc, y_sim, y_dmc, cols, le):
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


def multiseed_dmc_early_fusion(sim, dmc, y_sim, y_dmc, acoustic_cols, le):
    cols = acoustic_cols + CONDITION_C
    return multiseed_dmc_standalone(sim, dmc, y_sim, y_dmc, cols, le)


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
    print(f"  {name:<48}acc={np.mean(res['accs']):.4f}  macroF1={np.mean(res['macros']):.4f}  "
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
    print("Building ACOUSTIC-A/B/C variants")
    print("=" * 78)
    ACOUSTIC_A = MFCC  # 80
    ACOUSTIC_B = NON_MFCC  # 11

    shift_rows = []
    for col in MFCC:
        d = cohens_d(sim[col].fillna(0).values, dmc[col].fillna(0).values)
        shift_rows.append({"feature": col, "cohens_d": d})
    shift_df = pd.DataFrame(shift_rows).sort_values("cohens_d", key=lambda s: s.abs(), ascending=False)
    top10_shifted_mfcc = shift_df.head(10)["feature"].tolist()
    print(f"Top-10 most domain-shifted MFCC coefficients (dropped for ACOUSTIC-C):")
    print(shift_df.head(10).round(3).to_string(index=False))

    ACOUSTIC_C = [c for c in MFCC if c not in top10_shifted_mfcc]  # 70
    print(f"\nACOUSTIC-A: {len(ACOUSTIC_A)} feat (MFCC mean/std only)")
    print(f"ACOUSTIC-B: {len(ACOUSTIC_B)} feat (non-MFCC only)")
    print(f"ACOUSTIC-C: {len(ACOUSTIC_C)} feat (MFCC minus top-10 shifted)")

    variants = [
        ("ACOUSTIC-A (MFCC only, 80)", ACOUSTIC_A),
        ("ACOUSTIC-B (non-MFCC only, 11)", ACOUSTIC_B),
        ("ACOUSTIC-C (MFCC minus top-10 shifted, 70)", ACOUSTIC_C),
    ]

    print("\n" + "=" * 78)
    print("References (from prior experiments, for comparison)")
    print("=" * 78)
    res_text = multiseed_dmc_standalone(sim, dmc, y_sim, y_dmc, CONDITION_C, le)
    res_acoustic_full = multiseed_dmc_standalone(sim, dmc, y_sim, y_dmc, ACOUSTIC_FULL, le)
    line("Text-only (Condition C, 19)", res_text)
    line("Acoustic FULL (91, reference)", res_acoustic_full)

    print("\n" + "=" * 78)
    print("STANDALONE: each acoustic variant alone, on DMC")
    print("=" * 78)
    standalone_results = {}
    for name, cols in variants:
        res = multiseed_dmc_standalone(sim, dmc, y_sim, y_dmc, cols, le)
        standalone_results[name] = res
        line(name, res)

    print("\n" + "=" * 78)
    print("EARLY FUSION: each acoustic variant concatenated with Condition C")
    print("=" * 78)
    early_results = {}
    for name, cols in variants:
        res = multiseed_dmc_early_fusion(sim, dmc, y_sim, y_dmc, cols, le)
        early_results[name] = res
        line(name + " + text (early fusion)", res)

    print("\n" + "=" * 78)
    print("LATE FUSION (unweighted 0.5/0.5): each acoustic variant + Condition C")
    print("=" * 78)
    late_results = {}
    for name, cols in variants:
        res = multiseed_dmc_late_fusion(sim, dmc, y_sim, y_dmc, cols, le, w=0.5)
        late_results[name] = res
        line(name + " + text (late fusion 0.5/0.5)", res)

    print("\n" + "=" * 78)
    print("FINAL SUMMARY TABLE - DMC 59 calls, mean over 5 seeds")
    print("=" * 78)
    print(f"  {'condition':<48}{'acc':>8}{'macroF1':>10}{'HighRec':>10}{'MedRec':>10}")
    line("Text-only (Condition C, 19) [BEST SO FAR]", res_text)
    line("Acoustic FULL (91)", res_acoustic_full)
    for name, _ in variants:
        line(name + " (standalone)", standalone_results[name])
    for name, _ in variants:
        line(name + " (early fusion w/ text)", early_results[name])
    for name, _ in variants:
        line(name + " (late fusion w/ text)", late_results[name])

    print("\nAny variant beat text-only alone on macro-F1? (mean over 5 seeds)")
    text_macro = np.mean(res_text["macros"])
    for group_name, group in [("standalone", standalone_results), ("early fusion", early_results), ("late fusion", late_results)]:
        for name, res in group.items():
            m = np.mean(res["macros"])
            if m > text_macro:
                print(f"  YES: {name} ({group_name}) macroF1={m:.4f} > text-only {text_macro:.4f}")
    print("  (if nothing printed above, none beat text-only)")


if __name__ == "__main__":
    main()
