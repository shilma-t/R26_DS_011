"""
acoustic_diagnostic_and_combine.py
Same diagnostic applied to Condition C's text/context features (word_count
was a near-noise, domain-shifted feature dominating importance and causing
a Medium collapse) - now applied to the 91 acoustic features, to check
whether an analogous problem exists there before combining the "fixed"
acoustic set with Condition C.

Steps:
  1. Feature importance of the acoustic-only RF (91 feat) trained on
     simulated data.
  2. Domain shift (Cohen's d, simulated vs DMC) per acoustic feature.
  3. Flag features that are BOTH high-importance AND heavily domain-shifted
     (the same signature word_count had) as candidates to drop.
  4. Test an acoustic-trimmed set (flagged features removed) on DMC, 5-seed
     rigor, vs full acoustic-91 and vs v1.
  5. Combine the best acoustic subset with Condition C's 19 text/context
     features into one "COMBINED" candidate, test the same way.

All disposable/in-memory - frozen v1 artifacts untouched.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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
FULL_V1 = (ACOUSTIC + ["asr_quality", "keyword_count", "word_count", "repetition_rate",
    "sentiment_polarity"] + [c for c in CONDITION_C if c not in
    ("asr_quality", "keyword_count", "repetition_rate", "sentiment_polarity")])
assert len(ACOUSTIC) == 91


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


def fit_on_all_predict(X_train, y_train, X_test, seed):
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=seed)),
        ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict(X_test), pipe.predict_proba(X_test), pipe


def multiseed_dmc(sim, dmc, y_sim, y_dmc, cols, le):
    X_sim = sim[cols].fillna(0).values
    X_dmc = dmc[cols].fillna(0).values
    accs, macros, hrs, mrs = [], [], [], []
    preds = []
    medium_idx = list(le.classes_).index("Medium")
    for seed in SEEDS:
        pred, _, _ = fit_on_all_predict(X_sim, y_sim, X_dmc, seed)
        preds.append(pred)
        accs.append(accuracy_score(y_dmc, pred))
        macros.append(f1_score(y_dmc, pred, average="macro"))
        hrs.append((pred[y_dmc == 0] == 0).mean())
        mrs.append((pred[y_dmc == medium_idx] == medium_idx).mean())
    return {"accs": accs, "macros": macros, "hrs": hrs, "mrs": mrs, "preds": preds}


def report(name, res):
    print(f"\n{name}:")
    print(f"  mean acc={np.mean(res['accs']):.4f}  mean macroF1={np.mean(res['macros']):.4f}  "
          f"mean HighRecall={np.mean(res['hrs']):.4f}  mean MedRecall={np.mean(res['mrs']):.4f}")


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
    print("STEP 1: acoustic-only (91 feat) feature importance, seed=42")
    print("=" * 78)
    X_sim_ac = sim[ACOUSTIC].fillna(0).values
    _, _, pipe = fit_on_all_predict(X_sim_ac, y_sim, X_sim_ac[:5], 42)  # dummy predict to fit
    importances = pipe.named_steps["clf"].feature_importances_
    imp_df = pd.DataFrame({"feature": ACOUSTIC, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)
    print(imp_df.head(20).to_string(index=False))

    print("\n" + "=" * 78)
    print("STEP 2: domain shift (Cohen's d, sim vs DMC) - top 20 by |d|")
    print("=" * 78)
    shift_rows = []
    for col in ACOUSTIC:
        d = cohens_d(sim[col].fillna(0).values, dmc[col].fillna(0).values)
        shift_rows.append({"feature": col, "sim_mean": sim[col].fillna(0).mean(),
                          "dmc_mean": dmc[col].fillna(0).mean(), "cohens_d": d})
    shift_df = pd.DataFrame(shift_rows).sort_values("cohens_d", key=lambda s: s.abs(), ascending=False)
    print(shift_df.head(20).round(3).to_string(index=False))

    print("\n" + "=" * 78)
    print("STEP 3: flag features that are BOTH top-15 importance AND |d|>0.8 (large shift)")
    print("=" * 78)
    top_importance = set(imp_df.head(15)["feature"])
    large_shift = set(shift_df[shift_df["cohens_d"].abs() > 0.8]["feature"])
    flagged = top_importance & large_shift
    print(f"  Top-15 important features: {sorted(top_importance)}")
    print(f"  Large-shift (|d|>0.8) features: {len(large_shift)} total")
    print(f"  FLAGGED (both important AND shifted): {sorted(flagged) if flagged else 'NONE'}")

    if not flagged:
        print("\n  No features meet both criteria as strongly as word_count did.")
        print("  Trying a softer threshold: top-15 importance AND |d|>0.5 (moderate+ shift)")
        moderate_shift = set(shift_df[shift_df["cohens_d"].abs() > 0.5]["feature"])
        flagged = top_importance & moderate_shift
        print(f"  FLAGGED (moderate threshold): {sorted(flagged) if flagged else 'NONE'}")

    acoustic_trimmed = [c for c in ACOUSTIC if c not in flagged]
    print(f"\n  Acoustic trimmed set: {len(acoustic_trimmed)} features (dropped {len(flagged)})")

    print("\n" + "=" * 78)
    print("STEP 4: DMC results - acoustic full(91) vs acoustic trimmed vs v1 vs Condition C")
    print("=" * 78)
    res_ac_full = multiseed_dmc(sim, dmc, y_sim, y_dmc, ACOUSTIC, le)
    res_ac_trim = multiseed_dmc(sim, dmc, y_sim, y_dmc, acoustic_trimmed, le)
    res_v1 = multiseed_dmc(sim, dmc, y_sim, y_dmc, FULL_V1, le)
    res_condc = multiseed_dmc(sim, dmc, y_sim, y_dmc, CONDITION_C, le)

    report("Acoustic FULL (91)", res_ac_full)
    report(f"Acoustic TRIMMED ({len(acoustic_trimmed)}, flagged features dropped)", res_ac_trim)
    report("v1 (111, full baseline)", res_v1)
    report("Condition C (19, text/context best)", res_condc)

    print("\n" + "=" * 78)
    print("STEP 5: COMBINE best acoustic subset + Condition C text/context")
    print("=" * 78)
    best_acoustic = acoustic_trimmed if np.mean(res_ac_trim["macros"]) >= np.mean(res_ac_full["macros"]) else ACOUSTIC
    best_ac_name = f"acoustic trimmed ({len(best_acoustic)})" if best_acoustic is acoustic_trimmed else "acoustic full (91)"
    print(f"  Using {best_ac_name} as the acoustic half of the combination "
          f"(chose whichever had better mean macro-F1 on DMC)")

    combined_cols = best_acoustic + CONDITION_C
    res_combined = multiseed_dmc(sim, dmc, y_sim, y_dmc, combined_cols, le)
    report(f"COMBINED ({len(combined_cols)} = {best_ac_name} + Condition C 19)", res_combined)

    print("\n" + "=" * 78)
    print("FINAL SUMMARY TABLE - all conditions, DMC 59 calls, mean over 5 seeds")
    print("=" * 78)
    summary = [
        ("v1 (111, full baseline)", res_v1),
        ("Acoustic FULL (91)", res_ac_full),
        (f"Acoustic TRIMMED ({len(acoustic_trimmed)})", res_ac_trim),
        ("Condition C (19, text/context)", res_condc),
        (f"COMBINED ({len(combined_cols)})", res_combined),
    ]
    print(f"  {'condition':<38}{'acc':>8}{'macroF1':>10}{'HighRec':>10}{'MedRec':>10}")
    for name, res in summary:
        print(f"  {name:<38}{np.mean(res['accs']):>8.4f}{np.mean(res['macros']):>10.4f}"
              f"{np.mean(res['hrs']):>10.4f}{np.mean(res['mrs']):>10.4f}")

    print("\nWin counts vs v1 (5 seeds each):")
    for name, res in summary[1:]:
        acc_w = sum(1 for a, b in zip(res["accs"], res_v1["accs"]) if a >= b)
        hr_w = sum(1 for a, b in zip(res["hrs"], res_v1["hrs"]) if a >= b)
        print(f"  {name:<38} acc>=v1: {acc_w}/5   HR>=v1: {hr_w}/5")


if __name__ == "__main__":
    main()
