"""
feature_source_ablation.py
Tests whether acoustic features (91) structurally dominate/outvote text
features (20) in a way that hurts DMC generalization - the leading
unexplained hypothesis from the MODEL_OVERRIDE error-analysis cases.

1. Train acoustic-only, text/context-only, and full-111 models (same RF +
   SMOTE + class_weight settings as the frozen pipeline), on the full
   267-call simulated corpus, score against the existing 59-call DMC
   feature vectors. Averaged over 3 seeds for basic robustness.

2. Check the 10 MODEL_OVERRIDE cases specifically: does acoustic-only get
   the true label right more often than the full model on exactly these
   calls (where text signal was correct but the full model still failed)?

3. Down-weight acoustic columns (x0.5) in the FULL feature set and re-score,
   as a cheap partial-fusion-rebalance test.

Everything here is diagnostic/disposable - frozen v1 artifacts (features.csv,
urgency_rf_v1.joblib, dmc_evaluation.csv) are read-only and untouched.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
RF_BASE = dict(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1)
SEEDS = [1, 42, 99]

ACOUSTIC = ([f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")] +
    ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
     "spec_centroid_mean", "speaking_rate", "jitter",
     "rms_slope", "rms_gradient", "speaking_rate_change"])
TEXT_CONTEXT = ["asr_quality", "keyword_count", "word_count", "repetition_rate",
    "sentiment_polarity", "sin_context_score", "sin_danger_count",
    "sin_help_request_count", "sin_high_risk_phrase_count",
    "sin_location_context_count", "sin_person_at_risk_count",
    "sin_safety_state_count", "tam_context_score", "tam_danger_count",
    "tam_help_request_count", "tam_high_risk_phrase_count",
    "tam_location_context_count", "tam_person_at_risk_count",
    "tam_safety_state_count", "fire_smoke_match"]
FULL = ACOUSTIC + TEXT_CONTEXT

assert len(ACOUSTIC) == 91 and len(TEXT_CONTEXT) == 20 and len(FULL) == 111


def fire_smoke_match(text):
    kws = ["fire","smoke","burning","flames","gas","ගිනිගන්නව","ගිනිඅරගෙන",
           "ගින්නක්","ගිනි","දුමක්","දුම","தீ","புகை","gini"]
    if not isinstance(text, str): return 0
    t = text.lower()
    return int(any(k.lower() in t for k in kws))


def fit_predict(X_train, y_train, X_test, seed, weight_vec=None):
    """weight_vec (if given) is applied AFTER StandardScaler, so a constant
    multiplier actually reaches the RF/SMOTE instead of being cancelled out
    by re-standardization (the bug in the first Part 3 run)."""
    steps = [("scaler", StandardScaler())]
    if weight_vec is not None:
        steps.append(("weight", FunctionTransformer(lambda X: X * weight_vec)))
    steps += [
        ("smote", SMOTE(random_state=seed)),
        ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
    ]
    pipe = ImbPipeline(steps)
    pipe.fit(X_train, y_train)
    return pipe.predict(X_test), pipe.predict_proba(X_test)


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])

    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    y_sim = le.transform(sim["urgency_label"].values)

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    y_dmc = le.transform(dmc["true_label"].values)

    print("=" * 78)
    print("PART 1: acoustic-only vs text/context-only vs full-111")
    print(f"(averaged over seeds {SEEDS}, scored on the existing 59 DMC calls)")
    print("=" * 78)

    all_preds = {}  # condition -> {seed: y_pred} for later per-call analysis
    for name, cols in [("ACOUSTIC-only (91)", ACOUSTIC),
                       ("TEXT/CONTEXT-only (20)", TEXT_CONTEXT),
                       ("FULL (111)", FULL)]:
        X_sim = sim[cols].fillna(0).values
        X_dmc = dmc[cols].fillna(0).values

        accs, macros, hrs = [], [], []
        preds_by_seed = {}
        for seed in SEEDS:
            pred, _ = fit_predict(X_sim, y_sim, X_dmc, seed)
            preds_by_seed[seed] = pred
            accs.append(accuracy_score(y_dmc, pred))
            macros.append(f1_score(y_dmc, pred, average="macro"))
            hrs.append((pred[y_dmc == 0] == 0).mean())
        all_preds[name] = preds_by_seed

        print(f"\n{name}")
        print(f"  mean acc={np.mean(accs):.4f} (+/-{np.std(accs):.4f})  "
              f"mean macroF1={np.mean(macros):.4f}  mean HighRecall={np.mean(hrs):.4f}")

    print("\n" + "=" * 78)
    print("PART 2: the 10 MODEL_OVERRIDE cases - does ACOUSTIC-only beat FULL here?")
    print("=" * 78)

    err = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_error_analysis.csv")
    override_ids = err[err["category"] == "MODEL_OVERRIDE"]["recording_id"].tolist()
    dmc_idx = dmc.reset_index(drop=True)
    id_to_pos = {rid: i for i, rid in enumerate(dmc_idx["recording_id"])}

    seed = SEEDS[len(SEEDS) // 2]  # middle seed as a representative example
    acoustic_pred = all_preds["ACOUSTIC-only (91)"][seed]
    text_pred = all_preds["TEXT/CONTEXT-only (20)"][seed]
    full_pred = all_preds["FULL (111)"][seed]
    classes = le.classes_

    acoustic_wins = text_wins = full_wins = neither = 0
    for rid in override_ids:
        if rid not in id_to_pos:
            continue
        pos = id_to_pos[rid]
        true_label = dmc_idx.loc[pos, "true_label"]
        a_correct = classes[acoustic_pred[pos]] == true_label
        t_correct = classes[text_pred[pos]] == true_label
        f_correct = classes[full_pred[pos]] == true_label
        print(f"  {rid[:32]:<32} true={true_label:<7}  "
              f"acoustic={classes[acoustic_pred[pos]]:<7}({'OK' if a_correct else 'no'})  "
              f"text={classes[text_pred[pos]]:<7}({'OK' if t_correct else 'no'})  "
              f"full={classes[full_pred[pos]]:<7}({'OK' if f_correct else 'no'})")
        if a_correct and not f_correct:
            acoustic_wins += 1
        if t_correct and not f_correct:
            text_wins += 1
        if f_correct:
            full_wins += 1
        if not a_correct and not t_correct and not f_correct:
            neither += 1

    print(f"\n  Of {len(override_ids)} MODEL_OVERRIDE cases (seed={seed}):")
    print(f"    acoustic-only correct where full was wrong: {acoustic_wins}")
    print(f"    text-only correct where full was wrong    : {text_wins}")
    print(f"    full model actually correct               : {full_wins}")
    print(f"    none of the three got it right             : {neither}")

    print("\n" + "=" * 78)
    print("PART 3 (fixed): down-weight acoustic columns AFTER StandardScaler")
    print("(previous run multiplied raw columns before scaling, which")
    print(" StandardScaler silently cancelled out - invalid test, redone here)")
    print("=" * 78)

    X_sim_full = sim[FULL].fillna(0).values
    X_dmc_full = dmc[FULL].fillna(0).values
    n_acoustic = len(ACOUSTIC)

    for weight in [1.0, 0.5, 0.25, 0.1]:
        weight_vec = np.concatenate([
            np.full(n_acoustic, weight),
            np.full(len(TEXT_CONTEXT), 1.0),
        ])
        accs, macros, hrs = [], [], []
        for seed in SEEDS:
            pred, _ = fit_predict(X_sim_full, y_sim, X_dmc_full, seed, weight_vec=weight_vec)
            accs.append(accuracy_score(y_dmc, pred))
            macros.append(f1_score(y_dmc, pred, average="macro"))
            hrs.append((pred[y_dmc == 0] == 0).mean())
        tag = "baseline (x1.0)" if weight == 1.0 else f"acoustic x{weight}"
        print(f"  {tag:<20} mean acc={np.mean(accs):.4f}  "
              f"mean macroF1={np.mean(macros):.4f}  mean HighRecall={np.mean(hrs):.4f}")


if __name__ == "__main__":
    main()
