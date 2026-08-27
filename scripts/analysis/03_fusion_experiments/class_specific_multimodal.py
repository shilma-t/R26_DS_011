"""
class_specific_multimodal.py
Tests whether a CLASS-SPECIFIC (per-severity-class) modality combination
strategy recovers useful multimodal performance on DMC, instead of forcing
all acoustic+text features into one RF (early fusion) or using one global
fusion weight for every call regardless of predicted class (the late-fusion
approach tested earlier, which failed).

Protocol (DMC used ONLY as independent external validation - the per-class
modality preference is decided from SIMULATED 5-fold CV only, then applied
once, frozen, to DMC):

  STEP 1: text-only (Condition C, 19) and acoustic-only (91) models,
          5-fold CV on the ORIGINAL simulated training corpus (in-domain
          reference, matching PP1's ~60%+ accuracy scale) - per-class
          Precision/Recall/F1 for each modality separately.
  STEP 2: same two models retrained as production models (fit on all 267
          simulated calls) at 5 seeds, predict_proba on DMC - per-class
          P/R/F1 for text-only, acoustic-only, standard early fusion (110
          feat), and standard late fusion (unweighted 0.5/0.5).
  STEP 3: decide per-class fusion weights from STEP 1's simulated per-class
          F1 (whichever modality wins that class on simulated CV gets more
          weight for that class) - a genuine per-class generalization of
          the single global late-fusion weight, decided blind to DMC.
  STEP 4: apply the frozen per-class weights to DMC's already-computed
          STEP-2 probabilities, report full per-class P/R/F1, overall
          accuracy/macroF1, High recall, and count of High->Low errors
          specifically (the supervisor's stated priority).
  STEP 5: final side-by-side comparison table: text-only vs acoustic-only
          vs standard fusion vs class-specific fusion.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                              confusion_matrix)
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


def per_class_report(y_true, y_pred, classes, label):
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0)
    print(f"\n  {label}")
    for i, c in enumerate(classes):
        print(f"    {c:<8} precision={prec[i]:.3f}  recall={rec[i]:.3f}  "
              f"f1={f1[i]:.3f}  (n={support[i]})")
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    print(f"    -> accuracy={acc:.3f}  macroF1={macro:.3f}")
    return {"precision": prec, "recall": rec, "f1": f1, "acc": acc, "macro": macro}


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    classes = list(le.classes_)  # [High, Low, Medium]
    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim = sim.copy()
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    y_sim = le.transform(sim["urgency_label"].values)

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    y_dmc = le.transform(dmc["true_label"].values)

    print("=" * 78)
    print("STEP 1: SIMULATED 5-fold CV, per-class P/R/F1, text-only vs acoustic-only")
    print("(in-domain reference - this is the ORIGINAL simulated corpus, matching")
    print(" PP1's in-domain accuracy scale)")
    print("=" * 78)

    def cv_proba(cols):
        X = sim[cols].fillna(0).values
        all_pred = np.zeros((len(SEEDS), len(y_sim)), dtype=int)
        for si, seed in enumerate(SEEDS):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            proba = np.zeros((len(y_sim), 3))
            for tr, te in skf.split(X, y_sim):
                pipe = fit_pipe(seed)
                pipe.fit(X[tr], y_sim[tr])
                proba[te] = pipe.predict_proba(X[te])
            all_pred[si] = proba.argmax(axis=1)
        return all_pred

    text_cv_preds = cv_proba(CONDITION_C)
    acoustic_cv_preds = cv_proba(ACOUSTIC)

    # average per-class F1 across the 5 seeds
    text_f1_by_class = np.mean([
        precision_recall_fscore_support(y_sim, text_cv_preds[s], labels=range(3), zero_division=0)[2]
        for s in range(len(SEEDS))], axis=0)
    acoustic_f1_by_class = np.mean([
        precision_recall_fscore_support(y_sim, acoustic_cv_preds[s], labels=range(3), zero_division=0)[2]
        for s in range(len(SEEDS))], axis=0)

    print("\nSimulated per-class F1 (mean over 5 seeds):")
    for i, c in enumerate(classes):
        winner = "TEXT wins" if text_f1_by_class[i] > acoustic_f1_by_class[i] else "ACOUSTIC wins"
        print(f"  {c:<8} text_F1={text_f1_by_class[i]:.3f}  acoustic_F1={acoustic_f1_by_class[i]:.3f}  -> {winner}")

    per_class_report(y_sim, text_cv_preds[2], classes, "Text-only, simulated CV (seed=42 example)")
    per_class_report(y_sim, acoustic_cv_preds[2], classes, "Acoustic-only, simulated CV (seed=42 example)")

    print("\n" + "=" * 78)
    print("STEP 2: DMC (production models, 5 seeds) - text-only, acoustic-only, "
          "standard early fusion, standard late fusion")
    print("=" * 78)

    X_sim_tx = sim[CONDITION_C].fillna(0).values
    X_sim_ac = sim[ACOUSTIC].fillna(0).values
    X_sim_fused = sim[FUSED].fillna(0).values
    X_dmc_tx = dmc[CONDITION_C].fillna(0).values
    X_dmc_ac = dmc[ACOUSTIC].fillna(0).values
    X_dmc_fused = dmc[FUSED].fillna(0).values

    proba_tx_by_seed, proba_ac_by_seed, pred_fused_by_seed, pred_latefusion_by_seed = [], [], [], []
    for seed in SEEDS:
        pipe_tx = fit_pipe(seed); pipe_tx.fit(X_sim_tx, y_sim)
        proba_tx = pipe_tx.predict_proba(X_dmc_tx)
        proba_tx_by_seed.append(proba_tx)

        pipe_ac = fit_pipe(seed); pipe_ac.fit(X_sim_ac, y_sim)
        proba_ac = pipe_ac.predict_proba(X_dmc_ac)
        proba_ac_by_seed.append(proba_ac)

        pipe_fused = fit_pipe(seed); pipe_fused.fit(X_sim_fused, y_sim)
        pred_fused_by_seed.append(pipe_fused.predict(X_dmc_fused))

        pred_latefusion_by_seed.append((0.5 * proba_ac + 0.5 * proba_tx).argmax(axis=1))

    pred_text_by_seed = [p.argmax(axis=1) for p in proba_tx_by_seed]
    pred_acoustic_by_seed = [p.argmax(axis=1) for p in proba_ac_by_seed]

    seed_ex = 2  # seed=42
    per_class_report(y_dmc, pred_text_by_seed[seed_ex], classes, "Text-only, DMC (seed=42)")
    per_class_report(y_dmc, pred_acoustic_by_seed[seed_ex], classes, "Acoustic-only, DMC (seed=42)")
    per_class_report(y_dmc, pred_fused_by_seed[seed_ex], classes, "Standard EARLY fusion, DMC (seed=42)")
    per_class_report(y_dmc, pred_latefusion_by_seed[seed_ex], classes, "Standard LATE fusion (0.5/0.5), DMC (seed=42)")

    print("\n" + "=" * 78)
    print("STEP 3: decide per-class fusion weights from STEP 1 (simulated only)")
    print("=" * 78)
    class_weight_acoustic = []
    for i, c in enumerate(classes):
        if acoustic_f1_by_class[i] > text_f1_by_class[i]:
            w = 0.8   # acoustic wins this class on simulated -> trust acoustic more
        else:
            w = 0.2   # text wins this class on simulated -> trust text more
        class_weight_acoustic.append(w)
        print(f"  {c:<8}: frozen weight_acoustic={w}  "
              f"(text_F1={text_f1_by_class[i]:.3f} vs acoustic_F1={acoustic_f1_by_class[i]:.3f})")
    class_weight_acoustic = np.array(class_weight_acoustic)

    print("\n" + "=" * 78)
    print("STEP 4: apply frozen per-class weights to DMC (5 seeds)")
    print("combined_proba[:,c] = w[c]*P_acoustic[:,c] + (1-w[c])*P_text[:,c]")
    print("=" * 78)
    pred_classspecific_by_seed = []
    for s in range(len(SEEDS)):
        combined = (class_weight_acoustic * proba_ac_by_seed[s] +
                   (1 - class_weight_acoustic) * proba_tx_by_seed[s])
        pred_classspecific_by_seed.append(combined.argmax(axis=1))

    per_class_report(y_dmc, pred_classspecific_by_seed[seed_ex], classes,
                     "CLASS-SPECIFIC fusion, DMC (seed=42)")

    high_idx = classes.index("High")
    low_idx = classes.index("Low")
    print("\nHigh->Low error counts (out of 23 true-High calls), per approach, per seed:")
    for name, preds in [("Text-only", pred_text_by_seed), ("Acoustic-only", pred_acoustic_by_seed),
                        ("Standard early fusion", pred_fused_by_seed),
                        ("Standard late fusion", pred_latefusion_by_seed),
                        ("Class-specific fusion", pred_classspecific_by_seed)]:
        counts = [int(((y_dmc == high_idx) & (p == low_idx)).sum()) for p in preds]
        print(f"  {name:<24} High->Low counts per seed: {counts}  mean={np.mean(counts):.2f}")

    print("\n" + "=" * 78)
    print("STEP 5: FINAL COMPARISON - mean over 5 seeds, DMC")
    print("=" * 78)
    def summarize(name, preds):
        accs = [accuracy_score(y_dmc, p) for p in preds]
        macros = [f1_score(y_dmc, p, average="macro") for p in preds]
        hrs = [(p[y_dmc == high_idx] == high_idx).mean() for p in preds]
        f1_by_class_all = np.array([
            precision_recall_fscore_support(y_dmc, p, labels=range(3), zero_division=0)[2]
            for p in preds])
        print(f"\n  {name}")
        print(f"    acc={np.mean(accs):.4f}  macroF1={np.mean(macros):.4f}  HighRecall={np.mean(hrs):.4f}")
        for i, c in enumerate(classes):
            print(f"    {c:<8} F1={f1_by_class_all[:,i].mean():.3f}")
        return np.mean(accs), np.mean(macros), np.mean(hrs)

    summarize("Text-only", pred_text_by_seed)
    summarize("Acoustic-only", pred_acoustic_by_seed)
    summarize("Standard EARLY fusion", pred_fused_by_seed)
    summarize("Standard LATE fusion (0.5/0.5)", pred_latefusion_by_seed)
    summarize("CLASS-SPECIFIC fusion", pred_classspecific_by_seed)


if __name__ == "__main__":
    main()
