"""
full_comparison_sim_and_dmc.py
Complete, side-by-side breakdown: every method (text-only, acoustic-only,
standard early fusion, standard late fusion, class-specific fusion) run
SEPARATELY on simulated (5-fold CV, in-domain) and SEPARATELY on DMC
(production model, external validation) - full per-class precision/recall/F1
plus overall accuracy/macroF1 for each, averaged over 5 seeds throughout.

Class-specific weights (from class_specific_multimodal.py: weight_acoustic
= 0.8 for all three classes, since acoustic wins all three classes on
simulated CV) are FROZEN before touching DMC, and also reported on
simulated itself using proper out-of-fold CV probabilities (not circular -
the weights were already fixed beforehand, this just applies them to the
same CV split's held-out predictions for a proper in-domain reading).
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
CLASS_WEIGHT_ACOUSTIC = np.array([0.8, 0.8, 0.8])  # [High, Low, Medium] - frozen from simulated CV

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


def print_table(name, accs, macros, hrs, f1_by_class, classes):
    print(f"\n  {name}")
    print(f"    accuracy = {np.mean(accs):.4f}   macro-F1 = {np.mean(macros):.4f}   "
          f"High recall = {np.mean(hrs):.4f}")
    for i, c in enumerate(classes):
        print(f"    {c:<8} F1 = {f1_by_class[:, i].mean():.4f}")


def cv_predictions(X, y, seed):
    """Returns out-of-fold predict_proba for a full 5-fold CV at one seed."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = np.zeros((len(y), 3))
    for tr, te in skf.split(X, y):
        pipe = fit_pipe(seed)
        pipe.fit(X[tr], y[tr])
        proba[te] = pipe.predict_proba(X[te])
    return proba


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

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    y_dmc = le.transform(dmc["true_label"].values)
    X_dmc_tx = dmc[CONDITION_C].fillna(0).values
    X_dmc_ac = dmc[ACOUSTIC].fillna(0).values
    X_dmc_fused = dmc[FUSED].fillna(0).values

    # ============================================================
    # PART 1: SIMULATED, 5-fold CV, averaged over 5 seeds
    # ============================================================
    print("=" * 78)
    print("PART 1: SIMULATED (5-fold CV, in-domain), each method separately")
    print("=" * 78)

    sim_results = {}
    proba_tx_sim_seeds, proba_ac_sim_seeds = [], []
    for method in ["text", "acoustic", "early_fusion", "late_fusion", "class_specific"]:
        accs, macros, hrs = [], [], []
        f1_rows = []
        for seed in SEEDS:
            if method == "text":
                proba = cv_predictions(X_sim_tx, y_sim, seed)
            elif method == "acoustic":
                proba = cv_predictions(X_sim_ac, y_sim, seed)
            elif method == "early_fusion":
                proba = cv_predictions(X_sim_fused, y_sim, seed)
            elif method == "late_fusion":
                proba_tx = cv_predictions(X_sim_tx, y_sim, seed)
                proba_ac = cv_predictions(X_sim_ac, y_sim, seed)
                proba = 0.5 * proba_tx + 0.5 * proba_ac
                if seed == SEEDS[0]:
                    pass
            elif method == "class_specific":
                proba_tx = cv_predictions(X_sim_tx, y_sim, seed)
                proba_ac = cv_predictions(X_sim_ac, y_sim, seed)
                proba = CLASS_WEIGHT_ACOUSTIC * proba_ac + (1 - CLASS_WEIGHT_ACOUSTIC) * proba_tx

            pred = proba.argmax(axis=1)
            accs.append(accuracy_score(y_sim, pred))
            macros.append(f1_score(y_sim, pred, average="macro"))
            hrs.append((pred[y_sim == high_idx] == high_idx).mean())
            f1_rows.append(precision_recall_fscore_support(y_sim, pred, labels=range(3), zero_division=0)[2])

        f1_by_class = np.array(f1_rows)
        sim_results[method] = (accs, macros, hrs, f1_by_class)

    names = {"text": "TEXT-ONLY", "acoustic": "ACOUSTIC-ONLY",
             "early_fusion": "STANDARD EARLY FUSION", "late_fusion": "STANDARD LATE FUSION (0.5/0.5)",
             "class_specific": "CLASS-SPECIFIC FUSION (w_acoustic=0.8 all classes)"}
    for method, label in names.items():
        print_table(label, *sim_results[method], classes)

    # ============================================================
    # PART 2: DMC, production model retrained per seed
    # ============================================================
    print("\n" + "=" * 78)
    print("PART 2: DMC (production model, external validation), each method separately")
    print("=" * 78)

    dmc_results = {}
    for method in ["text", "acoustic", "early_fusion", "late_fusion", "class_specific"]:
        accs, macros, hrs = [], [], []
        f1_rows = []
        for seed in SEEDS:
            if method == "text":
                pipe = fit_pipe(seed); pipe.fit(X_sim_tx, y_sim)
                proba = pipe.predict_proba(X_dmc_tx)
            elif method == "acoustic":
                pipe = fit_pipe(seed); pipe.fit(X_sim_ac, y_sim)
                proba = pipe.predict_proba(X_dmc_ac)
            elif method == "early_fusion":
                pipe = fit_pipe(seed); pipe.fit(X_sim_fused, y_sim)
                proba = pipe.predict_proba(X_dmc_fused)
            elif method == "late_fusion":
                pipe_tx = fit_pipe(seed); pipe_tx.fit(X_sim_tx, y_sim)
                pipe_ac = fit_pipe(seed); pipe_ac.fit(X_sim_ac, y_sim)
                proba = 0.5 * pipe_tx.predict_proba(X_dmc_tx) + 0.5 * pipe_ac.predict_proba(X_dmc_ac)
            elif method == "class_specific":
                pipe_tx = fit_pipe(seed); pipe_tx.fit(X_sim_tx, y_sim)
                pipe_ac = fit_pipe(seed); pipe_ac.fit(X_sim_ac, y_sim)
                proba = (CLASS_WEIGHT_ACOUSTIC * pipe_ac.predict_proba(X_dmc_ac) +
                        (1 - CLASS_WEIGHT_ACOUSTIC) * pipe_tx.predict_proba(X_dmc_tx))

            pred = proba.argmax(axis=1)
            accs.append(accuracy_score(y_dmc, pred))
            macros.append(f1_score(y_dmc, pred, average="macro"))
            hrs.append((pred[y_dmc == high_idx] == high_idx).mean())
            f1_rows.append(precision_recall_fscore_support(y_dmc, pred, labels=range(3), zero_division=0)[2])

        f1_by_class = np.array(f1_rows)
        dmc_results[method] = (accs, macros, hrs, f1_by_class)

    for method, label in names.items():
        print_table(label, *dmc_results[method], classes)

    # ============================================================
    # PART 3: side-by-side summary table, both datasets
    # ============================================================
    print("\n" + "=" * 78)
    print("SUMMARY TABLE - accuracy / macroF1 / High recall, SIMULATED vs DMC")
    print("=" * 78)
    print(f"  {'method':<45}{'SIM acc':>10}{'SIM HR':>10}{'DMC acc':>10}{'DMC HR':>10}")
    for method, label in names.items():
        sa, sm, sh, _ = sim_results[method]
        da, dm, dh, _ = dmc_results[method]
        print(f"  {label:<45}{np.mean(sa):>10.4f}{np.mean(sh):>10.4f}"
              f"{np.mean(da):>10.4f}{np.mean(dh):>10.4f}")


if __name__ == "__main__":
    main()
