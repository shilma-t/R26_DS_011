"""
new_asr_hp_and_weight_tuning.py
Two checks on late fusion using the NEW ASR's text features consistently
(data/features_v2.csv for simulated, reports/pp2/dmc_evaluation_v2.csv for
DMC) instead of the original ASR -- text branch trained AND evaluated on the
new-ASR transcripts throughout, so this tests "what if the whole pipeline
committed to this ASR", not a train/eval mismatch.

Acoustic branch is unaffected by ASR choice either way (verified earlier:
max abs diff 0.0 between the two feature files' acoustic columns).

  A. Hyperparameter tuning -- same TUNED_TEXT/TUNED_ACOUSTIC params already
     established elsewhere in this project, re-tested with the new-ASR text
     branch, on: simulated CV, DMC n=59, DMC n=11 (flood-scoped).
  B. Fusion-weight sweep (w_acoustic 0.0-1.0) -- same grid as before,
     re-tested with the new-ASR text branch, same three views.

Does not retrain or replace any frozen/deployed model file.
"""
import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
SEEDS = [1, 7, 42, 99, 123]
DEFAULT_PARAMS = dict(n_estimators=200, max_depth=None, min_samples_leaf=2, max_features="sqrt")
TUNED_TEXT = dict(n_estimators=300, max_depth=None, min_samples_leaf=2, max_features="log2")
TUNED_ACOUSTIC = dict(n_estimators=500, max_depth=30, min_samples_leaf=1, max_features="sqrt")
WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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

TYPE_PATTERNS = {
    "fire": ["ගිනි", "ගින්", "දුම", "எரிச்சல்", "தீ", "புகை", "fire", "smoke", "burning"],
    "flood_water": ["වතුර", "ගංවතුර", "යටවෙලා", "හිරවෙලා", "සිරවෙලා", "කොටුවෙලා",
                    "வெள்ளம்", "தண்ணீர்", "flood", "water"],
    "tree_structural": ["ගහක්", "ගස්", "කඩන්නක්", "කඩන්නාව", "වැටිලා", "පුළුගහක්", "tree"],
    "tsunami_wind": ["සුනාමි", "සුළං", "සුළි", "tsunami"],
    "road_travel": ["පාර", "රූට්", "යන්න පුළුවන්", "route", "road", "kolamba", "කොළඹ"],
    "medical_rescue": ["හොස්පිට්ල්", "රෝහල", "දොක්තර", "வைத்தியர்", "hospital", "boat",
                       "බෝට්", "රැකවරණ", "රැකගන්න"],
    "status_no_impact": ["බලපෑමක් නැහැ", "අවදානමක් නැහ", "no impact"],
}
FLOOD_FIRE_CATEGORIES = {"fire", "flood_water", "flood_water+road_travel", "fire+flood_water"}


def categorize(text):
    if not isinstance(text, str) or not text.strip():
        return "empty"
    hits = [cat for cat, patterns in TYPE_PATTERNS.items() if any(p in text for p in patterns)]
    return "+".join(hits) if hits else "unclear"


def fire_smoke_match(text):
    kws = ["fire","smoke","burning","flames","gas","ගිනිගන්නව","ගිනිඅරගෙන",
           "ගින්නක්","ගිනි","දුමක්","දුම","தீ","புகை","gini"]
    if not isinstance(text, str): return 0
    t = text.lower()
    return int(any(k.lower() in t for k in kws))


def fit_pipe(seed, params):
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=seed)),
        ("clf", RandomForestClassifier(random_state=seed, class_weight="balanced", n_jobs=-1, **params)),
    ])


def cv_proba(X, y, seed, params):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out = np.zeros((len(y), 3))
    for train_idx, test_idx in skf.split(X, y):
        pipe = fit_pipe(seed, params)
        pipe.fit(X[train_idx], y[train_idx])
        out[test_idx] = pipe.predict_proba(X[test_idx])
    return out


def fit_predict_proba(X_train, y_train, X_test, seed, params):
    pipe = fit_pipe(seed, params)
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_test)


def metrics(y_true, pred, high_idx):
    acc = accuracy_score(y_true, pred)
    macro = f1_score(y_true, pred, average="macro")
    _, r, _, _ = precision_recall_fscore_support(y_true, pred, labels=[0, 1, 2], zero_division=0)
    return acc, macro, r


def print_row(name, acc, macro, rec, high_idx):
    print(f"  {name:<40} acc={acc:.4f}  macroF1={macro:.4f}  "
          f"HighRec={rec[high_idx]:.4f}  LowRec={rec[1]:.4f}  MedRec={rec[2]:.4f}")


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    high_idx = list(le.classes_).index("High")

    # New-ASR simulated corpus (features_v2.csv) -- used consistently for train+eval below
    sim2 = pd.read_csv(f"{ROOT}\\data\\features_v2.csv")
    if "fire_smoke_match" not in sim2.columns:
        sim2 = sim2.copy()
        sim2["fire_smoke_match"] = sim2["transcript"].map(fire_smoke_match)
    y_sim = le.transform(sim2["urgency_label"].values)
    X_sim_ac = sim2[ACOUSTIC].fillna(0).values
    X_sim_tx = sim2[CONDITION_C].fillna(0).values

    # New-ASR DMC (dmc_evaluation_v2.csv)
    dmc2 = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation_v2.csv")
    dmc_own = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")  # for true_label + categorize alignment
    y_dmc59 = le.transform(dmc_own["true_label"].values)
    X59_ac = dmc2[ACOUSTIC].fillna(0).values
    X59_tx = dmc2[CONDITION_C].fillna(0).values

    category = dmc2["transcript"].map(categorize)
    in_scope_11 = category.isin(FLOOD_FIRE_CATEGORIES).values
    y_dmc11 = y_dmc59[in_scope_11]
    X11_ac = X59_ac[in_scope_11]
    X11_tx = X59_tx[in_scope_11]
    print(f"New-ASR simulated n={len(sim2)}, DMC n=59, DMC flood-scoped n={in_scope_11.sum()}\n")

    # =========================== A. HYPERPARAMETER TUNING ===========================
    print("=" * 95)
    print("A. HYPERPARAMETER TUNING -- new-ASR text branch, default vs tuned, late fusion 0.5/0.5")
    print("=" * 95)

    for view_name, X_ac_eval, X_tx_eval, y_eval, is_cv in [
        ("SIMULATED (5-fold CV)", X_sim_ac, X_sim_tx, y_sim, True),
        ("DMC n=59", X59_ac, X59_tx, y_dmc59, False),
        ("DMC n=11 (flood-scoped)", X11_ac, X11_tx, y_dmc11, False),
    ]:
        print(f"\n  -- {view_name} --")
        for label, params in [("Default params", DEFAULT_PARAMS), ("HP-tuned", None)]:
            preds = []
            for seed in SEEDS:
                ac_params = TUNED_ACOUSTIC if label == "HP-tuned" else DEFAULT_PARAMS
                tx_params = TUNED_TEXT if label == "HP-tuned" else DEFAULT_PARAMS
                if is_cv:
                    proba_ac = cv_proba(X_ac_eval, y_eval, seed, ac_params)
                    proba_tx = cv_proba(X_tx_eval, y_eval, seed, tx_params)
                else:
                    proba_ac = fit_predict_proba(X_sim_ac, y_sim, X_ac_eval, seed, ac_params)
                    proba_tx = fit_predict_proba(X_sim_tx, y_sim, X_tx_eval, seed, tx_params)
                preds.append((0.5 * proba_ac + 0.5 * proba_tx).argmax(axis=1))
            accs, macros, recs = [], [], []
            for pred in preds:
                acc, macro, rec = metrics(y_eval, pred, high_idx)
                accs.append(acc); macros.append(macro); recs.append(rec)
            print_row(label, np.mean(accs), np.mean(macros), np.mean(recs, axis=0), high_idx)

    # =========================== B. FUSION-WEIGHT SWEEP ===========================
    print("\n" + "=" * 95)
    print("B. FUSION-WEIGHT SWEEP -- new-ASR text branch, default RF params, w_acoustic 0.0-1.0")
    print("=" * 95)

    for view_name, X_ac_eval, X_tx_eval, y_eval, is_cv in [
        ("SIMULATED (5-fold CV)", X_sim_ac, X_sim_tx, y_sim, True),
        ("DMC n=59", X59_ac, X59_tx, y_dmc59, False),
        ("DMC n=11 (flood-scoped)", X11_ac, X11_tx, y_dmc11, False),
    ]:
        print(f"\n  -- {view_name} --")
        # Precompute per-seed probas once, reuse across all weights (much faster)
        proba_ac_by_seed, proba_tx_by_seed = {}, {}
        for seed in SEEDS:
            if is_cv:
                proba_ac_by_seed[seed] = cv_proba(X_ac_eval, y_eval, seed, DEFAULT_PARAMS)
                proba_tx_by_seed[seed] = cv_proba(X_tx_eval, y_eval, seed, DEFAULT_PARAMS)
            else:
                proba_ac_by_seed[seed] = fit_predict_proba(X_sim_ac, y_sim, X_ac_eval, seed, DEFAULT_PARAMS)
                proba_tx_by_seed[seed] = fit_predict_proba(X_sim_tx, y_sim, X_tx_eval, seed, DEFAULT_PARAMS)

        rows = []
        for w in WEIGHT_GRID:
            accs, macros, recs = [], [], []
            for seed in SEEDS:
                combined = w * proba_ac_by_seed[seed] + (1 - w) * proba_tx_by_seed[seed]
                pred = combined.argmax(axis=1)
                acc, macro, rec = metrics(y_eval, pred, high_idx)
                accs.append(acc); macros.append(macro); recs.append(rec)
            rec_mean = np.mean(recs, axis=0)
            rows.append({"w_acoustic": w, "acc": np.mean(accs), "macro_f1": np.mean(macros),
                        "high_rec": rec_mean[high_idx], "low_rec": rec_mean[1], "med_rec": rec_mean[2]})
        grid = pd.DataFrame(rows)
        print(grid.round(4).to_string(index=False))
        best_acc = grid.loc[grid["acc"].idxmax()]
        best_macro = grid.loc[grid["macro_f1"].idxmax()]
        print(f"  Best by accuracy: w_acoustic={best_acc.w_acoustic:.1f} (acc={best_acc.acc:.4f})")
        print(f"  Best by macro-F1: w_acoustic={best_macro.w_acoustic:.1f} (macroF1={best_macro.macro_f1:.4f})")


if __name__ == "__main__":
    main()
