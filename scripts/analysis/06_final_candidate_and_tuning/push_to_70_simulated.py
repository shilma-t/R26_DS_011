"""
push_to_70_simulated.py
Three untried levers to push simulated accuracy toward 70%, before testing
the winner on the flood/fire-only DMC subset:

  1. RF hyperparameter tuning (n_estimators, max_depth, min_samples_leaf,
     max_features) - everything tonight used fixed defaults (200 trees,
     min_samples_leaf=2). Tuned separately for text-only, acoustic-only,
     and fused ("both together") feature sets.
  2. Stacking meta-learner (logistic regression on out-of-fold text-only +
     acoustic-only probabilities) instead of a fixed 0.5/0.5 late-fusion
     weight.
  3. Gradient boosting as an alternative classifier on the fused feature
     set, compared against RF.

All tuned/selected on SIMULATED 5-fold CV only (267 calls). The winner is
then re-evaluated on the flood/fire-only DMC subset (n=11) in a second
script, once a winner is picked here.
"""
import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
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

RF_GRID = [
    dict(n_estimators=200, max_depth=None, min_samples_leaf=2, max_features="sqrt"),
    dict(n_estimators=300, max_depth=None, min_samples_leaf=1, max_features="sqrt"),
    dict(n_estimators=300, max_depth=None, min_samples_leaf=2, max_features="log2"),
    dict(n_estimators=400, max_depth=20, min_samples_leaf=1, max_features="sqrt"),
    dict(n_estimators=400, max_depth=None, min_samples_leaf=1, max_features=None),
    dict(n_estimators=500, max_depth=30, min_samples_leaf=1, max_features="sqrt"),
    dict(n_estimators=300, max_depth=15, min_samples_leaf=1, max_features="sqrt"),
]


def fire_smoke_match(text):
    kws = ["fire","smoke","burning","flames","gas","ගිනිගන්නව","ගිනිඅරගෙන",
           "ගින්නක්","ගිනි","දුමක්","දුම","தீ","புகை","gini"]
    if not isinstance(text, str): return 0
    t = text.lower()
    return int(any(k.lower() in t for k in kws))


def cv_eval_rf(X, y, rf_params, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = np.zeros((len(y), 3))
    for tr, te in skf.split(X, y):
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=seed)),
            ("clf", RandomForestClassifier(random_state=seed, class_weight="balanced",
                                           n_jobs=-1, **rf_params)),
        ])
        pipe.fit(X[tr], y[tr])
        proba[te] = pipe.predict_proba(X[te])
    return proba


def cv_eval_gb(X, y, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = np.zeros((len(y), 3))
    for tr, te in skf.split(X, y):
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=seed)),
            ("clf", GradientBoostingClassifier(random_state=seed, n_estimators=200,
                                               max_depth=3, learning_rate=0.05)),
        ])
        pipe.fit(X[tr], y[tr])
        proba[te] = pipe.predict_proba(X[te])
    return proba


def score(y, proba):
    pred = proba.argmax(axis=1)
    return accuracy_score(y, pred), f1_score(y, pred, average="macro")


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim = sim.copy()
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    y_sim = le.transform(sim["urgency_label"].values)
    X_text = sim[CONDITION_C].fillna(0).values
    X_ac = sim[ACOUSTIC].fillna(0).values
    X_fused = sim[FUSED].fillna(0).values

    print("=" * 78)
    print("LEVER 1: RF hyperparameter grid search (fused 110-feat, simulated CV)")
    print("=" * 78)
    baseline_params = dict(n_estimators=200, max_depth=None, min_samples_leaf=2, max_features="sqrt")
    grid_results = []
    for params in RF_GRID:
        accs, macros = [], []
        for seed in SEEDS:
            proba = cv_eval_rf(X_fused, y_sim, params, seed)
            a, m = score(y_sim, proba)
            accs.append(a); macros.append(m)
        grid_results.append({"params": params, "mean_acc": np.mean(accs), "mean_macro": np.mean(macros)})
        print(f"  {params} -> acc={np.mean(accs):.4f}  macroF1={np.mean(macros):.4f}")

    best_rf = max(grid_results, key=lambda r: r["mean_acc"])
    print(f"\n  BEST RF params (fused): {best_rf['params']}  acc={best_rf['mean_acc']:.4f}")

    print("\n  Also tuning RF for TEXT-ONLY and ACOUSTIC-ONLY (for the stacking lever)")
    best_text_params, best_ac_params = None, None
    best_text_acc, best_ac_acc = -1, -1
    for params in RF_GRID:
        accs_t, accs_a = [], []
        for seed in SEEDS:
            proba_t = cv_eval_rf(X_text, y_sim, params, seed)
            proba_a = cv_eval_rf(X_ac, y_sim, params, seed)
            accs_t.append(score(y_sim, proba_t)[0])
            accs_a.append(score(y_sim, proba_a)[0])
        if np.mean(accs_t) > best_text_acc:
            best_text_acc = np.mean(accs_t); best_text_params = params
        if np.mean(accs_a) > best_ac_acc:
            best_ac_acc = np.mean(accs_a); best_ac_params = params
    print(f"  BEST RF params (text-only): {best_text_params}  acc={best_text_acc:.4f}")
    print(f"  BEST RF params (acoustic-only): {best_ac_params}  acc={best_ac_acc:.4f}")

    print("\n" + "=" * 78)
    print("LEVER 2: stacking meta-learner (logistic regression on OOF text+acoustic proba)")
    print("=" * 78)
    stack_accs, stack_macros = [], []
    for seed in SEEDS:
        proba_t = cv_eval_rf(X_text, y_sim, best_text_params, seed)
        proba_a = cv_eval_rf(X_ac, y_sim, best_ac_params, seed)
        meta_X = np.hstack([proba_t, proba_a])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        meta_pred = np.zeros(len(y_sim), dtype=int)
        for tr, te in skf.split(meta_X, y_sim):
            lr = LogisticRegression(max_iter=1000, class_weight="balanced")
            lr.fit(meta_X[tr], y_sim[tr])
            meta_pred[te] = lr.predict(meta_X[te])
        stack_accs.append(accuracy_score(y_sim, meta_pred))
        stack_macros.append(f1_score(y_sim, meta_pred, average="macro"))
    print(f"  Stacking meta-learner: acc={np.mean(stack_accs):.4f}  macroF1={np.mean(stack_macros):.4f}")

    print("\n" + "=" * 78)
    print("LEVER 3: Gradient Boosting vs tuned RF, on fused (110) feature set")
    print("=" * 78)
    gb_accs, gb_macros = [], []
    for seed in SEEDS:
        proba = cv_eval_gb(X_fused, y_sim, seed)
        a, m = score(y_sim, proba)
        gb_accs.append(a); gb_macros.append(m)
    print(f"  Gradient Boosting (fused): acc={np.mean(gb_accs):.4f}  macroF1={np.mean(gb_macros):.4f}")

    print("\n" + "=" * 78)
    print("FINAL COMPARISON - simulated 5-fold CV, mean over 5 seeds")
    print("=" * 78)
    baseline_accs, baseline_macros = [], []
    for seed in SEEDS:
        proba = cv_eval_rf(X_fused, y_sim, baseline_params, seed)
        a, m = score(y_sim, proba)
        baseline_accs.append(a); baseline_macros.append(m)

    print(f"  {'Method':<45}{'Accuracy':>12}{'MacroF1':>12}")
    print(f"  {'Baseline RF (fused, default params)':<45}{np.mean(baseline_accs):>12.4f}{np.mean(baseline_macros):>12.4f}")
    print(f"  {'Tuned RF (fused, lever 1)':<45}{best_rf['mean_acc']:>12.4f}{best_rf['mean_macro']:>12.4f}")
    print(f"  {'Stacking meta-learner (lever 2)':<45}{np.mean(stack_accs):>12.4f}{np.mean(stack_macros):>12.4f}")
    print(f"  {'Gradient Boosting (fused, lever 3)':<45}{np.mean(gb_accs):>12.4f}{np.mean(gb_macros):>12.4f}")

    print(f"\n  BEST TEXT-ONLY params for later use: {best_text_params}")
    print(f"  BEST ACOUSTIC-ONLY params for later use: {best_ac_params}")
    print(f"  BEST FUSED RF params for later use: {best_rf['params']}")


if __name__ == "__main__":
    main()
