"""
full_metrics_sim_and_dmc_scopematched.py
Full precision/recall/F1 per class, plus overall accuracy and High recall,
for text-only, acoustic-only, early fusion (both together), and late fusion
(0.5/0.5) - on SIMULATED (5-fold CV, in-domain) and on the DMC flood/fire-
only subset (n=11, best-effort scope-matched), 5 seeds averaged throughout.
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


def cv_predictions(X, y, seed):
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
    dmc_v2 = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation_v2.csv")
    dmc = dmc.reset_index(drop=True); dmc_v2 = dmc_v2.reset_index(drop=True)
    dmc_v2["category"] = dmc_v2["transcript"].map(categorize)
    dmc["category"] = dmc_v2["category"]
    dmc_ff = dmc[dmc["category"].isin(FLOOD_FIRE_CATEGORIES)].reset_index(drop=True)
    y_dmc_ff = le.transform(dmc_ff["true_label"].values)
    X_dmc_ff_tx = dmc_ff[CONDITION_C].fillna(0).values
    X_dmc_ff_ac = dmc_ff[ACOUSTIC].fillna(0).values
    X_dmc_ff_fused = dmc_ff[FUSED].fillna(0).values

    print(f"DMC flood/fire-only subset: n={len(dmc_ff)}")

    # ================= SIMULATED =================
    print("\n" + "=" * 78)
    print("SIMULATED (5-fold CV, in-domain)")
    print("=" * 78)

    sim_preds = {"text": [], "acoustic": [], "both": [], "late": []}
    for seed in SEEDS:
        proba_tx = cv_predictions(X_sim_tx, y_sim, seed)
        proba_ac = cv_predictions(X_sim_ac, y_sim, seed)
        proba_both = cv_predictions(X_sim_fused, y_sim, seed)
        sim_preds["text"].append(proba_tx.argmax(axis=1))
        sim_preds["acoustic"].append(proba_ac.argmax(axis=1))
        sim_preds["both"].append(proba_both.argmax(axis=1))
        sim_preds["late"].append((0.5 * proba_tx + 0.5 * proba_ac).argmax(axis=1))

    print_full_metrics("TEXT-ONLY", y_sim, sim_preds["text"], classes, high_idx)
    print_full_metrics("ACOUSTIC-ONLY", y_sim, sim_preds["acoustic"], classes, high_idx)
    print_full_metrics("BOTH TOGETHER (early fusion)", y_sim, sim_preds["both"], classes, high_idx)
    print_full_metrics("LATE FUSION (0.5/0.5)", y_sim, sim_preds["late"], classes, high_idx)

    # ================= DMC FLOOD/FIRE ONLY =================
    print("\n" + "=" * 78)
    print(f"DMC FLOOD/FIRE ONLY (n={len(dmc_ff)}, production model, external validation)")
    print("=" * 78)

    dmc_preds = {"text": [], "acoustic": [], "both": [], "late": []}
    for seed in SEEDS:
        pipe_tx = fit_pipe(seed); pipe_tx.fit(X_sim_tx, y_sim)
        pipe_ac = fit_pipe(seed); pipe_ac.fit(X_sim_ac, y_sim)
        pipe_both = fit_pipe(seed); pipe_both.fit(X_sim_fused, y_sim)

        proba_tx = pipe_tx.predict_proba(X_dmc_ff_tx)
        proba_ac = pipe_ac.predict_proba(X_dmc_ff_ac)
        proba_both = pipe_both.predict_proba(X_dmc_ff_fused)

        dmc_preds["text"].append(proba_tx.argmax(axis=1))
        dmc_preds["acoustic"].append(proba_ac.argmax(axis=1))
        dmc_preds["both"].append(proba_both.argmax(axis=1))
        dmc_preds["late"].append((0.5 * proba_tx + 0.5 * proba_ac).argmax(axis=1))

    print_full_metrics("TEXT-ONLY", y_dmc_ff, dmc_preds["text"], classes, high_idx)
    print_full_metrics("ACOUSTIC-ONLY", y_dmc_ff, dmc_preds["acoustic"], classes, high_idx)
    print_full_metrics("BOTH TOGETHER (early fusion)", y_dmc_ff, dmc_preds["both"], classes, high_idx)
    print_full_metrics("LATE FUSION (0.5/0.5)", y_dmc_ff, dmc_preds["late"], classes, high_idx)


if __name__ == "__main__":
    main()
