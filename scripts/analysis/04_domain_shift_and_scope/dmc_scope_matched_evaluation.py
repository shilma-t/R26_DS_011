"""
dmc_scope_matched_evaluation.py
Tests whether DMC's overall 45.8% (text-only) accuracy is being pulled down
by emergency types the simulated corpus was never trained on, by isolating
the flood/fire-only DMC subset (the emergency types the simulated training
data was actually designed around) and re-evaluating on that subset alone.

Uses the same keyword-heuristic categorization as error_by_emergency_type.py
(flood_water, flood_water+road_travel, fire, fire+flood_water categories) -
flagged clearly as a best-effort proxy since ~50% of DMC calls landed
"unclear" under this simple categorizer, so this is likely an UNDERCOUNT of
the true flood/fire subset, not an exact one. Small-n caveat applies (~11
calls) - report with that honestly, not as a definitive number.

Reports, for the full 59-call DMC set AND the flood/fire-only subset:
  - overall accuracy, macro-F1
  - per-class recall (High/Medium/Low)
  - High->Low error count specifically
  - text-only, acoustic-only, and late-fusion (0.5/0.5) side by side
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
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
    if not hits:
        return "unclear"
    return "+".join(hits)


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


def evaluate(y_true, preds_by_seed, classes, high_idx, low_idx):
    accs, macros = [], []
    recall_by_class = {c: [] for c in classes}
    high_to_low = []
    for pred in preds_by_seed:
        accs.append(accuracy_score(y_true, pred))
        macros.append(f1_score(y_true, pred, average="macro"))
        rec = recall_score(y_true, pred, labels=range(len(classes)), average=None, zero_division=0)
        for i, c in enumerate(classes):
            recall_by_class[c].append(rec[i])
        high_to_low.append(int(((y_true == high_idx) & (pred == low_idx)).sum()))
    return accs, macros, recall_by_class, high_to_low


def print_summary(name, y_true, preds_by_seed, classes, high_idx, low_idx, n_high):
    accs, macros, recall_by_class, h2l = evaluate(y_true, preds_by_seed, classes, high_idx, low_idx)
    print(f"\n  {name}  (n={len(y_true)}, n_true_High={n_high})")
    print(f"    accuracy={np.mean(accs):.4f}  macroF1={np.mean(macros):.4f}")
    for c in classes:
        print(f"    {c:<8} recall={np.mean(recall_by_class[c]):.4f}")
    print(f"    High->Low errors: mean={np.mean(h2l):.2f} out of {n_high} true-High  "
          f"(per-seed: {h2l})")


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    classes = list(le.classes_)
    high_idx = classes.index("High")
    low_idx = classes.index("Low")

    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim = sim.copy()
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    y_sim = le.transform(sim["urgency_label"].values)
    X_sim_tx = sim[CONDITION_C].fillna(0).values
    X_sim_ac = sim[ACOUSTIC].fillna(0).values

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    dmc_v2 = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation_v2.csv")
    dmc_v2["category"] = dmc_v2["transcript"].map(categorize)
    dmc = dmc.reset_index(drop=True)
    dmc_v2 = dmc_v2.reset_index(drop=True)
    assert (dmc["recording_id"] == dmc_v2["recording_id"]).all()
    dmc["category"] = dmc_v2["category"]

    flood_fire_mask = dmc["category"].isin(FLOOD_FIRE_CATEGORIES)
    print(f"Flood/fire-scope-matched DMC subset: {flood_fire_mask.sum()} of {len(dmc)} calls")
    print(dmc[flood_fire_mask][["recording_id", "true_label", "category"]].to_string(index=False))

    y_dmc_all = le.transform(dmc["true_label"].values)
    X_dmc_all_tx = dmc[CONDITION_C].fillna(0).values
    X_dmc_all_ac = dmc[ACOUSTIC].fillna(0).values

    dmc_ff = dmc[flood_fire_mask].reset_index(drop=True)
    y_dmc_ff = le.transform(dmc_ff["true_label"].values)
    X_dmc_ff_tx = dmc_ff[CONDITION_C].fillna(0).values
    X_dmc_ff_ac = dmc_ff[ACOUSTIC].fillna(0).values

    n_high_all = int((y_dmc_all == high_idx).sum())
    n_high_ff = int((y_dmc_ff == high_idx).sum())

    print("\nTraining production models at 5 seeds ...")
    tx_preds_all, ac_preds_all, late_preds_all = [], [], []
    tx_preds_ff, ac_preds_ff, late_preds_ff = [], [], []
    for seed in SEEDS:
        pipe_tx = fit_pipe(seed); pipe_tx.fit(X_sim_tx, y_sim)
        pipe_ac = fit_pipe(seed); pipe_ac.fit(X_sim_ac, y_sim)

        proba_tx_all = pipe_tx.predict_proba(X_dmc_all_tx)
        proba_ac_all = pipe_ac.predict_proba(X_dmc_all_ac)
        tx_preds_all.append(proba_tx_all.argmax(axis=1))
        ac_preds_all.append(proba_ac_all.argmax(axis=1))
        late_preds_all.append((0.5 * proba_tx_all + 0.5 * proba_ac_all).argmax(axis=1))

        proba_tx_ff = pipe_tx.predict_proba(X_dmc_ff_tx)
        proba_ac_ff = pipe_ac.predict_proba(X_dmc_ff_ac)
        tx_preds_ff.append(proba_tx_ff.argmax(axis=1))
        ac_preds_ff.append(proba_ac_ff.argmax(axis=1))
        late_preds_ff.append((0.5 * proba_tx_ff + 0.5 * proba_ac_ff).argmax(axis=1))

    print("\n" + "=" * 78)
    print("ALL 59 DMC calls (reference)")
    print("=" * 78)
    print_summary("Text-only", y_dmc_all, tx_preds_all, classes, high_idx, low_idx, n_high_all)
    print_summary("Acoustic-only", y_dmc_all, ac_preds_all, classes, high_idx, low_idx, n_high_all)
    print_summary("Late fusion (0.5/0.5)", y_dmc_all, late_preds_all, classes, high_idx, low_idx, n_high_all)

    print("\n" + "=" * 78)
    print(f"FLOOD/FIRE-ONLY DMC subset (n={flood_fire_mask.sum()}, best-effort categorized)")
    print("=" * 78)
    print_summary("Text-only", y_dmc_ff, tx_preds_ff, classes, high_idx, low_idx, n_high_ff)
    print_summary("Acoustic-only", y_dmc_ff, ac_preds_ff, classes, high_idx, low_idx, n_high_ff)
    print_summary("Late fusion (0.5/0.5)", y_dmc_ff, late_preds_ff, classes, high_idx, low_idx, n_high_ff)

    print("\n" + "=" * 78)
    print("SUMMARY TABLE")
    print("=" * 78)
    print(f"  {'Test set':<30}{'Method':<20}{'Accuracy':>10}{'MacroF1':>10}")
    for label, y, preds_dict in [
        ("All 59 DMC", y_dmc_all, {"Text-only": tx_preds_all, "Acoustic-only": ac_preds_all, "Late fusion": late_preds_all}),
        (f"Flood/fire only (n={flood_fire_mask.sum()})", y_dmc_ff, {"Text-only": tx_preds_ff, "Acoustic-only": ac_preds_ff, "Late fusion": late_preds_ff}),
    ]:
        for method, preds in preds_dict.items():
            accs = [accuracy_score(y, p) for p in preds]
            macros = [f1_score(y, p, average="macro") for p in preds]
            print(f"  {label:<30}{method:<20}{np.mean(accs):>10.4f}{np.mean(macros):>10.4f}")


if __name__ == "__main__":
    main()
