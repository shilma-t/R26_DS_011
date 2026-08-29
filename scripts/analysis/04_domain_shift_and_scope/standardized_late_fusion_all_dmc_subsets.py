"""
standardized_late_fusion_all_dmc_subsets.py
Extends the per-corpus acoustic standardization test (already run on the
n=25 authoritative flood subset) to the n=11 keyword-matched fire+flood set
and the full n=59 DMC set, so all three evaluation views can be compared
side by side using identical metrics.

Everything held identical to every other late-fusion test this session
EXCEPT how the acoustic branch's input is standardized at evaluation time:
  - BASELINE: sim-fit StandardScaler applied to DMC as-is (current approach)
  - STANDARDIZED: DMC acoustic features re-standardized using THAT SAME
    subset's own mean/std (unsupervised, no labels), computed independently
    for n=11 and n=59 respectively (not reusing n=25's stats).

Text branch: always the normal sim-fit scaler, in both baseline and
standardized rows, on every subset -- unchanged from every prior test.

Does NOT retrain, modify, or replace any frozen model. Fresh in-memory RF,
same RF_BASE recipe, same 5 seeds, same SMOTE pipeline as everywhere else.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
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


def fit_fresh(X, y, seed):
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=seed)),
        ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
    ])
    pipe.fit(X, y)
    return pipe


def full_metrics(y_true, pred):
    acc = accuracy_score(y_true, pred)
    macro = f1_score(y_true, pred, average="macro")
    p, r, _, _ = precision_recall_fscore_support(y_true, pred, labels=[0, 1, 2], zero_division=0)
    return acc, macro, r  # r indexed [High, Low, Medium] per LabelEncoder alphabetical order


def evaluate_late_fusion(X_sim_tx, X_sim_ac, y_sim, X_dmc_tx, X_dmc_ac, y_dmc, standardize):
    """standardize=False -> baseline (sim-fit scaler on acoustic too).
    standardize=True -> acoustic re-standardized using THIS subset's own stats."""
    if standardize:
        ac_scaler = StandardScaler().fit(X_dmc_ac)
        X_ac_eval = ac_scaler.transform(X_dmc_ac)

    preds = []
    for seed in SEEDS:
        pipe_tx = fit_fresh(X_sim_tx, y_sim, seed)
        pipe_ac = fit_fresh(X_sim_ac, y_sim, seed)
        proba_tx = pipe_tx.predict_proba(X_dmc_tx)
        clf_ac = pipe_ac.named_steps["clf"]
        if standardize:
            proba_ac = clf_ac.predict_proba(X_ac_eval)
        else:
            proba_ac = pipe_ac.predict_proba(X_dmc_ac)  # normal sim-fit scaler
        preds.append((0.5 * proba_tx + 0.5 * proba_ac).argmax(axis=1))

    accs, macros, recs = [], [], []
    for pred in preds:
        acc, macro, r = full_metrics(y_dmc, pred)
        accs.append(acc); macros.append(macro); recs.append(r)
    return np.mean(accs), np.mean(macros), np.mean(recs, axis=0)


def print_row(name, acc, macro, rec, classes, high_idx):
    print(f"  {name:<42} acc={acc:.4f}  macroF1={macro:.4f}  "
          f"HighRec={rec[high_idx]:.4f}  LowRec={rec[1]:.4f}  MedRec={rec[2]:.4f}")


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

    # ---- n=59: full DMC ----
    dmc59 = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    y_dmc59 = le.transform(dmc59["true_label"].values)
    X59_tx = dmc59[CONDITION_C].fillna(0).values
    X59_ac = dmc59[ACOUSTIC].fillna(0).values

    # ---- n=11: keyword-matched fire+flood scope-matched ----
    dmc_v2 = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation_v2.csv")
    dmc11_src = dmc59.reset_index(drop=True).copy()
    dmc_v2 = dmc_v2.reset_index(drop=True)
    dmc11_src["category"] = dmc_v2["transcript"].map(categorize)
    dmc11 = dmc11_src[dmc11_src["category"].isin(FLOOD_FIRE_CATEGORIES)].reset_index(drop=True)
    y_dmc11 = le.transform(dmc11["true_label"].values)
    X11_tx = dmc11[CONDITION_C].fillna(0).values
    X11_ac = dmc11[ACOUSTIC].fillna(0).values

    print(f"n=11 (keyword fire+flood): {len(dmc11)}   n=59 (full DMC): {len(dmc59)}\n")

    results = {}

    print("=" * 95)
    print("n=11 -- keyword-matched fire+flood scope-matched set")
    print("(CAUTION: 11 samples to estimate 91 feature std devs is very noisy -- treat cautiously)")
    print("=" * 95)
    acc, macro, rec = evaluate_late_fusion(X_sim_tx, X_sim_ac, y_sim, X11_tx, X11_ac, y_dmc11, standardize=False)
    results[("n11", "baseline")] = (acc, macro, rec)
    print_row("Baseline (sim-fit scaler)", acc, macro, rec, classes, high_idx)
    acc, macro, rec = evaluate_late_fusion(X_sim_tx, X_sim_ac, y_sim, X11_tx, X11_ac, y_dmc11, standardize=True)
    results[("n11", "standardized")] = (acc, macro, rec)
    print_row("Standardized (own n=11 stats)", acc, macro, rec, classes, high_idx)

    print("\n" + "=" * 95)
    print("n=59 -- full DMC set")
    print("=" * 95)
    acc, macro, rec = evaluate_late_fusion(X_sim_tx, X_sim_ac, y_sim, X59_tx, X59_ac, y_dmc59, standardize=False)
    results[("n59", "baseline")] = (acc, macro, rec)
    print_row("Baseline (sim-fit scaler)", acc, macro, rec, classes, high_idx)
    acc, macro, rec = evaluate_late_fusion(X_sim_tx, X_sim_ac, y_sim, X59_tx, X59_ac, y_dmc59, standardize=True)
    results[("n59", "standardized")] = (acc, macro, rec)
    print_row("Standardized (own n=59 stats)", acc, macro, rec, classes, high_idx)

    print("\n" + "=" * 95)
    print("FULL COMPARISON -- all three evaluation views, baseline vs standardized late fusion")
    print("=" * 95)
    print(f"  {'Subset':<10}{'Variant':<16}{'Accuracy':>10}{'MacroF1':>10}{'HighRec':>10}{'MedRec':>10}{'LowRec':>10}")
    for subset, n in [("n11", 11), ("n25", 25), ("n59", 59)]:
        for variant in ("baseline", "standardized"):
            if (subset, variant) not in results:
                if subset == "n25":
                    # already established in the prior session, reference only
                    ref = {
                        ("n25", "baseline"): (0.5440, 0.5162, {high_idx: 0.5067, 1: 0.5143, 2: 0.8000}),
                        ("n25", "standardized"): (0.5760, 0.5247, {high_idx: 0.6133, 1: 0.4286, 2: 0.7333}),
                    }
                    acc, macro, recd = ref[(subset, variant)]
                    print(f"  {subset:<10}{variant:<16}{acc:>10.4f}{macro:>10.4f}"
                          f"{recd[high_idx]:>10.4f}{recd[2]:>10.4f}{recd[1]:>10.4f}   (from prior run)")
                continue
            acc, macro, rec = results[(subset, variant)]
            print(f"  {subset:<10}{variant:<16}{acc:>10.4f}{macro:>10.4f}"
                  f"{rec[high_idx]:>10.4f}{rec[2]:>10.4f}{rec[1]:>10.4f}")
        print(f"  ({n} calls -- each = {100/n:.1f} accuracy points)")


if __name__ == "__main__":
    main()
