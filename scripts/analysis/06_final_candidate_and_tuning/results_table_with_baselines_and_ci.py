"""
results_table_with_baselines_and_ci.py
Produces the PP2 results table with the things the earlier reporting was
missing: majority-class baselines, per-class recall, and 95% Wilson
confidence intervals on accuracy.

Rationale: on imbalanced small test sets (DMC n=59, flood/fire subset
n=11) raw accuracy is close to meaningless without the majority baseline
next to it, and a point estimate without a CI overstates precision.

Also runs McNemar (exact binomial, paired) for the two comparisons that
actually get claimed in the report:
  - Condition C vs v1 on the full 59-call DMC set
  - late fusion vs v1 on the flood/fire subset

Models are averaged over 5 seeds for the reported metrics; the McNemar
test uses a single fixed seed (42) since the test needs one concrete
paired prediction set, not an average.
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
RF_BASE = dict(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1)
SEEDS = [1, 7, 42, 99, 123]
MCNEMAR_SEED = 42

ACOUSTIC = ([f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")] +
    ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
     "spec_centroid_mean", "speaking_rate", "jitter",
     "rms_slope", "rms_gradient", "speaking_rate_change"])
TEXT_V1 = ["asr_quality", "keyword_count", "word_count", "repetition_rate",
           "sentiment_polarity"]
CONTEXT = ["sin_context_score", "sin_danger_count", "sin_help_request_count",
    "sin_high_risk_phrase_count", "sin_location_context_count",
    "sin_person_at_risk_count", "sin_safety_state_count", "tam_context_score",
    "tam_danger_count", "tam_help_request_count", "tam_high_risk_phrase_count",
    "tam_location_context_count", "tam_person_at_risk_count",
    "tam_safety_state_count"]
FULL_V1 = ACOUSTIC + TEXT_V1 + CONTEXT + ["fire_smoke_match"]
CONDITION_C = (["asr_quality", "keyword_count", "repetition_rate", "sentiment_polarity"]
               + CONTEXT + ["fire_smoke_match"])

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
    hits = [c for c, ps in TYPE_PATTERNS.items() if any(p in text for p in ps)]
    return "+".join(hits) if hits else "unclear"


def fire_smoke_match(text):
    kws = ["fire","smoke","burning","flames","gas","ගිනිගන්නව","ගිනිඅරගෙන",
           "ගින්නක්","ගිනි","දුමක්","දුම","தீ","புகை","gini"]
    if not isinstance(text, str): return 0
    t = text.lower()
    return int(any(k.lower() in t for k in kws))


def wilson_ci(successes, n, z=1.96):
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, centre - half), min(1.0, centre + half))


def fit_pipe(seed, params=None):
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=seed)),
        ("clf", RandomForestClassifier(random_state=seed, **(params or RF_BASE))),
    ])


def metrics_from_preds(y_true, preds_by_seed, classes):
    """Mean metrics over seeds, plus a Wilson CI using the mean accuracy."""
    accs, macros = [], []
    rec = {c: [] for c in classes}
    for pred in preds_by_seed:
        accs.append(accuracy_score(y_true, pred))
        macros.append(f1_score(y_true, pred, average="macro"))
        r = recall_score(y_true, pred, labels=range(len(classes)),
                         average=None, zero_division=0)
        for i, c in enumerate(classes):
            rec[c].append(r[i])
    mean_acc = float(np.mean(accs))
    n = len(y_true)
    lo, hi = wilson_ci(mean_acc * n, n)
    return {
        "accuracy": mean_acc,
        "macro_f1": float(np.mean(macros)),
        "recall_High": float(np.mean(rec["High"])),
        "recall_Medium": float(np.mean(rec["Medium"])),
        "recall_Low": float(np.mean(rec["Low"])),
        "ci_low": lo,
        "ci_high": hi,
    }


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    classes = list(le.classes_)

    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim = sim.copy()
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    y_sim = le.transform(sim["urgency_label"].values)

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv").reset_index(drop=True)
    dmc_v2 = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation_v2.csv").reset_index(drop=True)
    dmc["category"] = dmc_v2["transcript"].map(categorize)
    y_dmc = le.transform(dmc["true_label"].values)

    ff_mask = dmc["category"].isin(FLOOD_FIRE_CATEGORIES).values
    dmc_ff = dmc[ff_mask].reset_index(drop=True)
    y_ff = le.transform(dmc_ff["true_label"].values)

    print(f"Full DMC n={len(y_dmc)}   flood/fire subset n={len(y_ff)}\n")

    # ---------- train production models once per seed, reuse everywhere -------
    X_sim_full = sim[FULL_V1].fillna(0).values
    X_sim_c = sim[CONDITION_C].fillna(0).values
    X_sim_ac = sim[ACOUSTIC].fillna(0).values

    def predict_sets(cols_sim_X, cols_name):
        """Returns dict: 'dmc' -> preds list, 'ff' -> preds list."""
        X_dmc = dmc[cols_name].fillna(0).values
        X_ff = dmc_ff[cols_name].fillna(0).values
        out_dmc, out_ff = [], []
        for seed in SEEDS:
            pipe = fit_pipe(seed)
            pipe.fit(cols_sim_X, y_sim)
            out_dmc.append(pipe.predict(X_dmc))
            out_ff.append(pipe.predict(X_ff))
        return out_dmc, out_ff

    v1_dmc, v1_ff = predict_sets(X_sim_full, FULL_V1)
    c_dmc, c_ff = predict_sets(X_sim_c, CONDITION_C)
    ac_dmc, ac_ff = predict_sets(X_sim_ac, ACOUSTIC)

    # late fusion = 0.5/0.5 of Condition C proba and acoustic-only proba
    lf_dmc, lf_ff = [], []
    for seed in SEEDS:
        p_tx = fit_pipe(seed); p_tx.fit(X_sim_c, y_sim)
        p_ac = fit_pipe(seed); p_ac.fit(X_sim_ac, y_sim)
        for src_df, store in ((dmc, lf_dmc), (dmc_ff, lf_ff)):
            proba = (0.5 * p_tx.predict_proba(src_df[CONDITION_C].fillna(0).values)
                     + 0.5 * p_ac.predict_proba(src_df[ACOUSTIC].fillna(0).values))
            store.append(proba.argmax(axis=1))

    # ---------- simulated in-domain (5-fold CV), all four configurations -----
    def cv_proba(X, seed):
        """Out-of-fold predict_proba for one seed."""
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        proba = np.zeros((len(y_sim), 3))
        for tr, te in skf.split(X, y_sim):
            pipe = fit_pipe(seed)
            pipe.fit(X[tr], y_sim[tr])
            proba[te] = pipe.predict_proba(X[te])
        return proba

    sim_full_preds, sim_c_preds, sim_ac_preds, sim_lf_preds = [], [], [], []
    for seed in SEEDS:
        pr_full = cv_proba(X_sim_full, seed)
        pr_c = cv_proba(X_sim_c, seed)
        pr_ac = cv_proba(X_sim_ac, seed)
        sim_full_preds.append(pr_full.argmax(axis=1))
        sim_c_preds.append(pr_c.argmax(axis=1))
        sim_ac_preds.append(pr_ac.argmax(axis=1))
        # late fusion uses the SAME fold structure for both branches, so the
        # combination is over genuinely held-out probabilities on both sides
        sim_lf_preds.append((0.5 * pr_c + 0.5 * pr_ac).argmax(axis=1))

    # ---------- assemble table ----------------------------------------------
    def majority_row(y_true, name):
        counts = np.bincount(y_true, minlength=3)
        maj_idx = int(counts.argmax())
        pred = np.full(len(y_true), maj_idx)
        m = metrics_from_preds(y_true, [pred], classes)
        m["model"] = f"Majority baseline (always {classes[maj_idx]})"
        m["dataset"] = name
        return m

    rows = []

    for name, y_true, sets in [
        ("SIMULATED (5-fold CV, n=267)", y_sim,
         [("v1 full 111-feature (early fusion)", sim_full_preds),
          ("Acoustic-only (91)", sim_ac_preds),
          ("Condition C (text/context, 19)", sim_c_preds),
          ("Late fusion 0.5/0.5", sim_lf_preds)]),
        ("DMC FULL (n=59)", y_dmc,
         [("v1 full 111-feature (early fusion)", v1_dmc),
          ("Acoustic-only (91)", ac_dmc),
          ("Condition C (text/context, 19)", c_dmc),
          ("Late fusion 0.5/0.5", lf_dmc)]),
        ("DMC FLOOD/FIRE SUBSET (n=11)", y_ff,
         [("v1 full 111-feature (early fusion)", v1_ff),
          ("Acoustic-only (91)", ac_ff),
          ("Condition C (text/context, 19)", c_ff),
          ("Late fusion 0.5/0.5", lf_ff)]),
    ]:
        rows.append(majority_row(y_true, name))
        for model_name, preds in sets:
            m = metrics_from_preds(y_true, preds, classes)
            m["model"] = model_name
            m["dataset"] = name
            rows.append(m)

    df = pd.DataFrame(rows)[
        ["dataset", "model", "accuracy", "macro_f1",
         "recall_High", "recall_Medium", "recall_Low", "ci_low", "ci_high"]
    ]

    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        print("=" * 108)
        print(ds)
        print("=" * 108)
        print(f"  {'model':<38}{'acc':>7}{'macroF1':>9}{'HighR':>8}{'MedR':>8}{'LowR':>8}   {'95% CI (acc)':>18}")
        for _, r in sub.iterrows():
            ci = f"[{r.ci_low:.3f}, {r.ci_high:.3f}]"
            print(f"  {r.model:<38}{r.accuracy:>7.3f}{r.macro_f1:>9.3f}"
                  f"{r.recall_High:>8.3f}{r.recall_Medium:>8.3f}{r.recall_Low:>8.3f}   {ci:>18}")
        print()

    df.to_csv(f"{ROOT}\\reports\\pp2_finalized_reports\\results_table_with_baselines_and_ci.csv",
              index=False)
    print("saved -> reports/pp2_finalized_reports/results_table_with_baselines_and_ci.csv\n")

    # ---------- McNemar (paired, exact binomial) ------------------------------
    print("=" * 108)
    print(f"McNemar exact tests (paired, single seed={MCNEMAR_SEED})")
    print("=" * 108)

    def mcnemar(y_true, pred_a, pred_b, label_a, label_b):
        a_ok = pred_a == y_true
        b_ok = pred_b == y_true
        b_only = int(np.sum(~a_ok & b_ok))   # B right, A wrong
        a_only = int(np.sum(a_ok & ~b_ok))   # A right, B wrong
        n_disc = a_only + b_only
        if n_disc == 0:
            print(f"  {label_b} vs {label_a}: no discordant pairs, test undefined")
            return
        res = binomtest(b_only, n_disc, 0.5)
        print(f"  {label_b} vs {label_a}")
        print(f"    {label_b} right / {label_a} wrong : {b_only}")
        print(f"    {label_a} right / {label_b} wrong : {a_only}")
        print(f"    discordant pairs = {n_disc}   exact p = {res.pvalue:.4f}"
              f"   -> {'SIGNIFICANT at 0.05' if res.pvalue < 0.05 else 'NOT significant at 0.05'}")

    seed_i = SEEDS.index(MCNEMAR_SEED)
    print("\nFull DMC (n=59):")
    mcnemar(y_dmc, v1_dmc[seed_i], c_dmc[seed_i], "v1", "Condition C")
    mcnemar(y_dmc, v1_dmc[seed_i], lf_dmc[seed_i], "v1", "Late fusion")

    print("\nFlood/fire subset (n=11):")
    mcnemar(y_ff, v1_ff[seed_i], lf_ff[seed_i], "v1", "Late fusion")
    mcnemar(y_ff, v1_ff[seed_i], c_ff[seed_i], "v1", "Condition C")


if __name__ == "__main__":
    main()
