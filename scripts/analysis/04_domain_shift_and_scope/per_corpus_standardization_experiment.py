"""
per_corpus_standardization_experiment.py
Diagnostic experiment: does standardizing DMC acoustic features using DMC's
OWN mean/std (instead of the simulated-corpus-fit scaler currently used)
improve acoustic-only transfer to real calls?

This is UNSUPERVISED domain adaptation -- it uses only the unlabeled DMC
feature VALUES, never DMC labels, and involves no accuracy-feedback loop
(the recentering is deterministic given the raw features, not chosen by
trying options and picking the best-scoring one). It is a different thing
from tuning weights/hyperparameters by watching DMC accuracy, which is what
"do not tune on DMC" has meant everywhere else in this project.

Does NOT retrain, modify, or replace the frozen acoustic-only model. The RF
classifier itself, its hyperparameters, and its training data are identical
to every other acoustic-only run in this project -- only how the INPUT
features are standardized before being handed to that same classifier
changes, and only at evaluation time.

Caveat stated up front: n=25 is a small sample to estimate 91 feature
means/stds robustly. Reported alongside a second version using the full
n=59 DMC set's stats (more stable estimate) applied to the same n=25 labeled
subset, so the sensitivity to sample size is visible, not hidden.

Also worth knowing before reading results: this technique requires having a
batch of DMC-domain audio to compute stats from BEFORE scoring calls. It
does not trivially apply to scoring one incoming live call in isolation --
noted in the conclusion, not swept under the rug.
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
assert len(ACOUSTIC) == 91


def full_metrics(y_true, pred, classes, high_idx):
    acc = accuracy_score(y_true, pred)
    macro = f1_score(y_true, pred, average="macro")
    p, r, _, _ = precision_recall_fscore_support(y_true, pred, labels=range(len(classes)), zero_division=0)
    return acc, macro, r, p


def report(name, y_true, preds_by_seed, classes, high_idx):
    accs, macros = [], []
    rec_rows, prec_rows = [], []
    for pred in preds_by_seed:
        acc, macro, r, p = full_metrics(y_true, pred, classes, high_idx)
        accs.append(acc); macros.append(macro)
        rec_rows.append(r); prec_rows.append(p)
    rec = np.mean(rec_rows, axis=0)
    prec = np.mean(prec_rows, axis=0)
    print(f"\n  {name}")
    for i, c in enumerate(classes):
        print(f"    {c:<8} recall={rec[i]:.3f}  precision={prec[i]:.3f}")
    print(f"    overall accuracy={np.mean(accs):.4f}   macro-F1={np.mean(macros):.4f}   "
          f"High recall={rec[high_idx]:.4f}")
    return np.mean(accs), np.mean(macros), rec[high_idx]


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    classes = list(le.classes_)
    high_idx = classes.index("High")

    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    y_sim = le.transform(sim["urgency_label"].values)
    X_sim_raw = sim[ACOUSTIC].fillna(0).values

    meta = pd.read_csv(f"{ROOT}\\data\\dmc_metadata.csv")
    dmc_all = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    dmc_all = dmc_all.merge(meta[["recording_id", "disaster_type"]], on="recording_id", how="inner")
    dmc_flood = dmc_all[dmc_all["disaster_type"] == "Flood"].reset_index(drop=True)
    y_dmc = le.transform(dmc_flood["true_label"].values)
    X_dmc_raw = dmc_flood[ACOUSTIC].fillna(0).values
    X_dmc_all_raw = dmc_all[ACOUSTIC].fillna(0).values  # all 59, for the more-stable-stats variant

    print(f"Simulated n={len(sim)}, DMC flood-25 n={len(dmc_flood)}, full DMC n={len(dmc_all)}\n")

    # ---- Variant A: BASELINE -- sim-fit scaler applied to DMC (current production approach) ----
    print("=" * 90)
    print("VARIANT A (baseline): StandardScaler fit on SIMULATED data, applied to DMC as-is")
    print("=" * 90)
    preds_a = []
    for seed in SEEDS:
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=seed)),
            ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
        ])
        pipe.fit(X_sim_raw, y_sim)
        preds_a.append(pipe.predict(X_dmc_raw))
    acc_a, macro_a, hr_a = report("Baseline (sim-fit scaler)", y_dmc, preds_a, classes, high_idx)

    # ---- Variant B: per-corpus standardization, stats from the n=25 flood set itself ----
    print("\n" + "=" * 90)
    print("VARIANT B: RF trained on sim-standardized sim data (UNCHANGED); at evaluation,")
    print("DMC features re-standardized using DMC-FLOOD-25's OWN mean/std (unsupervised,")
    print("no labels used) instead of the sim-fit scaler")
    print("=" * 90)
    dmc_scaler_25 = StandardScaler().fit(X_dmc_raw)
    X_dmc_selfstd_25 = dmc_scaler_25.transform(X_dmc_raw)
    preds_b = []
    for seed in SEEDS:
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=seed)),
            ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
        ])
        pipe.fit(X_sim_raw, y_sim)
        clf = pipe.named_steps["clf"]
        preds_b.append(clf.predict(X_dmc_selfstd_25))
    acc_b, macro_b, hr_b = report("Per-corpus standardized (stats from n=25 itself)", y_dmc, preds_b, classes, high_idx)

    # ---- Variant C: per-corpus standardization, stats from the full n=59 DMC (more stable) ----
    print("\n" + "=" * 90)
    print("VARIANT C: same idea, but DMC mean/std estimated from the FULL n=59 DMC set")
    print("(more stable estimate), then applied to standardize the n=25 flood subset for scoring")
    print("=" * 90)
    dmc_scaler_59 = StandardScaler().fit(X_dmc_all_raw)
    X_dmc_selfstd_59stats = dmc_scaler_59.transform(X_dmc_raw)
    preds_c = []
    for seed in SEEDS:
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=seed)),
            ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
        ])
        pipe.fit(X_sim_raw, y_sim)
        clf = pipe.named_steps["clf"]
        preds_c.append(clf.predict(X_dmc_selfstd_59stats))
    acc_c, macro_c, hr_c = report("Per-corpus standardized (stats from full n=59)", y_dmc, preds_c, classes, high_idx)

    # ---- Summary ----
    print("\n" + "=" * 90)
    print("SUMMARY -- acoustic-only, DMC flood-25")
    print("=" * 90)
    print(f"  {'Variant':<45}{'Accuracy':>10}{'MacroF1':>10}{'HighRec':>10}")
    print(f"  {'A: baseline (sim-fit scaler)':<45}{acc_a:>10.4f}{macro_a:>10.4f}{hr_a:>10.4f}")
    print(f"  {'B: self-standardized (n=25 stats)':<45}{acc_b:>10.4f}{macro_b:>10.4f}{hr_b:>10.4f}")
    print(f"  {'C: self-standardized (n=59 stats)':<45}{acc_c:>10.4f}{macro_c:>10.4f}{hr_c:>10.4f}")
    print(f"\n  n=25 -- each call = {100/25:.1f} accuracy points.")
    print(f"\n  Reference -- text-only on this same set: acc=0.6080 macroF1=0.5907 HighRec=0.5067")
    print(f"  Reference -- late fusion (0.5/0.5) on this same set: acc=0.5440 macroF1=0.5162 HighRec=0.5067")


if __name__ == "__main__":
    main()
