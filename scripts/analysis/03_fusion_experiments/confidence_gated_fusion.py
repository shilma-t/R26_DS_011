"""
confidence_gated_fusion.py
Confidence-gated (per-call adaptive) fusion, using an RMS-based audio-quality
proxy instead of asr_quality (which turned out to be nearly constant on DMC:
58/59 calls = 1.0, no usable spread - see prior diagnostic).

Gate signal: rms_cv = rms_std / rms_mean (coefficient of variation of frame
energy) - a proxy for how consistent/clean the recording's energy envelope
is, computed per call from features already saved (no new audio work).

Protocol, designed to stay DMC-blind during tuning:
  1. Normalize the gate using SIMULATED min/max only (fixed linear scaling,
     not optimized against any outcome) -> g in [0,1].
  2. Determine the gate's DIRECTION (does higher rms_cv mean "trust acoustic
     more" or "trust text more"?) using ONLY simulated 5-fold CV: compare,
     per call, whether the acoustic-only model or the text-only model (both
     already-used Condition C) got that call right, and correlate that with
     rms_cv. This uses only simulated labels, never DMC, to decide direction.
  3. Apply the resulting gate function on DMC exactly once: per call,
     weight_acoustic = g or (1-g) per the chosen direction, fused =
     weight_acoustic*P_acoustic + (1-weight_acoustic)*P_text.
  4. Compare against: text-only alone, unweighted late fusion (0.5/0.5),
     and the earlier CV-tuned fixed-weight late fusion (w=0.7, which failed).

All disposable/in-memory - frozen v1 artifacts untouched.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
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


def fire_smoke_match(text):
    kws = ["fire","smoke","burning","flames","gas","ගිනිගන්නව","ගිනිඅරගෙන",
           "ගින්නක්","ගිනි","දුමක්","දුම","தீ","புகை","gini"]
    if not isinstance(text, str): return 0
    t = text.lower()
    return int(any(k.lower() in t for k in kws))


def cv_proba(X, y, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out = np.zeros((len(y), 3))
    for train_idx, test_idx in skf.split(X, y):
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=seed)),
            ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        out[test_idx] = pipe.predict_proba(X[test_idx])
    return out


def fit_predict_proba(X_train, y_train, X_test, seed):
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=seed)),
        ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_test)


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim = sim.copy()
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    sim["rms_cv"] = sim["rms_std"] / sim["rms_mean"].replace(0, np.nan)
    sim["rms_cv"] = sim["rms_cv"].fillna(sim["rms_cv"].median())
    y_sim = le.transform(sim["urgency_label"].values)

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    dmc["rms_cv"] = dmc["rms_std"] / dmc["rms_mean"].replace(0, np.nan)
    dmc["rms_cv"] = dmc["rms_cv"].fillna(dmc["rms_cv"].median())
    y_dmc = le.transform(dmc["true_label"].values)
    medium_idx = list(le.classes_).index("Medium")

    print("=" * 78)
    print("STEP 1: normalize gate using SIMULATED min/max only (fixed scaling)")
    print("=" * 78)
    sim_min, sim_max = sim["rms_cv"].min(), sim["rms_cv"].max()
    print(f"  simulated rms_cv range: [{sim_min:.4f}, {sim_max:.4f}]")
    print(f"  DMC rms_cv range: [{dmc['rms_cv'].min():.4f}, {dmc['rms_cv'].max():.4f}] "
          f"(will clip to [0,1] after scaling by sim range)")

    def normalize(series):
        g = (series - sim_min) / (sim_max - sim_min)
        return g.clip(0, 1)

    sim["gate_raw"] = normalize(sim["rms_cv"])
    dmc["gate_raw"] = normalize(dmc["rms_cv"])

    print("\n" + "=" * 78)
    print("STEP 2: determine gate DIRECTION using SIMULATED 5-fold CV only")
    print("(does higher rms_cv correlate with acoustic being MORE often correct")
    print(" than text, or LESS? Decided from sim labels only, never DMC)")
    print("=" * 78)
    X_sim_ac = sim[ACOUSTIC].fillna(0).values
    X_sim_tx = sim[CONDITION_C].fillna(0).values

    correlations = []
    for seed in SEEDS:
        proba_ac = cv_proba(X_sim_ac, y_sim, seed)
        proba_tx = cv_proba(X_sim_tx, y_sim, seed)
        acoustic_correct = (proba_ac.argmax(axis=1) == y_sim).astype(int)
        text_correct = (proba_tx.argmax(axis=1) == y_sim).astype(int)
        advantage = acoustic_correct - text_correct  # +1 = acoustic right, text wrong; -1 = reverse
        corr = np.corrcoef(sim["gate_raw"].values, advantage)[0, 1]
        correlations.append(corr)
        print(f"  seed={seed:<4} corr(rms_cv_gate, acoustic_advantage_over_text) = {corr:+.4f}")

    mean_corr = np.mean(correlations)
    direction = 1 if mean_corr > 0 else -1
    print(f"\n  mean correlation = {mean_corr:+.4f}  ->  direction = {direction:+d}")
    if direction == 1:
        print("  Higher rms_cv => acoustic tends to be MORE reliable => weight_acoustic = gate")
    else:
        print("  Higher rms_cv => acoustic tends to be LESS reliable => weight_acoustic = 1-gate")

    def gate_to_weight(g):
        return g if direction == 1 else (1 - g)

    sim["weight_acoustic"] = sim["gate_raw"].map(gate_to_weight)
    dmc["weight_acoustic"] = dmc["gate_raw"].map(gate_to_weight)

    print(f"\n  DMC weight_acoustic distribution: mean={dmc['weight_acoustic'].mean():.3f}  "
          f"std={dmc['weight_acoustic'].std():.3f}  "
          f"min={dmc['weight_acoustic'].min():.3f}  max={dmc['weight_acoustic'].max():.3f}")

    print("\n" + "=" * 78)
    print("STEP 3: apply the per-call gate to DMC (production models, 5 seeds)")
    print("=" * 78)
    X_dmc_ac = dmc[ACOUSTIC].fillna(0).values
    X_dmc_tx = dmc[CONDITION_C].fillna(0).values
    w_dmc = dmc["weight_acoustic"].values.reshape(-1, 1)

    gated_accs, gated_macros, gated_hrs, gated_mrs = [], [], [], []
    uw_accs, uw_macros, uw_hrs, uw_mrs = [], [], [], []
    text_accs, text_macros, text_hrs, text_mrs = [], [], [], []
    last_gated_pred = None

    for seed in SEEDS:
        proba_ac_dmc = fit_predict_proba(X_sim_ac, y_sim, X_dmc_ac, seed)
        proba_tx_dmc = fit_predict_proba(X_sim_tx, y_sim, X_dmc_tx, seed)

        gated = w_dmc * proba_ac_dmc + (1 - w_dmc) * proba_tx_dmc
        unweighted = 0.5 * proba_ac_dmc + 0.5 * proba_tx_dmc

        pred_gated = gated.argmax(axis=1)
        pred_uw = unweighted.argmax(axis=1)
        pred_text = proba_tx_dmc.argmax(axis=1)
        last_gated_pred = pred_gated

        for accs, macros, hrs, mrs, pred in [
            (gated_accs, gated_macros, gated_hrs, gated_mrs, pred_gated),
            (uw_accs, uw_macros, uw_hrs, uw_mrs, pred_uw),
            (text_accs, text_macros, text_hrs, text_mrs, pred_text),
        ]:
            accs.append(accuracy_score(y_dmc, pred))
            macros.append(f1_score(y_dmc, pred, average="macro"))
            hrs.append((pred[y_dmc == 0] == 0).mean())
            mrs.append((pred[y_dmc == medium_idx] == medium_idx).mean())

        print(f"  seed={seed:<4} gated: acc={gated_accs[-1]:.4f} macroF1={gated_macros[-1]:.4f} "
              f"HR={gated_hrs[-1]:.4f} MedR={gated_mrs[-1]:.4f}   |   "
              f"unweighted: acc={uw_accs[-1]:.4f} HR={uw_hrs[-1]:.4f}   |   "
              f"text-only: acc={text_accs[-1]:.4f} HR={text_hrs[-1]:.4f}")

    print(f"\n  MEAN gated fusion   : acc={np.mean(gated_accs):.4f}  macroF1={np.mean(gated_macros):.4f}  "
          f"HR={np.mean(gated_hrs):.4f}  MedR={np.mean(gated_mrs):.4f}")
    print(f"  MEAN unweighted 0.5/0.5: acc={np.mean(uw_accs):.4f}  macroF1={np.mean(uw_macros):.4f}  "
          f"HR={np.mean(uw_hrs):.4f}  MedR={np.mean(uw_mrs):.4f}")
    print(f"  MEAN text-only alone : acc={np.mean(text_accs):.4f}  macroF1={np.mean(text_macros):.4f}  "
          f"HR={np.mean(text_hrs):.4f}  MedR={np.mean(text_mrs):.4f}")

    print(f"\n  gated >= text-only on acc    : {sum(g>=t for g,t in zip(gated_accs,text_accs))}/5")
    print(f"  gated >= text-only on macroF1: {sum(g>=t for g,t in zip(gated_macros,text_macros))}/5")
    print(f"  gated >= text-only on HighRec: {sum(g>=t for g,t in zip(gated_hrs,text_hrs))}/5")
    print(f"  gated >= unweighted on acc    : {sum(g>=u for g,u in zip(gated_accs,uw_accs))}/5")
    print(f"  gated >= unweighted on macroF1: {sum(g>=u for g,u in zip(gated_macros,uw_macros))}/5")

    print(f"\nConfusion matrix, gated fusion, seed={SEEDS[-1]}:")
    cm = confusion_matrix(y_dmc, last_gated_pred, labels=le.transform(le.classes_))
    print("classes:", list(le.classes_))
    print(cm)
    print(classification_report(y_dmc, last_gated_pred, target_names=le.classes_, digits=3))


if __name__ == "__main__":
    main()
