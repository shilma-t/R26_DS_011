"""
short_audio_safeguard_simulated_only.py
Investigates a principled, duration-based safeguard for the locked late-fusion
model (0.5/0.5 text+acoustic), motivated by a live demo call where a 2.7s
turn ("our house has no current" -- clearly Low by text: danger_count=0,
context_score=0, text_proba favoured Low at 0.437) was pushed to High purely
by the acoustic branch (proba High=0.432 on that short clip).

Rule: tested and frozen on SIMULATED DATA ONLY (5-fold CV), never on DMC.
DMC (n=59, n=11) is touched exactly once at the end, for reporting only --
matching the "do not tune on DMC" discipline used throughout this project.

Candidate safeguard: below a duration threshold T, downweight the acoustic
branch (make fusion more text-heavy) instead of the fixed 0.5/0.5. Several
(T, short_weight) pairs are swept on simulated CV; DMC is evaluated only for
the single best config chosen from simulated results.
"""
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
AUDIO_DIR = os.path.join(ROOT, "audio")
DMC_AUDIO_DIR = os.path.join(ROOT, "audio", "dmc_16k")
RF_BASE = dict(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1)
SEEDS = [1, 7, 42, 99, 123]

ACOUSTIC = ([f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")] +
    ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
     "spec_centroid_mean", "speaking_rate", "jitter",
     "rms_slope", "rms_gradient", "speaking_rate_change"])
TEXT_ONLY_COLS = ["asr_quality", "keyword_count", "repetition_rate", "sentiment_polarity",
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


def duration_seconds(path):
    try:
        return librosa.get_duration(path=path)
    except Exception:
        return np.nan


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


def full_metrics(y_true, pred, high_idx):
    acc = accuracy_score(y_true, pred)
    macro = f1_score(y_true, pred, average="macro")
    p, r, _, _ = precision_recall_fscore_support(y_true, pred, labels=[0, 1, 2], zero_division=0)
    return acc, macro, r[high_idx]


def fused_pred(proba_text, proba_acoustic, durations, threshold, short_text_weight):
    """Below threshold seconds: use short_text_weight for text (rest to acoustic).
    At/above threshold: fixed 0.5/0.5, unchanged from the locked model."""
    w_text = np.where(durations < threshold, short_text_weight, 0.5)
    w_text = w_text[:, None]
    fused = w_text * proba_text + (1 - w_text) * proba_acoustic
    return fused.argmax(axis=1)


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    high_idx = list(le.classes_).index("High")

    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    y_sim = le.transform(sim["urgency_label"].values)
    X_sim_tx = sim[TEXT_ONLY_COLS].fillna(0).values
    X_sim_ac = sim[ACOUSTIC].fillna(0).values

    print("Computing simulated-corpus durations ...")
    sim_durations = sim["filename"].map(lambda f: duration_seconds(os.path.join(AUDIO_DIR, f))).values
    print(f"  simulated: n={len(sim_durations)}  "
          f"min={np.nanmin(sim_durations):.1f}s  p10={np.nanpercentile(sim_durations,10):.1f}s  "
          f"median={np.nanmedian(sim_durations):.1f}s  max={np.nanmax(sim_durations):.1f}s")
    n_short = {}
    for t in [2, 3, 4, 5, 6]:
        n_short[t] = int((sim_durations < t).sum())
    print(f"  count below threshold: " + "  ".join(f"<{t}s:{n}" for t, n in n_short.items()))
    print()

    # ---- baseline: plain 0.5/0.5 late fusion, simulated 5-fold CV ----
    print("=" * 90)
    print("BASELINE -- locked late fusion (0.5/0.5, all durations), simulated 5-fold CV")
    print("=" * 90)
    base_accs, base_macros, base_highs = [], [], []
    for seed in SEEDS:
        proba_tx = cv_proba(X_sim_tx, y_sim, seed)
        proba_ac = cv_proba(X_sim_ac, y_sim, seed)
        pred = (0.5 * proba_tx + 0.5 * proba_ac).argmax(axis=1)
        acc, macro, high = full_metrics(y_sim, pred, high_idx)
        base_accs.append(acc); base_macros.append(macro); base_highs.append(high)
    print(f"  acc={np.mean(base_accs):.4f}  macroF1={np.mean(base_macros):.4f}  HighRec={np.mean(base_highs):.4f}")
    print()

    # ---- sweep threshold x short_text_weight, simulated 5-fold CV only ----
    print("=" * 90)
    print("SWEEP -- short-audio safeguard, simulated 5-fold CV only (DMC untouched)")
    print("=" * 90)
    results = []
    for threshold in [2.0, 3.0, 4.0, 5.0]:
        for short_text_weight in [0.65, 0.75, 0.85]:
            accs, macros, highs = [], [], []
            for seed in SEEDS:
                proba_tx = cv_proba(X_sim_tx, y_sim, seed)
                proba_ac = cv_proba(X_sim_ac, y_sim, seed)
                pred = fused_pred(proba_tx, proba_ac, sim_durations, threshold, short_text_weight)
                acc, macro, high = full_metrics(y_sim, pred, high_idx)
                accs.append(acc); macros.append(macro); highs.append(high)
            n_affected = int((sim_durations < threshold).sum())
            row = dict(threshold=threshold, short_text_weight=short_text_weight,
                       n_affected=n_affected,
                       acc=np.mean(accs), macro=np.mean(macros), high=np.mean(highs))
            results.append(row)
            print(f"  T<{threshold:.0f}s  text_w={short_text_weight:.2f}  "
                  f"(n_affected={n_affected:3d})  "
                  f"acc={row['acc']:.4f}  macroF1={row['macro']:.4f}  HighRec={row['high']:.4f}")

    print()
    baseline_macro = np.mean(base_macros)
    improved = [r for r in results if r["macro"] > baseline_macro and r["acc"] >= np.mean(base_accs)]
    if not improved:
        print("No swept config beat the locked baseline on BOTH accuracy and macro-F1 "
              "(simulated CV). Safeguard does not clear the bar -- baseline stays as is.")
        return
    best = max(improved, key=lambda r: r["macro"])
    print(f"Best config beating baseline on simulated CV: threshold<{best['threshold']}s, "
          f"short_text_weight={best['short_text_weight']}  "
          f"acc={best['acc']:.4f} (base {np.mean(base_accs):.4f})  "
          f"macroF1={best['macro']:.4f} (base {baseline_macro:.4f})  "
          f"HighRec={best['high']:.4f} (base {np.mean(base_highs):.4f})")
    print()
    print("FREEZING this config. Testing once on DMC (n=59 full, n=11 flood/fire subset) ...")
    print()

    # ---- frozen config, single DMC pass ----
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

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    y_dmc = le.transform(dmc["true_label"].values)
    X_dmc_tx = dmc[TEXT_ONLY_COLS].fillna(0).values
    X_dmc_ac = dmc[ACOUSTIC].fillna(0).values

    print("Computing DMC durations ...")
    dmc_durations = dmc["filename"].map(lambda f: duration_seconds(os.path.join(DMC_AUDIO_DIR, f))).values
    print(f"  DMC: n={len(dmc_durations)}  min={np.nanmin(dmc_durations):.1f}s  "
          f"median={np.nanmedian(dmc_durations):.1f}s  n<{best['threshold']}s="
          f"{int((dmc_durations < best['threshold']).sum())}")
    print()

    category = dmc["transcript"].map(categorize)
    in_scope_11 = category.isin(FLOOD_FIRE_CATEGORIES).values
    in_scope_59 = np.ones(len(dmc), dtype=bool)

    for label, mask in [("n=59 (full DMC)", in_scope_59), ("n=11 (flood/fire subset)", in_scope_11)]:
        y = y_dmc[mask]
        durations = dmc_durations[mask]
        base_preds, safe_preds = [], []
        for seed in SEEDS:
            pipe_tx = ImbPipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state=seed)),
                                    ("clf", RandomForestClassifier(random_state=seed, **RF_BASE))])
            pipe_ac = ImbPipeline([("scaler", StandardScaler()), ("smote", SMOTE(random_state=seed)),
                                    ("clf", RandomForestClassifier(random_state=seed, **RF_BASE))])
            pipe_tx.fit(X_sim_tx, y_sim)
            pipe_ac.fit(X_sim_ac, y_sim)
            proba_tx = pipe_tx.predict_proba(X_dmc_tx[mask])
            proba_ac = pipe_ac.predict_proba(X_dmc_ac[mask])
            base_preds.append((0.5 * proba_tx + 0.5 * proba_ac).argmax(axis=1))
            safe_preds.append(fused_pred(proba_tx, proba_ac, durations,
                                          best["threshold"], best["short_text_weight"]))

        print(f"--- {label} ---")
        for name, preds in [("baseline (locked, 0.5/0.5 always)", base_preds),
                             ("safeguard (frozen from simulated)", safe_preds)]:
            accs = [accuracy_score(y, p) for p in preds]
            macros = [f1_score(y, p, average="macro") for p in preds]
            highs = [precision_recall_fscore_support(y, p, labels=[0,1,2], zero_division=0)[1][high_idx]
                     for p in preds]
            print(f"  {name:<36} acc={np.mean(accs):.4f}  macroF1={np.mean(macros):.4f}  "
                  f"HighRec={np.mean(highs):.4f}")
        print()


if __name__ == "__main__":
    main()
