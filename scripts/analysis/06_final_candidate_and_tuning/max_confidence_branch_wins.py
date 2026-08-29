"""
max_confidence_branch_wins.py
Tests a specific alternative to 0.5/0.5 late fusion: per call, compare
max(text_proba) vs max(acoustic_proba) and use the FULL prediction of
whichever branch is more confident in itself, discarding the other branch
entirely for that call. Different from the earlier confidence_gated_fusion.py
(which gated on an rms_cv audio-quality proxy, not on the branches' own
confidence).

Protocol: evaluated on simulated 5-fold CV only. DMC is touched once, at the
end, only if this beats the locked baseline on BOTH accuracy and macro-F1 on
simulated CV -- same "do not tune on DMC" discipline as the rest of this
project.
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


def max_confidence_wins(proba_text, proba_acoustic):
    text_wins = proba_text.max(axis=1) >= proba_acoustic.max(axis=1)
    pred = np.where(text_wins, proba_text.argmax(axis=1), proba_acoustic.argmax(axis=1))
    return pred, text_wins


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    high_idx = list(le.classes_).index("High")

    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim = sim.copy()
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    y_sim = le.transform(sim["urgency_label"].values)
    X_sim_tx = sim[TEXT_ONLY_COLS].fillna(0).values
    X_sim_ac = sim[ACOUSTIC].fillna(0).values

    print("=" * 90)
    print("SIMULATED 5-fold CV -- max-confidence-branch-wins vs locked 0.5/0.5 baseline")
    print("=" * 90)

    base_accs, base_macros, base_highs = [], [], []
    mc_accs, mc_macros, mc_highs = [], [], []
    text_win_fracs = []
    for seed in SEEDS:
        proba_tx = cv_proba(X_sim_tx, y_sim, seed)
        proba_ac = cv_proba(X_sim_ac, y_sim, seed)

        base_pred = (0.5 * proba_tx + 0.5 * proba_ac).argmax(axis=1)
        acc, macro, high = full_metrics(y_sim, base_pred, high_idx)
        base_accs.append(acc); base_macros.append(macro); base_highs.append(high)

        mc_pred, text_wins = max_confidence_wins(proba_tx, proba_ac)
        acc, macro, high = full_metrics(y_sim, mc_pred, high_idx)
        mc_accs.append(acc); mc_macros.append(macro); mc_highs.append(high)
        text_win_fracs.append(text_wins.mean())

        print(f"  seed={seed:<4} baseline: acc={base_accs[-1]:.4f} macroF1={base_macros[-1]:.4f} "
              f"HighRec={base_highs[-1]:.4f}   |   max-conf: acc={mc_accs[-1]:.4f} "
              f"macroF1={mc_macros[-1]:.4f} HighRec={mc_highs[-1]:.4f}  "
              f"(text won {text_wins.mean()*100:.0f}% of calls)")

    print()
    print(f"  MEAN baseline (0.5/0.5) : acc={np.mean(base_accs):.4f}  macroF1={np.mean(base_macros):.4f}  "
          f"HighRec={np.mean(base_highs):.4f}")
    print(f"  MEAN max-conf-wins      : acc={np.mean(mc_accs):.4f}  macroF1={np.mean(mc_macros):.4f}  "
          f"HighRec={np.mean(mc_highs):.4f}")
    print(f"  mean fraction of calls where text branch was more confident: "
          f"{np.mean(text_win_fracs)*100:.1f}%")
    print()

    beats_both = (np.mean(mc_accs) >= np.mean(base_accs)) and (np.mean(mc_macros) >= np.mean(base_macros))
    if not beats_both:
        print("max-confidence-branch-wins does NOT beat the locked baseline on both "
              "accuracy and macro-F1 (simulated CV). Per protocol, DMC is not touched; "
              "this stays a documented negative result.")
        return

    print("max-confidence-branch-wins beats baseline on simulated CV. "
          "Testing once on DMC (n=59 full, n=11 flood/fire subset) ...")
    print()

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
    category = dmc["transcript"].map(categorize)
    in_scope_11 = category.isin(FLOOD_FIRE_CATEGORIES).values
    in_scope_59 = np.ones(len(dmc), dtype=bool)

    for label, mask in [("n=59 (full DMC)", in_scope_59), ("n=11 (flood/fire subset)", in_scope_11)]:
        y = y_dmc[mask]
        base_preds, mc_preds = [], []
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
            mc_pred, _ = max_confidence_wins(proba_tx, proba_ac)
            mc_preds.append(mc_pred)

        print(f"--- {label} ---")
        for name, preds in [("baseline (locked, 0.5/0.5)", base_preds),
                             ("max-confidence-branch-wins", mc_preds)]:
            accs = [accuracy_score(y, p) for p in preds]
            macros = [f1_score(y, p, average="macro") for p in preds]
            highs = [precision_recall_fscore_support(y, p, labels=[0,1,2], zero_division=0)[1][high_idx]
                     for p in preds]
            print(f"  {name:<30} acc={np.mean(accs):.4f}  macroF1={np.mean(macros):.4f}  "
                  f"HighRec={np.mean(highs):.4f}")
        print()


if __name__ == "__main__":
    main()
