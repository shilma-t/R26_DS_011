"""
hazard_keywords_fused_test.py
Follow-up to hazard_type_keywords_test.py: same tree-hazard/tsunami-mention
features, but now tested FUSED with the 91 acoustic features (early
concatenation), not text-only. Compares:
  - Condition C (19, text-only, no hazard keywords)
  - Condition E (21, text-only, +hazard keywords)
  - Acoustic + Condition C (110, fused, no hazard keywords)
  - Acoustic + Condition E (112, fused, +hazard keywords)
on BOTH simulated 5-fold CV and DMC, same 5-seed rigor as everything else.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
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
NEW_FEATURES = ["sin_tree_hazard_count", "sin_tsunami_mention_count"]
CONDITION_E = CONDITION_C + NEW_FEATURES
FUSED_C = ACOUSTIC + CONDITION_C
FUSED_E = ACOUSTIC + CONDITION_E

TREE_HAZARD_WORDS = ["ගහක්", "ගස්", "කඩන්නක්", "කඩන්නාව", "වැටිලා", "පුළුගහක්"]
TSUNAMI_WORDS = ["සුනාමි", "සුළං", "සුළි"]


def fire_smoke_match(text):
    kws = ["fire","smoke","burning","flames","gas","ගිනිගන්නව","ගිනිඅරගෙන",
           "ගින්නක්","ගිනි","දුමක්","දුම","தீ","புகை","gini"]
    if not isinstance(text, str): return 0
    t = text.lower()
    return int(any(k.lower() in t for k in kws))


def tree_hazard_count(text):
    if not isinstance(text, str): return 0
    return sum(1 for w in TREE_HAZARD_WORDS if w in text)


def tsunami_mention_count(text):
    if not isinstance(text, str): return 0
    return sum(1 for w in TSUNAMI_WORDS if w in text)


def add_new_features(df):
    df = df.copy()
    df["sin_tree_hazard_count"] = df["transcript"].map(tree_hazard_count)
    df["sin_tsunami_mention_count"] = df["transcript"].map(tsunami_mention_count)
    return df


def cv_eval_simulated(sim, y_sim, cols):
    X = sim[cols].fillna(0).values
    accs, macros, hrs = [], [], []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        proba = np.zeros((len(y_sim), 3))
        for tr, te in skf.split(X, y_sim):
            pipe = ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=seed)),
                ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
            ])
            pipe.fit(X[tr], y_sim[tr])
            proba[te] = pipe.predict_proba(X[te])
        pred = proba.argmax(axis=1)
        accs.append(accuracy_score(y_sim, pred))
        macros.append(f1_score(y_sim, pred, average="macro"))
        hrs.append((pred[y_sim == 0] == 0).mean())
    return accs, macros, hrs


def dmc_eval(sim, dmc, y_sim, y_dmc, cols, le):
    X_sim = sim[cols].fillna(0).values
    X_dmc = dmc[cols].fillna(0).values
    accs, macros, hrs, mrs = [], [], [], []
    medium_idx = list(le.classes_).index("Medium")
    for seed in SEEDS:
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=seed)),
            ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
        ])
        pipe.fit(X_sim, y_sim)
        pred = pipe.predict(X_dmc)
        accs.append(accuracy_score(y_dmc, pred))
        macros.append(f1_score(y_dmc, pred, average="macro"))
        hrs.append((pred[y_dmc == 0] == 0).mean())
        mrs.append((pred[y_dmc == medium_idx] == medium_idx).mean())
    return accs, macros, hrs, mrs


def line(name, accs, macros, hrs, mrs=None):
    s = f"  {name:<32} acc={np.mean(accs):.4f}  macroF1={np.mean(macros):.4f}  HR={np.mean(hrs):.4f}"
    if mrs is not None:
        s += f"  MedR={np.mean(mrs):.4f}"
    print(s)


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim = sim.copy()
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    sim = add_new_features(sim)
    y_sim = le.transform(sim["urgency_label"].values)

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    dmc = add_new_features(dmc)
    y_dmc = le.transform(dmc["true_label"].values)

    conditions = [
        ("Condition C (19, text-only)", CONDITION_C),
        ("Condition E (21, text+hazard)", CONDITION_E),
        ("FUSED C (110, acoustic+text)", FUSED_C),
        ("FUSED E (112, acoustic+text+hazard)", FUSED_E),
    ]

    print("=" * 78)
    print("SIMULATED 5-fold CV")
    print("=" * 78)
    sim_results = {}
    for name, cols in conditions:
        accs, macros, hrs = cv_eval_simulated(sim, y_sim, cols)
        sim_results[name] = (accs, macros, hrs)
        line(name, accs, macros, hrs)

    print("\n" + "=" * 78)
    print("DMC (production model, 5 seeds)")
    print("=" * 78)
    dmc_results = {}
    for name, cols in conditions:
        accs, macros, hrs, mrs = dmc_eval(sim, dmc, y_sim, y_dmc, cols, le)
        dmc_results[name] = (accs, macros, hrs, mrs)
        line(name, accs, macros, hrs, mrs)

    print("\n" + "=" * 78)
    print("Does fusion (with or without hazard keywords) beat text-only on DMC?")
    print("=" * 78)
    text_only_macro = np.mean(dmc_results["Condition C (19, text-only)"][1])
    text_hazard_macro = np.mean(dmc_results["Condition E (21, text+hazard)"][1])
    for name in ["FUSED C (110, acoustic+text)", "FUSED E (112, acoustic+text+hazard)"]:
        m = np.mean(dmc_results[name][1])
        print(f"  {name}: macroF1={m:.4f}  vs text-only={text_only_macro:.4f}  "
              f"vs text+hazard={text_hazard_macro:.4f}  "
              f"-> {'BEATS text-only' if m > text_only_macro else 'still below text-only'}")


if __name__ == "__main__":
    main()
