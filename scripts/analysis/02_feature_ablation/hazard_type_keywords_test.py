"""
hazard_type_keywords_test.py
Adds two NEW, separate keyword categories discovered from the DMC
categorization pass:
  - tree/structural hazard (observed DMC Medium calls: "ගහක් කඩන්නක්" =
    "a tree breaking", "ගහත් වැටිලා" = "a tree has fallen", "පුළුගහක් බර
    වෙනවා" = "a tree/pole is leaning heavily") -> no immediate life threat,
    maps toward Medium per Shilma's correction.
  - tsunami/wind hazard mention (observed DMC Low call: a tsunami-threat
    status inquiry, "no impact" - an informational/status call, not itself
    a danger signal) -> maps toward Low.

These are added as NEW features (sin_tree_hazard_count,
sin_tsunami_mention_count), NOT folded into the existing danger_count/
high_risk_phrase_count/context_score arithmetic - those existing buckets
contribute a full point toward context_score (which pushes toward High),
and tree/tsunami mentions should NOT automatically do that.

IMPORTANT CAVEAT (confirmed before running): the 267-call simulated
training corpus contains ZERO tsunami mentions and only 2-3 tree mentions
- essentially no training exposure for the RF to learn a real association
from. This test is run anyway as a documented confirmatory check, per
Shilma's explicit request, alongside the existing Condition C baseline.
Tested on BOTH simulated (5-fold CV) and DMC (production, 5 seeds).
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

CONDITION_C = ["asr_quality", "keyword_count", "repetition_rate", "sentiment_polarity",
    "sin_context_score", "sin_danger_count", "sin_help_request_count",
    "sin_high_risk_phrase_count", "sin_location_context_count",
    "sin_person_at_risk_count", "sin_safety_state_count", "tam_context_score",
    "tam_danger_count", "tam_help_request_count", "tam_high_risk_phrase_count",
    "tam_location_context_count", "tam_person_at_risk_count",
    "tam_safety_state_count", "fire_smoke_match"]
NEW_FEATURES = ["sin_tree_hazard_count", "sin_tsunami_mention_count"]
CONDITION_E = CONDITION_C + NEW_FEATURES

# Words observed directly in the DMC transcripts categorized earlier.
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


def cv_eval_simulated(sim, y_sim, cols, le):
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

    print("=" * 78)
    print("Training-data coverage check (v1 Whisper transcripts, official features.csv)")
    print("=" * 78)
    print(f"  sim tree_hazard hits: {(sim['sin_tree_hazard_count']>0).sum()}/267")
    print(f"  sim tsunami hits: {(sim['sin_tsunami_mention_count']>0).sum()}/267")
    print(f"  DMC tree_hazard hits: {(dmc['sin_tree_hazard_count']>0).sum()}/59")
    print(f"  DMC tsunami hits: {(dmc['sin_tsunami_mention_count']>0).sum()}/59")

    print("\n" + "=" * 78)
    print("SIMULATED 5-fold CV: Condition C (19) vs Condition E (21, +tree/tsunami)")
    print("=" * 78)
    accs_c, macros_c, hrs_c = cv_eval_simulated(sim, y_sim, CONDITION_C, le)
    accs_e, macros_e, hrs_e = cv_eval_simulated(sim, y_sim, CONDITION_E, le)
    print(f"  Condition C: acc={np.mean(accs_c):.4f}  macroF1={np.mean(macros_c):.4f}  HR={np.mean(hrs_c):.4f}")
    print(f"  Condition E: acc={np.mean(accs_e):.4f}  macroF1={np.mean(macros_e):.4f}  HR={np.mean(hrs_e):.4f}")

    print("\n" + "=" * 78)
    print("DMC: Condition C (19) vs Condition E (21, +tree/tsunami)")
    print("=" * 78)
    accs_c, macros_c, hrs_c, mrs_c = dmc_eval(sim, dmc, y_sim, y_dmc, CONDITION_C, le)
    accs_e, macros_e, hrs_e, mrs_e = dmc_eval(sim, dmc, y_sim, y_dmc, CONDITION_E, le)
    print(f"  Condition C: acc={np.mean(accs_c):.4f}  macroF1={np.mean(macros_c):.4f}  "
          f"HR={np.mean(hrs_c):.4f}  MedR={np.mean(mrs_c):.4f}")
    print(f"  Condition E: acc={np.mean(accs_e):.4f}  macroF1={np.mean(macros_e):.4f}  "
          f"HR={np.mean(hrs_e):.4f}  MedR={np.mean(mrs_e):.4f}")

    acc_w = sum(1 for a, b in zip(accs_e, accs_c) if a >= b)
    macro_w = sum(1 for a, b in zip(macros_e, macros_c) if a >= b)
    print(f"\n  Condition E >= Condition C on DMC acc: {acc_w}/5,  macroF1: {macro_w}/5")


if __name__ == "__main__":
    main()
