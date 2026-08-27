"""
compression_ratio_feature_test.py
Adds a character-level repetition/hallucination detector (zlib compression
ratio of the transcript - highly repetitive text compresses much more than
natural language, and it catches repetition regardless of whitespace, unlike
the existing word-token-based repetition_rate which missed 2 of 5 known
hallucination-loop DMC calls entirely).

Tests Condition C + compression_ratio (20 features) against Condition C
alone (19 features) on DMC, same 5-seed rigor, and specifically checks
whether the 5 known text-only High->Low failure calls get fixed.

Also builds features.csv-side compression_ratio for the simulated corpus
(scripted calls, should compress far less than hallucinated garbage - i.e.
have HIGH compression_ratio, near 1.0, since real sentences don't repeat).
"""
import zlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
CONDITION_D = CONDITION_C + ["compression_ratio"]

TARGET_CALLS = [
    "Call recording tamil",
    "Call recording 7251128_172437",
    "Call recording 7_251128_092626",
    "Call recording _251129_092106",
    "Call recording _251128_055921",
]


def fire_smoke_match(text):
    kws = ["fire","smoke","burning","flames","gas","ගිනිගන්නව","ගිනිඅරගෙන",
           "ගින්නක්","ගිනි","දුමක්","දුම","தீ","புகை","gini"]
    if not isinstance(text, str): return 0
    t = text.lower()
    return int(any(k.lower() in t for k in kws))


def compression_ratio(text):
    if not isinstance(text, str) or len(text) == 0:
        return 1.0
    raw = text.encode("utf-8")
    compressed = zlib.compress(raw, 9)
    return len(compressed) / len(raw)


def fit_predict_proba(X_train, y_train, X_test, seed):
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=seed)),
        ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_test)


def multiseed_dmc(sim, dmc, y_sim, y_dmc, cols, le):
    X_sim = sim[cols].fillna(0).values
    X_dmc = dmc[cols].fillna(0).values
    accs, macros, hrs, mrs = [], [], [], []
    medium_idx = list(le.classes_).index("Medium")
    preds = []
    for seed in SEEDS:
        proba = fit_predict_proba(X_sim, y_sim, X_dmc, seed)
        pred = proba.argmax(axis=1)
        preds.append(pred)
        accs.append(accuracy_score(y_dmc, pred))
        macros.append(f1_score(y_dmc, pred, average="macro"))
        hrs.append((pred[y_dmc == 0] == 0).mean())
        mrs.append((pred[y_dmc == medium_idx] == medium_idx).mean())
    return {"accs": accs, "macros": macros, "hrs": hrs, "mrs": mrs, "preds": preds}


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim = sim.copy()
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    sim["compression_ratio"] = sim["transcript"].map(compression_ratio)
    y_sim = le.transform(sim["urgency_label"].values)

    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    dmc["compression_ratio"] = dmc["transcript"].map(compression_ratio)
    y_dmc = le.transform(dmc["true_label"].values)
    dmc_reset = dmc.reset_index(drop=True)

    print("=" * 78)
    print("Sanity check: simulated (scripted) vs DMC (real) compression_ratio")
    print("=" * 78)
    print(f"  Simulated mean: {sim['compression_ratio'].mean():.4f}  "
          f"(scripted calls should compress LESS -> ratio closer to 1.0)")
    print(f"  DMC mean: {dmc['compression_ratio'].mean():.4f}")

    print("\n" + "=" * 78)
    print("DMC results: Condition C (19) vs Condition C + compression_ratio (20)")
    print("=" * 78)
    res_c = multiseed_dmc(sim, dmc, y_sim, y_dmc, CONDITION_C, le)
    res_d = multiseed_dmc(sim, dmc, y_sim, y_dmc, CONDITION_D, le)

    print(f"\n  Condition C (19)              : acc={np.mean(res_c['accs']):.4f}  "
          f"macroF1={np.mean(res_c['macros']):.4f}  HR={np.mean(res_c['hrs']):.4f}  "
          f"MedR={np.mean(res_c['mrs']):.4f}")
    print(f"  Condition C + compression (20): acc={np.mean(res_d['accs']):.4f}  "
          f"macroF1={np.mean(res_d['macros']):.4f}  HR={np.mean(res_d['hrs']):.4f}  "
          f"MedR={np.mean(res_d['mrs']):.4f}")

    acc_w = sum(1 for a, b in zip(res_d["accs"], res_c["accs"]) if a >= b)
    macro_w = sum(1 for a, b in zip(res_d["macros"], res_c["macros"]) if a >= b)
    hr_w = sum(1 for a, b in zip(res_d["hrs"], res_c["hrs"]) if a >= b)
    print(f"\n  Condition D >= Condition C on acc    : {acc_w}/5")
    print(f"  Condition D >= Condition C on macroF1: {macro_w}/5")
    print(f"  Condition D >= Condition C on HighRec: {hr_w}/5")

    print("\n" + "=" * 78)
    print("Does adding compression_ratio fix any of the 5 target calls?")
    print("=" * 78)
    id_to_pos = {rid: i for i, rid in enumerate(dmc_reset["recording_id"])}
    classes = list(le.classes_)
    for rid in TARGET_CALLS:
        if rid not in id_to_pos:
            continue
        pos = id_to_pos[rid]
        preds_c = [classes[res_c["preds"][s][pos]] for s in range(len(SEEDS))]
        preds_d = [classes[res_d["preds"][s][pos]] for s in range(len(SEEDS))]
        times_high_c = sum(1 for p in preds_c if p == "High")
        times_high_d = sum(1 for p in preds_d if p == "High")
        print(f"  {rid}")
        print(f"    Condition C  predictions: {preds_c}  (High in {times_high_c}/5)")
        print(f"    Condition D  predictions: {preds_d}  (High in {times_high_d}/5)")


if __name__ == "__main__":
    main()
