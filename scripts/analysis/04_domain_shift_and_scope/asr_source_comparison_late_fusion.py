"""
asr_source_comparison_late_fusion.py
Targeted experiment: does the deployed late-fusion model's text branch
perform differently when fed the OWN-ASR transcripts it was trained/
evaluated on (Lingalingeswaran/whisper-small-sinhala_v3, dmc_evaluation.csv)
versus Component 2's (the teammate's) fine-tuned faster-whisper transcripts
(dmc_evaluation_v2.csv) -- which is what actually reaches this model in live
deployment, since urgency_bridge.py never re-transcribes.

Only the TEXT branch's input changes between the two conditions. The
acoustic branch, the model itself, and the fusion recipe (0.5/0.5, default
params) are held completely identical -- this isolates the ASR-source
effect from every other variable already investigated.

Verified before running: dmc_evaluation.csv and dmc_evaluation_v2.csv are
row-aligned by recording_id (same order, same 59 calls), and the two ASR
transcripts are genuinely different for the same audio.

Evaluated on the n=11 keyword-matched fire+flood subset and the n=25
authoritative flood-only subset -- the same two "most trustworthy" views
used throughout this session.
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
    return acc, macro, r


def evaluate(X_sim_tx, X_sim_ac, y_sim, X_dmc_tx, X_dmc_ac, y_dmc):
    preds = []
    for seed in SEEDS:
        pipe_tx = fit_fresh(X_sim_tx, y_sim, seed)
        pipe_ac = fit_fresh(X_sim_ac, y_sim, seed)
        proba_tx = pipe_tx.predict_proba(X_dmc_tx)
        proba_ac = pipe_ac.predict_proba(X_dmc_ac)
        preds.append((0.5 * proba_tx + 0.5 * proba_ac).argmax(axis=1))
    accs, macros, recs = [], [], []
    for pred in preds:
        acc, macro, r = full_metrics(y_dmc, pred)
        accs.append(acc); macros.append(macro); recs.append(r)
    return np.mean(accs), np.mean(macros), np.mean(recs, axis=0)


def print_row(name, acc, macro, rec, high_idx):
    print(f"  {name:<40} acc={acc:.4f}  macroF1={macro:.4f}  "
          f"HighRec={rec[high_idx]:.4f}  LowRec={rec[1]:.4f}  MedRec={rec[2]:.4f}")


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    high_idx = list(le.classes_).index("High")

    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    if "fire_smoke_match" not in sim.columns:
        sim = sim.copy()
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
    y_sim = le.transform(sim["urgency_label"].values)
    X_sim_tx = sim[CONDITION_C].fillna(0).values
    X_sim_ac = sim[ACOUSTIC].fillna(0).values

    dmc_own = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")       # own ASR
    dmc_friend = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation_v2.csv") # friend's ASR
    assert (dmc_own["recording_id"].astype(str).values ==
            dmc_friend["recording_id"].astype(str).values).all(), "row misalignment!"

    meta = pd.read_csv(f"{ROOT}\\data\\dmc_metadata.csv")

    # n=11: keyword-matched fire+flood. Category MUST be decided from the
    # FRIEND'S-ASR transcript (dmc_evaluation_v2.csv) -- every prior script this
    # session (full_metrics_sim_and_dmc_scopematched.py etc.) categorized this way,
    # not from dmc_evaluation.csv's own-ASR transcript. Matching that exactly is
    # what guarantees this is the SAME 11 calls already reported throughout, so
    # only the ASR-source effect is isolated here, not a different subset too.
    category = dmc_friend["transcript"].map(categorize)
    in_scope_11 = category.isin(FLOOD_FIRE_CATEGORIES)

    # n=25: authoritative flood-only
    disaster_type = dmc_own.merge(meta[["recording_id", "disaster_type"]],
                                  on="recording_id", how="left")["disaster_type"]
    in_scope_25 = (disaster_type == "Flood").values
    in_scope_59 = np.ones(len(dmc_own), dtype=bool)  # full set, no filtering

    for label, mask, n in [("n=11 (keyword fire+flood)", in_scope_11, 11),
                           ("n=25 (authoritative flood)", in_scope_25, 25),
                           ("n=59 (full DMC)", in_scope_59, 59)]:
        print("\n" + "=" * 90)
        print(f"{label}")
        print("=" * 90)

        y_dmc = le.transform(dmc_own.loc[mask, "true_label"].values)
        X_ac = dmc_own.loc[mask, ACOUSTIC].fillna(0).values  # acoustic identical either way

        X_tx_own = dmc_own.loc[mask, CONDITION_C].fillna(0).values
        X_tx_friend = dmc_friend.loc[mask, CONDITION_C].fillna(0).values

        acc, macro, rec = evaluate(X_sim_tx, X_sim_ac, y_sim, X_tx_own, X_ac, y_dmc)
        print_row("OWN ASR (trained/evaluated on this)", acc, macro, rec, high_idx)

        acc, macro, rec = evaluate(X_sim_tx, X_sim_ac, y_sim, X_tx_friend, X_ac, y_dmc)
        print_row("FRIEND'S ASR (what live deployment sends)", acc, macro, rec, high_idx)

        print(f"  ({n} calls -- each = {100/n:.1f} accuracy points)")


if __name__ == "__main__":
    main()
