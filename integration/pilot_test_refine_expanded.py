"""
pilot_test_refine_expanded.py
Expanded pilot: does refinement generalize beyond the 7 known-worst High
failures, and does it risk REGRESSING calls that already transcribe well?

Sample (12 NEW tests; the 7 High ASR_FAILURE calls already have results
from quick_test_refine_high_misses_v2.py and are not re-run here):
  4 Medium ASR_FAILURE misses
  4 Low ASR_FAILURE misses
  4 already-CORRECT calls (regression check - does refinement ever break
    a call that the raw pipeline already got right?)
"""
import os
import sys

os.environ["USE_VERIFIED_REFERENCES"] = "false"

import joblib
import numpy as np
import pandas as pd

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
PIPE = os.path.join(ROOT, "scripts", "pipeline")
INTEGRATION = os.path.join(ROOT, "integration")
BACKEND = r"C:\Users\shafr\Desktop\component 2\backend"
for p in (PIPE, INTEGRATION, BACKEND):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(BACKEND)
from dotenv import load_dotenv
load_dotenv()
os.environ["USE_VERIFIED_REFERENCES"] = "false"

from text_features_from_transcript import text_features_from_transcript, fire_smoke_match
from safe_refine import safe_refine

MODEL_PATH = os.path.join(ROOT, "models", "urgency_rf_v1.joblib")
CALLER_DIR = os.path.join(ROOT, "audio", "dmc_caller")
DMC_EVAL = os.path.join(ROOT, "reports", "pp2", "dmc_evaluation.csv")
METADATA = os.path.join(ROOT, "data", "dmc_metadata.csv")

TARGETS = {
    "Medium_ASR_FAILURE": [
        "Call recording 0_251129_090737", "Call recording 251128_064500",
        "Call recording _251128_154736", "Call recording _251129_103931",
    ],
    "Low_ASR_FAILURE": [
        "890001000492661", "890001000492665",
        "890001000492675", "Call recording 0_251128_090910",
    ],
    "Already_CORRECT_regression_check": [
        "890001000492658", "890001000492722",
        "890001000492735", "890001000492744",
    ],
}

ACOUSTIC = ([f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")] +
    ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
     "spec_centroid_mean", "speaking_rate", "jitter",
     "rms_slope", "rms_gradient", "speaking_rate_change"])
TEXT = ["asr_quality", "keyword_count", "word_count", "repetition_rate", "sentiment_polarity"]
CONTEXT = ["sin_context_score", "sin_danger_count", "sin_help_request_count",
    "sin_high_risk_phrase_count", "sin_location_context_count",
    "sin_person_at_risk_count", "sin_safety_state_count",
    "tam_context_score", "tam_danger_count", "tam_help_request_count",
    "tam_high_risk_phrase_count", "tam_location_context_count",
    "tam_person_at_risk_count", "tam_safety_state_count"]
FEATURE_COLS = ACOUSTIC + TEXT + CONTEXT + ["fire_smoke_match"]


def main():
    payload = joblib.load(MODEL_PATH)
    model, classes = payload["model"], payload["classes"]

    dmc_eval = pd.read_csv(DMC_EVAL).set_index("recording_id")
    meta = pd.read_csv(METADATA, dtype={"recording_id": str})
    meta["stem"] = meta["filename"].map(lambda f: os.path.splitext(f)[0])
    meta = meta.set_index("stem")

    tally = {"fixed": 0, "broke": 0, "unchanged_correct": 0,
            "unchanged_wrong": 0, "fell_back": 0}

    for group, ids in TARGETS.items():
        print(f"\n{'='*70}\n{group}\n{'='*70}")
        for rec_id in ids:
            if rec_id not in dmc_eval.index:
                print(f"  SKIP {rec_id}: not found"); continue

            old_row = dmc_eval.loc[rec_id]
            acoustic_vals = old_row[ACOUSTIC].astype(float)
            path = os.path.join(CALLER_DIR, f"{rec_id}_caller.wav")
            lang = str(meta.loc[rec_id, "language"]).lower().strip() if rec_id in meta.index else "sinhala"
            lang_code = {"sinhala": "si-LK", "tamil": "ta-IN", "english": "en-US"}.get(lang, "si-LK")

            result = safe_refine(path, lang_code, file_stem=rec_id)
            text = result["text"]
            if result["fell_back"]:
                tally["fell_back"] += 1

            textual = text_features_from_transcript(text, lang_code)
            row = {**acoustic_vals.to_dict(),
                  **{k: v for k, v in textual.items() if k not in ("transcript", "language_normalised")},
                  "fire_smoke_match": fire_smoke_match(text)}
            vector = np.array([[float(row.get(c, 0.0) or 0.0) for c in FEATURE_COLS]])
            proba = model.predict_proba(vector)[0]
            new_pred = classes[int(proba.argmax())]

            old_pred, true_label = old_row["predicted_label"], old_row["true_label"]
            old_correct = old_pred == true_label
            new_correct = new_pred == true_label

            if not old_correct and new_correct:
                status = "FIXED"; tally["fixed"] += 1
            elif old_correct and not new_correct:
                status = "BROKE (regression!)"; tally["broke"] += 1
            elif old_correct and new_correct:
                status = "unchanged (still correct)"; tally["unchanged_correct"] += 1
            else:
                status = "unchanged (still wrong)"; tally["unchanged_wrong"] += 1

            tag = "REFUSED->raw" if result["fell_back"] else ("refined" if result["refined"] else "raw/normalize")
            print(f"  {rec_id[:32]:<32} true={true_label:<7} old={old_pred:<7} new={new_pred:<7} [{status}] ({tag})")
            print(f"      old kw={old_row['keyword_count']:.1f}  new kw={textual['keyword_count']:.1f}  text: {text[:60]}")

    print(f"\n{'='*70}\nPILOT SUMMARY (12 new calls)\n{'='*70}")
    for k, v in tally.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
