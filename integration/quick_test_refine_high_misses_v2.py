"""
quick_test_refine_high_misses_v2.py
Same 7-call test as quick_test_refine_high_misses.py, using safe_refine()
(refusal-detected, falls back to raw draft) instead of the raw pipeline call.
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

TARGETS = [
    "Call recording 251127_191819", "Call recording 251128_173111",
    "Call recording 251129_111451", "Call recording 7251128_172437",
    "Call recording 7_251128_092626", "Call recording _251128_055921",
    "Call recording _251129_092106",
]

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

    print(f"Testing {len(TARGETS)} known High/ASR_FAILURE misses (refusal-safe) ...\n")
    flipped = 0
    fell_back_count = 0
    for rec_id in TARGETS:
        if rec_id not in dmc_eval.index:
            print(f"  SKIP {rec_id}")
            continue

        old_row = dmc_eval.loc[rec_id]
        acoustic_vals = old_row[ACOUSTIC].astype(float)

        path = os.path.join(CALLER_DIR, f"{rec_id}_caller.wav")
        lang = str(meta.loc[rec_id, "language"]).lower().strip() if rec_id in meta.index else "sinhala"
        lang_code = {"sinhala": "si-LK", "tamil": "ta-IN", "english": "en-US"}.get(lang, "si-LK")

        result = safe_refine(path, lang_code, file_stem=rec_id)
        text = result["text"]
        if result["fell_back"]:
            fell_back_count += 1

        textual = text_features_from_transcript(text, lang_code)
        row = {**acoustic_vals.to_dict(),
              **{k: v for k, v in textual.items() if k not in ("transcript", "language_normalised")},
              "fire_smoke_match": fire_smoke_match(text)}
        vector = np.array([[float(row.get(c, 0.0) or 0.0) for c in FEATURE_COLS]])
        proba = model.predict_proba(vector)[0]
        new_pred = classes[int(proba.argmax())]

        old_pred = old_row["predicted_label"]
        true_label = old_row["true_label"]
        status = "FIXED" if (old_pred != true_label and new_pred == true_label) else \
                 ("still wrong" if new_pred != true_label else "already correct")
        if status == "FIXED":
            flipped += 1

        tag = "REFUSED->raw" if result["fell_back"] else ("refined" if result["refined"] else "raw/normalize")
        print(f"  {rec_id[:32]:<32} true={true_label}  old={old_pred} -> new={new_pred}  [{status}]  ({tag})")
        print(f"      old kw={old_row['keyword_count']:.1f}  new kw={textual['keyword_count']:.1f}")
        print(f"      text: {text[:70]}")

    print(f"\n{flipped} / {len(TARGETS)} previously-missed High calls now correctly classified.")
    print(f"{fell_back_count} / {len(TARGETS)} calls had a refusal caught and fell back to raw ASR.")


if __name__ == "__main__":
    main()
