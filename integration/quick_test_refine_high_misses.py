"""
quick_test_refine_high_misses.py
Fast, decisive test: does GPT-4o refinement fix the SPECIFIC High-severity
calls we already diagnosed as ASR_FAILURE, using the EXISTING v1 model
(no retrain needed)?

Acoustic features are reused unchanged from dmc_evaluation.csv (proven
byte-identical to the fresh extraction, per earlier verification). Only
text/context features are recomputed from the refined transcript, then
scored with the cached v1 model.

This answers "is the 5-hour full run worth it" in ~10-15 minutes.
"""
import os
import sys

os.environ["USE_VERIFIED_REFERENCES"] = "false"

import joblib
import numpy as np
import pandas as pd

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
BACKEND = r"C:\Users\shafr\Desktop\component 2\backend"
PIPE = os.path.join(ROOT, "scripts", "pipeline")
for p in (PIPE, BACKEND):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(BACKEND)
from dotenv import load_dotenv
load_dotenv()
os.environ["USE_VERIFIED_REFERENCES"] = "false"

from text_features_from_transcript import text_features_from_transcript, fire_smoke_match

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
    import services.transcript_refiner as tr
    assert tr.USE_VERIFIED_REFERENCES is False
    from services.speech_service import transcribe_audio_file

    payload = joblib.load(MODEL_PATH)
    model, classes = payload["model"], payload["classes"]

    dmc_eval = pd.read_csv(DMC_EVAL).set_index("recording_id")
    meta = pd.read_csv(METADATA, dtype={"recording_id": str})
    meta["stem"] = meta["filename"].map(lambda f: os.path.splitext(f)[0])
    meta = meta.set_index("stem")

    print(f"Testing {len(TARGETS)} known High/ASR_FAILURE misses ...\n")
    flipped = 0
    for rec_id in TARGETS:
        if rec_id not in dmc_eval.index:
            print(f"  SKIP {rec_id}: not in dmc_evaluation.csv")
            continue

        old_row = dmc_eval.loc[rec_id]
        acoustic_vals = old_row[ACOUSTIC].astype(float)

        path = os.path.join(CALLER_DIR, f"{rec_id}_caller.wav")
        lang = str(meta.loc[rec_id, "language"]).lower().strip() if rec_id in meta.index else "sinhala"
        lang_code = {"sinhala": "si-LK", "tamil": "ta-IN", "english": "en-US"}.get(lang, "si-LK")

        result = transcribe_audio_file(path, file_name=f"{rec_id}.wav", forced_language=lang_code)
        new_text = result.get("text") or ""
        textual = text_features_from_transcript(new_text, lang_code)

        row = {**acoustic_vals.to_dict(),
              **{k: v for k, v in textual.items() if k not in ("transcript", "language_normalised")},
              "fire_smoke_match": fire_smoke_match(new_text)}
        vector = np.array([[float(row.get(c, 0.0) or 0.0) for c in FEATURE_COLS]])
        proba = model.predict_proba(vector)[0]
        new_pred = classes[int(proba.argmax())]

        old_pred = old_row["predicted_label"]
        true_label = old_row["true_label"]
        status = "FIXED" if (old_pred != true_label and new_pred == true_label) else \
                 ("still wrong" if new_pred != true_label else "already correct")
        if status == "FIXED":
            flipped += 1

        print(f"  {rec_id[:32]:<32} true={true_label}  old_pred={old_pred} -> new_pred={new_pred}  [{status}]")
        print(f"      old kw={old_row['keyword_count']:.1f}  new kw={textual['keyword_count']:.1f}")
        print(f"      new text: {new_text[:70]}")

    print(f"\n{flipped} / {len(TARGETS)} previously-missed High calls now correctly classified.")


if __name__ == "__main__":
    main()
