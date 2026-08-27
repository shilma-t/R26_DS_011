"""
test_refine_on_dmc.py
Direct test: does GPT-4o refinement actually help on DMC audio specifically?

No leakage risk here - DMC filenames (890001..., "Call recording ...") don't
exist in her verified-reference corpus, so USE_VERIFIED_REFERENCES=false is
just an extra safety guarantee, not strictly required for DMC.

Compares raw ASR vs refined ASR on the same DMC calls, side by side.
"""
import os
import sys

os.environ["USE_VERIFIED_REFERENCES"] = "false"

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
BACKEND = r"C:\Users\shafr\Desktop\component 2\backend"
PIPE = os.path.join(ROOT, "scripts", "pipeline")
for p in (PIPE, BACKEND):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(BACKEND)
from dotenv import load_dotenv
load_dotenv()

from services.whisper_engine import transcribe_with_forced_language
from services.speech_service import transcribe_audio_file
from text_features_from_transcript import text_features_from_transcript

CALLER_DIR = os.path.join(ROOT, "audio", "dmc_caller")

# Calls that produced clearly garbled raw transcripts in the v2 DMC run
TEST_CALLS = [
    ("890001000492658_caller.wav", "si-LK"),
    ("890001000492659_caller.wav", "si-LK"),
    ("Call recording _251130_150637_caller.wav", "si-LK"),
]

for fname, lang in TEST_CALLS:
    path = os.path.join(CALLER_DIR, fname)
    if not os.path.exists(path):
        print(f"MISSING: {fname}")
        continue

    print("=" * 70)
    print(fname)

    raw = transcribe_with_forced_language(path, lang)
    raw_feat = text_features_from_transcript(raw.text, lang)
    print(f"  RAW      kw={raw_feat['keyword_count']:.1f}  {raw.text}")

    result = transcribe_audio_file(path, file_name=fname, forced_language=lang)
    refined_text = result.get("text") or ""
    ref_feat = text_features_from_transcript(refined_text, lang)
    print(f"  REFINED  kw={ref_feat['keyword_count']:.1f}  engine={result.get('refinement_engine')}  {refined_text}")
    print()
