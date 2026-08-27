"""
build_features_v2.py
Re-transcribe the 267 simulated training calls with Component 2's ASR
(faster-whisper, fine-tuned for Sinhala) and rebuild the text/context
features from those transcripts, producing a v2 feature matrix consistent
with the ASR actually used at inference time.

Acoustic features are UNCHANGED - they don't depend on ASR and are copied
from data/features.csv. Only the 5 text + 14 context + fire_smoke_match
columns are recomputed, via text_features_from_transcript.py, so the logic
is identical to the frozen pipeline's extract_textual() - just fed a
different transcript.

Why this matters: v1's classifier was trained on transcripts from
Lingalingeswaran/whisper-small-sinhala_v3. Feeding it transcripts from a
different, better ASR at inference time without retraining is the same
class of mismatch as tonight's preprocessing bug - text features would sit
in a distribution the model never saw during training.

Usage:
  python integration/build_features_v2.py --n 5      # benchmark only
  python integration/build_features_v2.py             # full 267 calls
"""
import os
import sys
import time
import argparse

import pandas as pd

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
BACKEND = r"C:\Users\shafr\Desktop\component 2\backend"
PIPE = os.path.join(ROOT, "scripts", "pipeline")

for p in (PIPE, BACKEND):
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(BACKEND)  # her code expects relative paths from backend/
from dotenv import load_dotenv
load_dotenv()

from text_features_from_transcript import (
    text_features_from_transcript,
    fire_smoke_match,
)

CLEAN_DIR = os.path.join(ROOT, "audio", "clean")
LABELS = os.path.join(ROOT, "data", "labels_frozen_20260817.csv")
FEATURES_V1 = os.path.join(ROOT, "data", "features.csv")
OUT = os.path.join(ROOT, "data", "features_v2.csv")

LANG_TO_CODE = {
    "sinhala": "si-LK", "singlish": "si-LK",
    "tamil": "ta-IN", "tanglish": "ta-IN",
    "english": "en-US",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="Benchmark: only process the first N calls, no save")
    args = ap.parse_args()

    from services.whisper_engine import transcribe_with_forced_language

    labels = pd.read_csv(LABELS)
    v1 = pd.read_csv(FEATURES_V1)
    v1_by_file = v1.set_index("filename")

    acoustic_cols = [c for c in v1.columns if c.startswith("mfcc_") or c in {
        "pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
        "spec_centroid_mean", "speaking_rate", "jitter",
        "rms_slope", "rms_gradient", "speaking_rate_change",
    }]

    rows = labels if args.n is None else labels.head(args.n)
    print(f"Re-transcribing {len(rows)} call(s) with Component 2's ASR ...")

    out_rows = []
    t_start = time.time()
    for i, row in enumerate(rows.itertuples(), 1):
        fname = row.filename
        stem = os.path.splitext(fname)[0]
        wav = os.path.join(CLEAN_DIR, stem + ".wav")
        if not os.path.exists(wav):
            print(f"  [{i}] MISSING {fname}")
            continue
        if fname not in v1_by_file.index:
            print(f"  [{i}] no v1 acoustic row for {fname}, skipping")
            continue

        lang_code = LANG_TO_CODE.get(str(row.language).lower().strip(), "en-US")
        t0 = time.time()
        result = transcribe_with_forced_language(wav, lang_code)
        dt = time.time() - t0

        textual = text_features_from_transcript(result.text, lang_code)
        acoustic = v1_by_file.loc[fname, acoustic_cols].to_dict()

        out_rows.append({
            "filename": fname,
            "urgency_label": row.urgency_label,
            "language": row.language,
            **acoustic,
            **{k: v for k, v in textual.items()
               if k not in ("transcript", "language_normalised")},
            "fire_smoke_match": fire_smoke_match(result.text),
            "transcript": result.text,
        })
        print(f"  [{i}/{len(rows)}] {fname:<28} {dt:5.1f}s  "
              f"kw={textual['keyword_count']:.1f}  {result.text[:44]}")

    elapsed = time.time() - t_start
    per_call = elapsed / max(len(out_rows), 1)
    print(f"\n{len(out_rows)} call(s) in {elapsed:.0f}s  ({per_call:.1f}s/call)")

    if args.n:
        print(f"Full 267-call run estimate: ~{per_call * 267 / 60:.0f} min")
        return

    out = pd.DataFrame(out_rows)
    out.to_csv(OUT, index=False)
    print(f"\nSaved -> {OUT}")
    print(f"mean keyword_count  v1={v1['keyword_count'].mean():.3f}  "
          f"v2={out['keyword_count'].mean():.3f}")


if __name__ == "__main__":
    main()
