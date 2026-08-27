"""
build_features_v3.py
v3 training features: SAME acoustic features as v1 (unchanged, copied
directly from data/features.csv), text/context features rebuilt from
GPT-4o-refined transcripts (Component 2's full hybrid ASR + refinement
pipeline), with the verified-reference exact-match shortcut EXPLICITLY
disabled so refinement is genuine GPT-4o cleanup, never a leaked ground
truth from her corpus (which shares filenames with our simulated corpus).

This isolates transcript quality as the ONLY variable versus v1: same
audio, same acoustic extraction, same model architecture - only the text
source changes. Required for a clean v1-vs-v3 comparison.

Usage:
  python integration/build_features_v3.py --n 5   # benchmark only
  python integration/build_features_v3.py           # full 267 calls
"""
import os
import sys
import time
import argparse

os.environ["USE_VERIFIED_REFERENCES"] = "false"   # MUST be set before any
                                                    # speech_service import

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
os.environ["USE_VERIFIED_REFERENCES"] = "false"    # re-assert after dotenv

from text_features_from_transcript import text_features_from_transcript, fire_smoke_match

CLEAN_DIR = os.path.join(ROOT, "audio", "clean")
LABELS = os.path.join(ROOT, "data", "labels_frozen_20260817.csv")
FEATURES_V1 = os.path.join(ROOT, "data", "features.csv")
OUT = os.path.join(ROOT, "data", "features_v3.csv")

LANG_TO_CODE = {"sinhala": "si-LK", "singlish": "si-LK",
                "tamil": "ta-IN", "tanglish": "ta-IN", "english": "en-US"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    import services.transcript_refiner as tr
    print(f"USE_VERIFIED_REFERENCES = {tr.USE_VERIFIED_REFERENCES}  (must be False)")
    assert tr.USE_VERIFIED_REFERENCES is False, "leakage guard not active - abort"

    from services.speech_service import transcribe_audio_file

    labels = pd.read_csv(LABELS)
    v1 = pd.read_csv(FEATURES_V1)
    v1_by_file = v1.set_index("filename")
    acoustic_cols = [c for c in v1.columns if c.startswith("mfcc_") or c in {
        "pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
        "spec_centroid_mean", "speaking_rate", "jitter",
        "rms_slope", "rms_gradient", "speaking_rate_change"}]

    rows = labels if args.n is None else labels.head(args.n)
    print(f"Refining {len(rows)} call(s) (leakage-safe) ...")

    out_rows = []
    t0 = time.time()
    for i, row in enumerate(rows.itertuples(), 1):
        fname = row.filename
        stem = os.path.splitext(fname)[0]
        wav = os.path.join(CLEAN_DIR, stem + ".wav")
        if not os.path.exists(wav) or fname not in v1_by_file.index:
            print(f"  [{i}] skip {fname}")
            continue

        lang_code = LANG_TO_CODE.get(str(row.language).lower().strip(), "en-US")
        tc0 = time.time()
        result = transcribe_audio_file(wav, file_name=fname, forced_language=lang_code)
        dt = time.time() - tc0
        text = result.get("text") or ""
        engine = result.get("refinement_engine")

        textual = text_features_from_transcript(text, lang_code)
        acoustic = v1_by_file.loc[fname, acoustic_cols].to_dict()
        out_rows.append({
            "filename": fname, "urgency_label": row.urgency_label, "language": row.language,
            **acoustic,
            **{k: v for k, v in textual.items() if k not in ("transcript", "language_normalised")},
            "fire_smoke_match": fire_smoke_match(text), "transcript": text,
        })
        print(f"  [{i}/{len(rows)}] {fname:<24} {dt:5.1f}s  {str(engine):<20} "
              f"kw={textual['keyword_count']:.1f}  {text[:36]}")

    elapsed = time.time() - t0
    n = len(out_rows)
    print(f"\n{n} call(s) in {elapsed:.0f}s ({elapsed/max(n,1):.1f}s/call)")
    if args.n:
        print(f"Full 267-call projection: ~{elapsed/max(n,1)*267/60:.0f} min")
        return

    out = pd.DataFrame(out_rows)
    out.to_csv(OUT, index=False)
    print(f"Saved -> {OUT}")
    print(f"mean keyword_count  v1={v1['keyword_count'].mean():.3f}  v3={out['keyword_count'].mean():.3f}")


if __name__ == "__main__":
    main()
