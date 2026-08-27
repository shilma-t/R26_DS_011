"""
build_features_v2_refined.py
Same purpose as build_features_v2.py, but goes through Component 2's FULL
transcription pipeline (multi-engine ASR + GPT-4o corpus-guided refinement)
instead of the raw single-engine call - i.e. exactly what the live demo and
any deployed system actually run.

Monkeypatches the OpenAI client to record real token usage per call, so the
benchmark reports MEASURED cost, not an estimate.

Usage:
  python integration/build_features_v2_refined.py --n 5   # benchmark only
  python integration/build_features_v2_refined.py          # full 267 calls
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

os.chdir(BACKEND)
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

# ── token usage capture: wrap the OpenAI client without editing her code ────
_usage_log = []

def _install_usage_probe():
    import openai
    original_create = openai.resources.chat.completions.Completions.create

    def patched_create(self, *args, **kwargs):
        response = original_create(self, *args, **kwargs)
        try:
            u = response.usage
            _usage_log.append({
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            })
        except Exception:
            pass
        return response

    openai.resources.chat.completions.Completions.create = patched_create


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="Benchmark: only process the first N calls, no save")
    args = ap.parse_args()

    _install_usage_probe()

    from services.speech_service import transcribe_audio_file

    labels = pd.read_csv(LABELS)
    v1 = pd.read_csv(FEATURES_V1)
    v1_by_file = v1.set_index("filename")

    acoustic_cols = [c for c in v1.columns if c.startswith("mfcc_") or c in {
        "pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
        "spec_centroid_mean", "speaking_rate", "jitter",
        "rms_slope", "rms_gradient", "speaking_rate_change",
    }]

    rows = labels if args.n is None else labels.head(args.n)
    print(f"Transcribing {len(rows)} call(s) via the FULL hybrid+refine pipeline ...")

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
        before = len(_usage_log)
        t0 = time.time()
        result = transcribe_audio_file(wav, file_name=fname, forced_language=lang_code)
        dt = time.time() - t0

        text = result.get("text") or ""
        used_llm = len(_usage_log) > before
        call_tokens = _usage_log[before]["total_tokens"] if used_llm else 0

        textual = text_features_from_transcript(text, lang_code)
        acoustic = v1_by_file.loc[fname, acoustic_cols].to_dict()

        out_rows.append({
            "filename": fname,
            "urgency_label": row.urgency_label,
            "language": row.language,
            **acoustic,
            **{k: v for k, v in textual.items()
               if k not in ("transcript", "language_normalised")},
            "fire_smoke_match": fire_smoke_match(text),
            "transcript": text,
        })
        tag = f"gpt4o:{call_tokens}tok" if used_llm else "no-refine"
        print(f"  [{i}/{len(rows)}] {fname:<28} {dt:5.1f}s  {tag:<16} "
              f"kw={textual['keyword_count']:.1f}  {text[:40]}")

    elapsed = time.time() - t_start
    n_done = len(out_rows)
    per_call = elapsed / max(n_done, 1)

    total_tokens = sum(u["total_tokens"] for u in _usage_log)
    n_refined = len(_usage_log)

    print(f"\n{n_done} call(s) in {elapsed:.0f}s  ({per_call:.1f}s/call)")
    print(f"GPT-4o refinement fired on {n_refined}/{n_done} calls")
    if _usage_log:
        avg_tok = total_tokens / n_refined
        print(f"tokens used: {total_tokens} total, {avg_tok:.0f} avg/refined-call")

    if args.n:
        refine_rate = n_refined / max(n_done, 1)
        est_calls_267 = 267
        est_refined = refine_rate * est_calls_267
        est_tokens = (total_tokens / max(n_refined, 1)) * est_refined if n_refined else 0
        print(f"\nFull 267-call projection:")
        print(f"  time   : ~{per_call * 267 / 60:.0f} min")
        print(f"  tokens : ~{est_tokens:.0f}  "
              f"(based on {n_refined}/{n_done} calls actually invoking GPT-4o)")
        return

    out = pd.DataFrame(out_rows)
    out.to_csv(OUT, index=False)
    print(f"\nSaved -> {OUT}")
    print(f"mean keyword_count  v1={v1['keyword_count'].mean():.3f}  "
          f"v2={out['keyword_count'].mean():.3f}")


if __name__ == "__main__":
    main()
