"""
extract_caller_audio.py
Step 2 of the DMC caller-isolation pipeline.

Reads the human-verified worksheet (reports/pp2/dmc_verify_worksheet.xlsx),
takes every speaker marked as the caller, and writes one caller-only WAV per
call to audio/dmc_caller/.

Label handling:
  yes / yes-mixed / yes/mixed   -> caller, included
  no / no-mixed / noise / unclear -> excluded

Ten calls have the caller split across two diarization labels. All of that
speaker's segments are merged, sorted chronologically, and overlapping
intervals are unioned so no audio is duplicated.

All output is resampled to 16 kHz so the eight 8 kHz recordings match the
rest of the corpus and the 267 simulated training calls.

Usage:
  python scripts/pipeline/extract_caller_audio.py
  python scripts/pipeline/extract_caller_audio.py --raw   # mixed-call baseline

Outputs:
  audio/dmc_caller/<recording_id>_caller.wav
  audio/dmc_raw/<recording_id>_raw.wav          (with --raw)
  reports/pp2/dmc_caller_manifest.csv
"""

import os
import argparse

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

_ROOT      = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
DMC_DIR    = os.path.join(_ROOT, "audio", "dmc")
CALLER_DIR = os.path.join(_ROOT, "audio", "dmc_caller")
RAW_DIR    = os.path.join(_ROOT, "audio", "dmc_raw")
REPORTS    = os.path.join(_ROOT, "reports", "pp2")

MANIFEST  = os.path.join(REPORTS, "dmc_diarization_manifest.csv")
WORKSHEET = os.path.join(REPORTS, "dmc_verify_worksheet.xlsx")

TARGET_SR = 16000
CALLER_VALUES = {"yes", "yes-mixed"}


def normalise(value):
    """Fold label spelling variants: 'Yes/Mixed' -> 'yes-mixed'."""
    return str(value or "").strip().lower().replace("/", "-").replace(" ", "")


def to_target_sr(audio, sr):
    if sr == TARGET_SR:
        return audio
    from math import gcd
    g = gcd(TARGET_SR, sr)
    return resample_poly(audio, TARGET_SR // g, sr // g).astype(np.float32)


def merge_intervals(segments):
    """Union overlapping [start, end) intervals so audio is never duplicated."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: s[0])
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(m) for m in merged]


def load_caller_map():
    """recording_id -> set of speaker labels the human marked as caller."""
    ws = pd.read_excel(WORKSHEET, dtype=str)
    ws = ws[ws["speaker"].notna()
            & ws["speaker"].astype(str).str.startswith("SPEAKER")].copy()
    ws["norm"] = ws["is_caller"].map(normalise)

    unknown = sorted(set(ws["norm"]) - CALLER_VALUES
                     - {"no", "no-mixed", "noise", "unclear", ""})
    if unknown:
        print(f"NOTE: unrecognised label value(s) treated as non-caller: {unknown}\n")

    blank = ws[ws["norm"] == ""]
    if len(blank):
        print(f"WARNING: {len(blank)} row(s) have no label and will be excluded.\n")

    mapping = {}
    for rec_id, group in ws.groupby("recording_id"):
        mapping[str(rec_id)] = {
            row.speaker for row in group.itertuples() if row.norm in CALLER_VALUES
        }
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="store_true",
                        help="Also write whole-call audio as the contamination baseline")
    args = parser.parse_args()

    for path in (MANIFEST, WORKSHEET):
        if not os.path.exists(path):
            raise SystemExit(f"Not found: {path}")

    os.makedirs(CALLER_DIR, exist_ok=True)
    if args.raw:
        os.makedirs(RAW_DIR, exist_ok=True)

    caller_map = load_caller_map()

    man = pd.read_csv(MANIFEST, dtype={"recording_id": str, "filename": str})
    man = man[man["speaker"].notna() & (man["speaker"] != "ERROR")]
    for col in ("start", "end", "duration"):
        man[col] = pd.to_numeric(man[col], errors="coerce")
    man = man.dropna(subset=["start", "end", "duration"])

    rows = []
    print(f"{'recording':<34} {'spk':>3} {'caller':>8} {'call':>8} {'%':>4}")
    print("-" * 62)

    for rec_id, call in man.groupby("recording_id", sort=False):
        rec_id = str(rec_id)
        fname = call["filename"].iloc[0]
        wav_path = os.path.join(DMC_DIR, fname)

        if not os.path.exists(wav_path):
            print(f"{rec_id:<34}  MISSING AUDIO")
            rows.append({"recording_id": rec_id, "status": "MISSING_AUDIO"})
            continue

        callers = caller_map.get(rec_id, set())
        if not callers:
            print(f"{rec_id:<34}  NO CALLER MARKED - skipped")
            rows.append({"recording_id": rec_id, "filename": fname,
                         "status": "NO_CALLER"})
            continue

        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = to_target_sr(audio, sr)

        segs = call[call["speaker"].isin(callers)]
        intervals = merge_intervals(
            [(r.start, r.end) for r in segs.itertuples()]
        )

        chunks = []
        for start, end in intervals:
            a = max(0, int(start * TARGET_SR))
            b = min(len(audio), int(end * TARGET_SR))
            if b > a:
                chunks.append(audio[a:b])

        if not chunks:
            print(f"{rec_id:<34}  EMPTY after extraction")
            rows.append({"recording_id": rec_id, "filename": fname,
                         "status": "EMPTY"})
            continue

        caller_audio = np.concatenate(chunks)
        caller_sec = len(caller_audio) / TARGET_SR
        call_sec = len(audio) / TARGET_SR

        out_path = os.path.join(CALLER_DIR, f"{rec_id}_caller.wav")
        sf.write(out_path, caller_audio, TARGET_SR, subtype="PCM_16")

        if args.raw:
            sf.write(os.path.join(RAW_DIR, f"{rec_id}_raw.wav"),
                     audio, TARGET_SR, subtype="PCM_16")

        pct = 100 * caller_sec / call_sec if call_sec else 0
        flag = "  <-- split" if len(callers) > 1 else ""
        print(f"{rec_id:<34} {len(callers):>3} {caller_sec:>7.1f}s "
              f"{call_sec:>7.1f}s {pct:>3.0f}%{flag}")

        rows.append({
            "recording_id":       rec_id,
            "filename":           fname,
            "caller_wav":         f"audio/dmc_caller/{rec_id}_caller.wav",
            "caller_speakers":    "+".join(sorted(callers)),
            "n_caller_speakers":  len(callers),
            "caller_duration_s":  round(caller_sec, 1),
            "call_duration_s":    round(call_sec, 1),
            "caller_pct":         round(pct, 1),
            "source_sr":          sr,
            "status":             "OK",
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(REPORTS, "dmc_caller_manifest.csv"), index=False)

    ok = out[out["status"] == "OK"] if "status" in out else out
    print("-" * 62)
    print(f"Extracted {len(ok)}/{len(out)} calls -> audio/dmc_caller/")
    if len(ok):
        print(f"  caller audio: {ok['caller_duration_s'].sum() / 60:.1f} min total, "
              f"median {ok['caller_duration_s'].median():.1f}s, "
              f"min {ok['caller_duration_s'].min():.1f}s, "
              f"max {ok['caller_duration_s'].max():.1f}s")
        print(f"  split-caller calls: {(ok['n_caller_speakers'] > 1).sum()}")
        resampled = (ok["source_sr"] != TARGET_SR).sum()
        if resampled:
            print(f"  resampled to {TARGET_SR} Hz: {resampled} call(s)")
    if args.raw:
        print(f"Baseline whole-call audio -> audio/dmc_raw/")
    print("\nManifest -> reports/pp2/dmc_caller_manifest.csv")


if __name__ == "__main__":
    main()
