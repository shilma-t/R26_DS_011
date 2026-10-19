"""
make_verify_clips.py
Cut one short audio clip per speaker so caller/operator can be judged by ear.

Each clip stitches that speaker's longest segments together, so you hear only
that voice with no timestamp hunting. Clips are numbered in the same order as
the verification worksheet, so you can work straight down both lists.

Run build_verify_sheet.py afterwards to produce the matching worksheet.

Usage:
  python scripts/pipeline/make_verify_clips.py            # pilot manifest
  python scripts/pipeline/make_verify_clips.py --full     # all 59 calls

Output:
  audio/dmc_verify/NNN_<recording_id>__SPEAKER_XX.wav
"""

import os
import argparse

import numpy as np
import pandas as pd
import soundfile as sf

_ROOT   = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
DMC_DIR = os.path.join(_ROOT, "audio", "dmc")
OUT_DIR = os.path.join(_ROOT, "audio", "dmc_verify")
REPORTS = os.path.join(_ROOT, "reports", "pp2")

PILOT_CSV = os.path.join(REPORTS, "dmc_pilot_diarization.csv")
FULL_CSV  = os.path.join(REPORTS, "dmc_diarization_manifest.csv")

MAX_CLIP_SEC = 12.0   # enough to identify a voice, short enough to get through 130 of them
MIN_SEG_SEC  = 1.0    # skip backchannels and overlap fragments
GAP_SEC      = 0.25   # silence inserted between stitched segments


def safe_name(text):
    """Filename-safe recording id. Must match build_verify_sheet.py."""
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(text))


def load_manifest(path):
    df = pd.read_csv(path, dtype={"recording_id": str, "filename": str})
    df = df[df["speaker"].notna() & (df["speaker"] != "ERROR")]
    for col in ("start", "end", "duration"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["start", "end", "duration"])


def build_clip(audio, sr, segments):
    """Stitch a speaker's longest segments up to MAX_CLIP_SEC."""
    gap = np.zeros(int(GAP_SEC * sr), dtype=audio.dtype)
    pieces = []
    total = 0.0

    for seg in segments:
        start = max(0, int(seg["start"] * sr))
        end   = min(len(audio), int(seg["end"] * sr))
        if end <= start:
            continue
        pieces.append(audio[start:end])
        pieces.append(gap)
        total += (end - start) / sr
        if total >= MAX_CLIP_SEC:
            break

    return np.concatenate(pieces) if pieces else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Use the full manifest instead of the pilot one")
    args = parser.parse_args()

    src = FULL_CSV if args.full else PILOT_CSV
    if not os.path.exists(src):
        raise SystemExit(
            f"Manifest not found: {src}\n"
            "Run diarize_dmc.py first (or drop --full to use the pilot)."
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_manifest(src)

    written = 0
    skipped = []

    # sort=False keeps manifest order, which build_verify_sheet.py also uses
    for index, (rec_id, call) in enumerate(df.groupby("recording_id", sort=False), start=1):
        fname = call["filename"].iloc[0]
        wav_path = os.path.join(DMC_DIR, fname)

        if not os.path.exists(wav_path):
            skipped.append(f"{rec_id}: audio file missing")
            continue

        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Speakers ordered by total speech, so the dominant voice is clip 1
        totals = call.groupby("speaker")["duration"].sum().sort_values(ascending=False)

        print(f"[{index:>3}] {rec_id}")

        for speaker in totals.index:
            sgrp = call[call["speaker"] == speaker]

            usable = sgrp[sgrp["duration"] >= MIN_SEG_SEC]
            if usable.empty:
                usable = sgrp

            picks = usable.nlargest(3, "duration").sort_values("start")
            clip = build_clip(audio, sr, picks.to_dict("records"))

            if clip is None:
                skipped.append(f"{rec_id} / {speaker}: no usable segments")
                print(f"        {speaker}: skipped, no usable segments")
                continue

            out_name = f"{index:03d}_{safe_name(rec_id)}__{speaker}.wav"
            sf.write(os.path.join(OUT_DIR, out_name), clip, sr)
            written += 1
            print(f"        {speaker}: {len(clip) / sr:4.1f}s  ->  {out_name}")

    print("\n" + "=" * 62)
    print(f"Wrote {written} clips to audio/dmc_verify/")
    if skipped:
        print(f"\n{len(skipped)} skipped:")
        for line in skipped:
            print(f"  - {line}")
    print("=" * 62)
    print("\nNext: python scripts/pipeline/build_verify_sheet.py"
          + (" --full" if args.full else ""))


if __name__ == "__main__":
    main()
