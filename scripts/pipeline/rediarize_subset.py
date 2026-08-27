"""
rediarize_subset.py
Re-run diarization on a subset of calls and merge back into the master manifest.

Used after fixing an audio problem on some files (e.g. resampling the 8 kHz
mu-law recordings to 16 kHz) so those calls can be re-diarized without
discarding results - or verification labels - for everything else.

Rows for the named calls are replaced in dmc_diarization_manifest.csv.
All other rows are left exactly as they were.

Usage:
  python scripts/pipeline/rediarize_subset.py --token HF_TOKEN --prefix 890001
  python scripts/pipeline/rediarize_subset.py --token HF_TOKEN --files a.wav b.wav
"""

import os
import shutil
import argparse
from datetime import datetime

import pandas as pd
import soundfile as sf
import torch
from pyannote.audio import Pipeline

_ROOT     = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
DMC_DIR   = os.path.join(_ROOT, "audio", "dmc")
SEG_DIR   = os.path.join(_ROOT, "audio", "dmc_diarized")
REPORTS   = os.path.join(_ROOT, "reports", "pp2")
MANIFEST  = os.path.join(REPORTS, "dmc_diarization_manifest.csv")


def diarize_file(pipeline, wav_path):
    audio, sr = sf.read(wav_path, dtype="float32")
    audio = audio[None, :] if audio.ndim == 1 else audio.T
    waveform = torch.from_numpy(audio)

    result = pipeline({"waveform": waveform, "sample_rate": sr})
    annotation = result.speaker_diarization

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "start":    round(turn.start, 3),
            "end":      round(turn.end, 3),
            "duration": round(turn.end - turn.start, 3),
            "speaker":  speaker,
        })
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--prefix", help="Re-diarize every WAV whose name starts with this")
    parser.add_argument("--files", nargs="+", help="Explicit WAV filenames")
    args = parser.parse_args()

    if not args.prefix and not args.files:
        raise SystemExit("Give --prefix or --files to choose which calls to re-diarize.")

    if args.files:
        wavs = list(args.files)
    else:
        wavs = sorted(
            f for f in os.listdir(DMC_DIR)
            if f.lower().endswith(".wav") and f.startswith(args.prefix)
        )

    if not wavs:
        raise SystemExit("No matching WAV files found.")

    print(f"Re-diarizing {len(wavs)} call(s):")
    for w in wavs:
        print(f"  {w}")
    print()

    if not os.path.exists(MANIFEST):
        raise SystemExit(f"Manifest not found: {MANIFEST}")

    # Snapshot the manifest before touching it
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = MANIFEST.replace(".csv", f".backup_{stamp}.csv")
    shutil.copy2(MANIFEST, backup)
    print(f"Manifest backed up -> {os.path.basename(backup)}\n")

    print("Loading pyannote pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=args.token,
    )
    print("Pipeline loaded.\n")

    new_rows = []
    redone = []

    for i, fname in enumerate(wavs, start=1):
        wav_path = os.path.join(DMC_DIR, fname)
        rec_id = os.path.splitext(fname)[0]
        info = sf.info(wav_path)
        print(f"[{i}/{len(wavs)}] {fname}  ({info.samplerate} Hz, {info.duration:.1f}s)")

        try:
            segments = diarize_file(pipeline, wav_path)
        except Exception as error:
            print(f"    ERROR: {error}")
            continue

        if not segments:
            print("    WARNING: no segments found, leaving previous rows in place")
            continue

        durations = {}
        for s in segments:
            durations[s["speaker"]] = durations.get(s["speaker"], 0) + s["duration"]
        total = sum(durations.values())
        heuristic_caller = max(durations, key=durations.get)

        share = "  ".join(
            f"{spk} {100 * dur / total:.0f}%"
            for spk, dur in sorted(durations.items(), key=lambda kv: -kv[1])
        )
        print(f"    {len(durations)} speakers, {len(segments)} segments, "
              f"{total:.1f}s speech   |  {share}")

        call_df = pd.DataFrame(segments)
        call_df["recording_id"] = rec_id
        call_df["filename"] = fname
        call_df["heuristic_caller"] = heuristic_caller
        call_df["caller_label"] = ""
        call_df.to_csv(os.path.join(SEG_DIR, f"{rec_id}_segments.csv"), index=False)

        for s in segments:
            new_rows.append({
                "recording_id": rec_id,
                "filename": fname,
                "start": s["start"],
                "end": s["end"],
                "duration": s["duration"],
                "speaker": s["speaker"],
                "heuristic_caller": heuristic_caller,
                "caller_label": "",
                "note": f"re-diarized {stamp} at {info.samplerate} Hz",
            })
        redone.append(rec_id)

    if not new_rows:
        print("\nNothing was re-diarized. Manifest unchanged.")
        return

    manifest = pd.read_csv(MANIFEST, dtype={"recording_id": str, "filename": str})
    before = len(manifest)

    kept = manifest[~manifest["recording_id"].astype(str).isin(redone)]
    merged = pd.concat([kept, pd.DataFrame(new_rows)], ignore_index=True)
    merged.to_csv(MANIFEST, index=False)

    print("\n" + "=" * 62)
    print(f"Re-diarized  : {len(redone)} call(s)")
    print(f"Rows replaced: {before - len(kept)}  ->  {len(new_rows)}")
    print(f"Manifest     : {len(merged)} rows total")
    print("=" * 62)
    print("\nNext, regenerate clips and worksheet:")
    print("  python scripts/pipeline/make_verify_clips.py --full")
    print("  python scripts/pipeline/build_verify_sheet.py --full --force")
    print("\nNOTE: your existing labels for these calls no longer apply -")
    print("the speaker labels have been reassigned. Relabel just these rows.")


if __name__ == "__main__":
    main()
