"""
diarize_pilot.py
Run diarization on 5 representative calls to verify the pipeline
before processing all 58 files.

Picks calls across short / medium / long durations automatically.

Usage:
  python scripts/pipeline/diarize_pilot.py --token YOUR_HF_TOKEN

Outputs:
  audio/dmc_diarized/<recording_id>_segments.csv   — per-call segment tables
  reports/pp2/dmc_pilot_diarization.csv            — combined pilot manifest
"""

import os
import argparse
import numpy as np
import pandas as pd
import soundfile as sf

_ROOT   = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
DMC_DIR = os.path.join(_ROOT, "audio", "dmc")
OUT_DIR = os.path.join(_ROOT, "audio", "dmc_diarized")
REPORTS = os.path.join(_ROOT, "reports", "pp2")
os.makedirs(OUT_DIR, exist_ok=True)


def pick_pilot_files(n=5):
    """Pick n WAV files spanning short/medium/long durations."""
    files = []
    for f in os.listdir(DMC_DIR):
        if not f.lower().endswith(".wav"):
            continue
        info = sf.info(os.path.join(DMC_DIR, f))
        files.append({"filename": f, "duration": info.duration})

    df = pd.DataFrame(files).sort_values("duration").reset_index(drop=True)
    total = len(df)

    # Pick evenly spread indices: shortest, 25th pct, median, 75th pct, longest
    indices = [0,
               total // 4,
               total // 2,
               3 * total // 4,
               total - 1]
    picked = df.iloc[indices].reset_index(drop=True)
    return picked


def diarize_file(pipeline, wav_path):
    import torch
    import soundfile as sf
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim == 1:
        audio = audio[None, :]  # (1, samples)
    else:
        audio = audio.T         # (channels, samples)
    waveform = torch.from_numpy(audio)
    diarization = pipeline({"waveform": waveform, "sample_rate": sr})
    annotation = diarization.speaker_diarization
    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "start":    round(turn.start, 3),
            "end":      round(turn.end,   3),
            "duration": round(turn.end - turn.start, 3),
            "speaker":  speaker,
        })
    return segments


def summarise(segments, recording_id, filename):
    if not segments:
        return

    df = pd.DataFrame(segments)
    spk_dur = df.groupby("speaker")["duration"].sum().sort_values(ascending=False)
    total   = df["duration"].sum()

    print(f"\n  Recording : {recording_id}")
    print(f"  Filename  : {filename}")
    print(f"  Segments  : {len(df)}  |  total speech: {round(total,1)}s")
    print(f"  Speakers  :")
    for spk, dur in spk_dur.items():
        pct = round(100 * dur / total, 1)
        hint = " ← likely CALLER (most speech)" if spk == spk_dur.index[0] else ""
        print(f"    {spk}: {round(dur,1)}s ({pct}%){hint}")

    print(f"  First 5 segments:")
    print(df.head(5).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="HuggingFace access token")
    parser.add_argument("--n", type=int, default=5, help="Number of pilot files (default 5)")
    args = parser.parse_args()

    pilot = pick_pilot_files(args.n)
    print("=" * 65)
    print(f"DIARIZATION PILOT — {len(pilot)} calls")
    print("=" * 65)
    print(pilot[["filename", "duration"]].to_string(index=False))
    print()

    print("Loading pyannote pipeline...")
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=args.token
    )
    print("Pipeline loaded.\n")

    all_rows = []

    for _, row in pilot.iterrows():
        fname  = row["filename"]
        rec_id = os.path.splitext(fname)[0]
        path   = os.path.join(DMC_DIR, fname)

        print(f"Processing: {fname}  ({round(row['duration'],1)}s)")
        try:
            segments = diarize_file(pipeline, path)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        summarise(segments, rec_id, fname)

        # Save per-call CSV
        seg_df = pd.DataFrame(segments)
        seg_df["recording_id"]      = rec_id
        seg_df["filename"]          = fname
        seg_df["caller_label"]      = ""   # fill in after listening
        seg_df["verified_by_human"] = "no"
        seg_path = os.path.join(OUT_DIR, f"{rec_id}_segments.csv")
        seg_df.to_csv(seg_path, index=False)
        print(f"  Saved → audio/dmc_diarized/{rec_id}_segments.csv")

        for s in segments:
            all_rows.append({
                "recording_id": rec_id,
                "filename":     fname,
                **s,
                "caller_label":      "",
                "verified_by_human": "no",
            })

    # Save combined pilot manifest
    out = os.path.join(REPORTS, "dmc_pilot_diarization.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)

    print("\n" + "=" * 65)
    print("PILOT COMPLETE")
    print("=" * 65)
    print(f"Saved → reports/pp2/dmc_pilot_diarization.csv")
    print()
    print("NEXT STEPS:")
    print("  1. Listen to these 5 calls")
    print("  2. Open each audio/dmc_diarized/<id>_segments.csv")
    print("  3. Fill in 'caller_label' with the SPEAKER_XX that is the caller")
    print("  4. Set verified_by_human = yes")
    print("  5. Check: does the heuristic (most speech = caller) hold?")
    print("  6. If yes → run full diarization: diarize_dmc.py --token TOKEN")
    print("  7. If no  → tell me what pattern you see and we'll adjust")


if __name__ == "__main__":
    main()
