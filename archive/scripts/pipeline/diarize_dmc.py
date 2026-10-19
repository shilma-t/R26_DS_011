"""
diarize_dmc.py
Step 1 of the DMC caller-isolation pipeline.

For each WAV file in audio/dmc/:
  1. Run pyannote speaker diarization
  2. Save a segment manifest (recording_id, start, end, speaker, duration)
  3. Export caller-only WAV to audio/dmc_caller/

The script does NOT auto-assign which diarization label = caller.
It saves the manifest so you can manually verify a sample first.
After verification, run extract_caller_audio.py to do the final extraction.

Usage:
  python scripts/pipeline/diarize_dmc.py --token YOUR_HF_TOKEN

Outputs:
  reports/pp2/dmc_diarization_manifest.csv   — all segments, all speakers
  audio/dmc_diarized/                        — per-call segment CSVs
"""

import os
import argparse
import pandas as pd
import soundfile as sf
import numpy as np
from pyannote.audio import Pipeline

_ROOT     = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
DMC_DIR   = os.path.join(_ROOT, "audio", "dmc")
OUT_DIR   = os.path.join(_ROOT, "audio", "dmc_diarized")
REPORTS   = os.path.join(_ROOT, "reports", "pp2")

os.makedirs(OUT_DIR, exist_ok=True)


def diarize_file(pipeline, wav_path):
    """Run diarization on one WAV. Returns list of (start, end, speaker) tuples."""
    import torch
    import soundfile as sf
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim == 1:
        audio = audio[None, :]
    else:
        audio = audio.T
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="HuggingFace access token")
    args = parser.parse_args()

    print("Loading pyannote diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=args.token
    )
    print("Pipeline loaded.\n")

    wav_files = sorted([
        f for f in os.listdir(DMC_DIR)
        if f.lower().endswith(".wav")
        and not f.startswith("890001")  # skip original 8kHz WAVs for now — they need upsampling
    ])

    # Also include the original 8 WAV files
    wav_files_8k = sorted([
        f for f in os.listdir(DMC_DIR)
        if f.lower().endswith(".wav") and f.startswith("890001")
    ])

    print(f"Files to diarize: {len(wav_files)} converted + {len(wav_files_8k)} original WAVs")
    print("Processing converted M4A WAVs first (16kHz)...\n")

    all_rows = []

    for fname in wav_files + wav_files_8k:
        wav_path = os.path.join(DMC_DIR, fname)
        recording_id = os.path.splitext(fname)[0]
        print(f"  [{wav_files.index(fname) + 1 if fname in wav_files else '8k'}/{len(wav_files)}] {fname}")

        try:
            segments = diarize_file(pipeline, wav_path)
        except Exception as e:
            print(f"    ERROR: {e}")
            all_rows.append({
                "recording_id": recording_id,
                "filename": fname,
                "start": None, "end": None, "duration": None,
                "speaker": "ERROR",
                "caller_label": "",
                "note": str(e),
            })
            continue

        if not segments:
            print(f"    WARNING: no segments found")
            continue

        # Count unique speakers
        speakers = list(set(s["speaker"] for s in segments))
        total_dur = sum(s["duration"] for s in segments)

        # Heuristic: speaker with MORE total speaking time is more likely the CALLER
        # (callers explain the situation; operators ask short questions)
        # THIS IS JUST A HINT — verify manually before trusting it.
        spk_dur = {}
        for s in segments:
            spk_dur[s["speaker"]] = spk_dur.get(s["speaker"], 0) + s["duration"]
        heuristic_caller = max(spk_dur, key=spk_dur.get)

        print(f"    Speakers: {speakers}  |  total speech: {round(total_dur,1)}s")
        for spk, dur in sorted(spk_dur.items()):
            hint = " ← likely caller (more speech)" if spk == heuristic_caller else ""
            print(f"      {spk}: {round(dur,1)}s{hint}")

        # Save per-call segment CSV
        call_df = pd.DataFrame(segments)
        call_df["recording_id"] = recording_id
        call_df["filename"] = fname
        call_df["heuristic_caller"] = heuristic_caller
        call_df["caller_label"] = ""  # fill in after manual verification
        call_csv = os.path.join(OUT_DIR, f"{recording_id}_segments.csv")
        call_df.to_csv(call_csv, index=False)

        for s in segments:
            all_rows.append({
                "recording_id": recording_id,
                "filename": fname,
                "start": s["start"],
                "end": s["end"],
                "duration": s["duration"],
                "speaker": s["speaker"],
                "heuristic_caller": heuristic_caller,
                "caller_label": "",  # YOU fill this in after manual check
                "note": "",
            })

    # Save master manifest
    manifest_path = os.path.join(REPORTS, "dmc_diarization_manifest.csv")
    pd.DataFrame(all_rows).to_csv(manifest_path, index=False)

    print(f"\n{'='*65}")
    print("DIARIZATION COMPLETE")
    print(f"{'='*65}")
    print(f"Manifest saved → reports/pp2/dmc_diarization_manifest.csv")
    print(f"Per-call CSVs  → audio/dmc_diarized/")
    print()
    print("NEXT STEP — manual verification:")
    print("  1. Listen to 10-15 calls")
    print("  2. Open reports/pp2/dmc_diarization_manifest.csv")
    print("  3. Fill in 'caller_label' column with the SPEAKER_XX label that is the caller")
    print("     (one label per recording_id, e.g. SPEAKER_00 or SPEAKER_01)")
    print("  4. Run: python scripts/pipeline/extract_caller_audio.py")


if __name__ == "__main__":
    main()
