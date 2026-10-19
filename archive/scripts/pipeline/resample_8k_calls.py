"""
resample_8k_calls.py
Upsample 8 kHz mu-law DMC recordings to 16 kHz PCM.

Eight of the DMC recordings arrived as 8 kHz mu-law telephony audio while the
other 51 are 16 kHz. pyannote's segmentation and embedding models are trained
at 16 kHz, and speaker embeddings degrade badly on 8 kHz input - four of these
eight diarized incorrectly.

Resampling cannot restore the missing 4-8 kHz band, but it does put the audio
into the format the models expect, which usually recovers most of the loss.

Originals are never modified. Output goes to audio/dmc_16k/.

Usage:
  python scripts/pipeline/resample_8k_calls.py
  python scripts/pipeline/resample_8k_calls.py --apply   # copy into audio/dmc/

Without --apply this only writes audio/dmc_16k/ and reports what it found.
With --apply the originals are backed up to audio/dmc_8k_original/ and the
16 kHz versions take their place in audio/dmc/, so downstream scripts pick
them up with no path changes.
"""

import os
import shutil
import argparse

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_ROOT     = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
DMC_DIR   = os.path.join(_ROOT, "audio", "dmc")
OUT_DIR   = os.path.join(_ROOT, "audio", "dmc_16k")
BACKUP    = os.path.join(_ROOT, "audio", "dmc_8k_original")

TARGET_SR = 16000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Back up originals and swap the 16 kHz versions into audio/dmc/")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    targets = []
    for fname in sorted(os.listdir(DMC_DIR)):
        if not fname.lower().endswith(".wav"):
            continue
        info = sf.info(os.path.join(DMC_DIR, fname))
        if info.samplerate != TARGET_SR:
            targets.append((fname, info.samplerate, info.subtype))

    if not targets:
        print(f"Nothing to do - every WAV in audio/dmc/ is already {TARGET_SR} Hz.")
        return

    print(f"Found {len(targets)} file(s) below {TARGET_SR} Hz:\n")

    for fname, src_sr, subtype in targets:
        src_path = os.path.join(DMC_DIR, fname)

        # soundfile decodes mu-law to float automatically
        audio, sr = sf.read(src_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if TARGET_SR % sr == 0:
            up, down = TARGET_SR // sr, 1
        else:
            from math import gcd
            g = gcd(TARGET_SR, sr)
            up, down = TARGET_SR // g, sr // g

        resampled = resample_poly(audio, up, down).astype(np.float32)

        # resample_poly can overshoot slightly at transients
        peak = np.abs(resampled).max()
        if peak > 0.999:
            resampled = resampled * (0.999 / peak)

        out_path = os.path.join(OUT_DIR, fname)
        sf.write(out_path, resampled, TARGET_SR, subtype="PCM_16")

        print(f"  {fname}")
        print(f"    {sr} Hz {subtype}  ->  {TARGET_SR} Hz PCM_16"
              f"   ({len(audio)/sr:.1f}s, x{up}/{down})")

    print(f"\nWrote {len(targets)} file(s) to audio/dmc_16k/")

    if not args.apply:
        print("\nOriginals untouched. Re-run with --apply to swap these into")
        print("audio/dmc/ (originals are backed up first) so the diarization")
        print("and feature scripts pick them up without any path changes.")
        return

    os.makedirs(BACKUP, exist_ok=True)
    moved = 0
    for fname, _, _ in targets:
        src = os.path.join(DMC_DIR, fname)
        backup_path = os.path.join(BACKUP, fname)

        if os.path.exists(backup_path):
            print(f"  backup already exists, not overwriting: {fname}")
        else:
            shutil.copy2(src, backup_path)

        shutil.copy2(os.path.join(OUT_DIR, fname), src)
        moved += 1

    print(f"\nBacked up {len(targets)} original(s) to audio/dmc_8k_original/")
    print(f"Swapped {moved} file(s) into audio/dmc/ at {TARGET_SR} Hz.")
    print("\nNext: re-diarize just these files, then regenerate their clips.")


if __name__ == "__main__":
    main()
