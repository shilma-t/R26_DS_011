"""
check_dmc_audio.py
Quick audit of DMC audio files — sample rate, channels, duration.
Run this first to decide how to handle two-speaker recordings.
"""
import os
import librosa
import soundfile as sf

DMC_DIR = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011\audio\dmc"

def check(path):
    try:
        info = sf.info(path)
        return {
            "file":     os.path.basename(path),
            "channels": info.channels,
            "sr":       info.samplerate,
            "duration": round(info.duration, 1),
            "format":   info.format,
        }
    except Exception as e:
        return {"file": os.path.basename(path), "error": str(e)}

def main():
    files = [f for f in os.listdir(DMC_DIR)
             if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))]
    if not files:
        print(f"No audio files found in {DMC_DIR}")
        return

    print(f"Found {len(files)} audio files\n")
    print(f"{'File':<40} {'Ch':>4} {'SR':>8} {'Dur(s)':>8} {'Format'}")
    print("-" * 70)

    stereo_count = 0
    mono_count   = 0
    for f in sorted(files):
        r = check(os.path.join(DMC_DIR, f))
        if "error" in r:
            print(f"{r['file']:<40}  ERROR: {r['error']}")
        else:
            print(f"{r['file']:<40} {r['channels']:>4} {r['sr']:>8} {r['duration']:>8} {r['format']}")
            if r["channels"] >= 2:
                stereo_count += 1
            else:
                mono_count += 1

    print("-" * 70)
    print(f"\nStereo (2+ channels): {stereo_count}")
    print(f"Mono (1 channel)    : {mono_count}")

    if stereo_count > 0:
        print("\n✓ Stereo files found — can extract caller channel directly.")
        print("  Typically: channel 0 = caller, channel 1 = operator (verify by listening)")
    if mono_count > 0:
        print("\n⚠ Mono files — both speakers mixed. Options:")
        print("  1. Use speaker diarization (pyannote.audio) to isolate caller segments")
        print("  2. Accept mixed audio and document as a limitation")

if __name__ == "__main__":
    main()
