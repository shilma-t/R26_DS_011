"""
convert_dmc_m4a.py
Convert all M4A files in audio/dmc/ to WAV (16kHz mono) using ffmpeg.
Also checks channel count on existing WAV files.
Output WAVs go to the same folder with same base name.
"""
import os
import subprocess
import soundfile as sf

DMC_DIR = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011\audio\dmc"

def convert_m4a(src, dst):
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-ar", "16000",   # 16 kHz
        "-ac", "1",       # mono
        dst
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def main():
    files = os.listdir(DMC_DIR)
    m4a_files = [f for f in files if f.lower().endswith(".m4a")]
    wav_files  = [f for f in files if f.lower().endswith(".wav")]

    print(f"M4A files to convert: {len(m4a_files)}")
    print(f"WAV files already:    {len(wav_files)}\n")

    # Convert M4A → WAV
    ok, fail = 0, 0
    for fname in sorted(m4a_files):
        src = os.path.join(DMC_DIR, fname)
        dst = os.path.join(DMC_DIR, os.path.splitext(fname)[0] + ".wav")
        if os.path.exists(dst):
            print(f"  SKIP (already exists): {fname}")
            ok += 1
            continue
        success = convert_m4a(src, dst)
        if success:
            info = sf.info(dst)
            print(f"  OK  {fname} → {info.channels}ch {info.samplerate}Hz {round(info.duration,1)}s")
            ok += 1
        else:
            print(f"  FAIL {fname}")
            fail += 1

    print(f"\nConversion done: {ok} OK, {fail} failed")

    # Re-check all WAVs now
    print("\nFull WAV audit:")
    all_wavs = [f for f in os.listdir(DMC_DIR) if f.lower().endswith(".wav")]
    stereo, mono = 0, 0
    for fname in sorted(all_wavs):
        info = sf.info(os.path.join(DMC_DIR, fname))
        tag = "STEREO" if info.channels >= 2 else "mono"
        print(f"  {tag}  {fname}  {info.channels}ch {info.samplerate}Hz {round(info.duration,1)}s")
        if info.channels >= 2:
            stereo += 1
        else:
            mono += 1

    print(f"\nStereo: {stereo}  |  Mono: {mono}")
    if stereo > 0:
        print("→ Stereo files detected. We can extract the caller channel only.")
    else:
        print("→ All mono. Will use full audio; document as limitation in paper.")

if __name__ == "__main__":
    main()
