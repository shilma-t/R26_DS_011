"""
step2_audit.py
Generates reports/audit_transcripts.xlsx for manual Step 2 review.
- 30 Sinhala calls (10 High, 10 Medium, 10 Low) via Sinhala fine-tune ASR
- 10 Tamil calls  (4 High, 3 Medium, 3 Low)  via Whisper-base
Columns to fill: Usable, Meaning_OK, Notes
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import librosa

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(_ROOT, "audio", "clean")
LABELS    = os.path.join(_ROOT, "labels.csv")
OUTPUT    = os.path.join(_ROOT, "reports", "audit_transcripts.xlsx")
TARGET_SR = 16_000

_whisper   = None
_sinhala   = None

def get_whisper():
    global _whisper
    if _whisper is None:
        import whisper
        print("Loading Whisper-base...")
        _whisper = whisper.load_model("base")
    return _whisper

def get_sinhala():
    global _sinhala
    if _sinhala is None:
        from transformers import pipeline
        print("Loading Sinhala fine-tune...")
        _sinhala = pipeline(
            "automatic-speech-recognition",
            model="Lingalingeswaran/whisper-small-sinhala_v3",
        )
    return _sinhala

def transcribe(wav_path, language):
    y, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    if language == "sinhala":
        res = get_sinhala()(y.astype(np.float32))
        return res.get("text", "").strip()
    else:
        res = get_whisper().transcribe(y.astype(np.float32), task="transcribe")
        return res.get("text", "").strip()

def sample_calls(df, language, counts):
    lang_df = df[df["language"] == language]
    parts = []
    for label, n in counts.items():
        subset = lang_df[lang_df["urgency_label"] == label]
        parts.append(subset.head(n))
    return pd.concat(parts, ignore_index=True)

def main():
    df = pd.read_csv(LABELS)

    sin_sample = sample_calls(df, "sinhala", {"High": 10, "Medium": 10, "Low": 10})
    tam_sample = sample_calls(df, "tamil",   {"High": 4,  "Medium": 3,  "Low": 3})

    rows = []
    total = len(sin_sample) + len(tam_sample)
    done  = 0

    for _, row in pd.concat([sin_sample, tam_sample], ignore_index=True).iterrows():
        fname = row["filename"]
        lang  = row["language"]
        label = row["urgency_label"]
        wav   = os.path.join(CLEAN_DIR, os.path.splitext(fname)[0] + ".wav")

        done += 1
        print(f"[{done}/{total}] {fname} [{lang}] [{label}]")

        if not os.path.exists(wav):
            print(f"  SKIP — no audio file")
            continue

        text = transcribe(wav, lang)
        print(f"  {text[:90]}")

        rows.append({
            "filename":      fname,
            "language":      lang,
            "urgency_label": label,
            "transcript":    text,
            "Usable":        "",   # fill: Usable / Partial / Unusable
            "Meaning_OK":    "",   # fill: Yes / No
            "Notes":         "",   # fill: anything unusual
        })

    out = pd.DataFrame(rows)
    out.to_excel(OUTPUT, index=False, engine="openpyxl")
    print(f"\nSaved {len(out)} rows to {OUTPUT}")
    print("Open the file and fill in: Usable, Meaning_OK, Notes for each row.")

if __name__ == "__main__":
    main()
