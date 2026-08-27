"""
11_context_diagnostic.py
Quick sanity check on the updated extract_textual pipeline.
Processes 5 Sinhala + 5 Tamil + 5 English/Singlish calls,
saves reports/context_features_diagnostic.csv.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import librosa

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from context_features import extract_context_features

_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(_ROOT, "audio", "clean")
LABELS    = os.path.join(_ROOT, "labels.csv")
OUTPUT    = os.path.join(_ROOT, "reports", "context_features_diagnostic.csv")
TARGET_SR = 16_000
N_EACH    = 5


def asr_quality_score(text: str) -> float:
    cleaned = text.strip()
    if not cleaned:
        return 0.0
    alpha_chars = [ch for ch in cleaned if ch.isalpha()]
    if not alpha_chars:
        return 0.0
    bad = sum(1 for ch in alpha_chars
              if (0x0600 <= ord(ch) <= 0x06FF) or (0x0400 <= ord(ch) <= 0x04FF))
    return float(1.0 - bad / len(alpha_chars))


_whisper_model = None
_sinhala_asr   = None


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print("  Loading Whisper-base ...")
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def get_sinhala():
    global _sinhala_asr
    if _sinhala_asr is None:
        from transformers import pipeline
        print("  Loading Sinhala fine-tune ...")
        _sinhala_asr = pipeline(
            "automatic-speech-recognition",
            model="Lingalingeswaran/whisper-small-sinhala_v3",
        )
    return _sinhala_asr


def transcribe(y, language):
    lang = language.lower().strip()
    if lang == "sinhala":
        pipe = get_sinhala()
        res  = pipe(y.astype(np.float32))
        return res.get("text", "")
    else:
        model  = get_whisper()
        result = model.transcribe(y.astype(np.float32), task="transcribe")
        return result.get("text", "")


def main():
    df_labels = pd.read_csv(LABELS)

    # Sample N_EACH per language group
    rows_sin = df_labels[df_labels["language"] == "sinhala"].head(N_EACH)
    rows_tam = df_labels[df_labels["language"] == "tamil"].head(N_EACH)
    rows_eng = df_labels[~df_labels["language"].isin(["sinhala", "tamil"])].head(N_EACH)
    sample   = pd.concat([rows_sin, rows_tam, rows_eng], ignore_index=True)

    results = []
    for _, row in sample.iterrows():
        fname = row["filename"]
        lang  = row.get("language", "")
        stem  = os.path.splitext(fname)[0]
        wav   = os.path.join(CLEAN_DIR, stem + ".wav")

        if not os.path.exists(wav):
            print(f"  [SKIP] {fname}")
            continue

        print(f"\n── {fname} [{lang}] [{row['urgency_label']}] ──")
        y, _ = librosa.load(wav, sr=TARGET_SR, mono=True)

        text = transcribe(y, lang)
        q    = asr_quality_score(text)
        ctx  = extract_context_features(text, lang)

        prefix = "sin" if lang == "sinhala" else "tam"
        if lang in ("sinhala", "tamil"):
            d  = ctx[f"{prefix}_danger_count"]
            im = ctx[f"{prefix}_immediacy_count"]
            s  = ctx[f"{prefix}_safety_count"]
            ph = ctx[f"{prefix}_high_risk_phrase"]
            sc = ctx[f"{prefix}_context_score"]
        else:
            d = im = s = ph = sc = 0

        print(f"  transcript : {text[:80]}")
        print(f"  q={q:.2f}  danger={d}  immed={im}  safety={s}  phrase={ph}  score={sc}")

        results.append({
            "filename":      fname,
            "language":      lang,
            "urgency_label": row["urgency_label"],
            "transcript":    text,
            "asr_quality":   q,
            **ctx,
        })

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"\n\nDiagnostic saved to {OUTPUT}")
    print(out[["filename","language","urgency_label","asr_quality",
               "sin_danger_count","sin_context_score",
               "tam_danger_count","tam_context_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
