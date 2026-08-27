"""
09_compare_asr.py
Compares Whisper-base vs Lingalingeswaran/whisper-small-sinhala_v3
on all Sinhala calls in the dataset.
Reports transcript, ASR quality score, and keyword hits for each model.
"""

import os
import warnings
import numpy as np
import pandas as pd
import librosa
import whisper

warnings.filterwarnings("ignore")

CLEAN_DIR    = os.path.join("..", "audio", "clean")
LABELS_CSV   = os.path.join("..", "labels.csv")
REPORTS_DIR  = os.path.join("..", "reports")
TARGET_SR    = 16_000

SINHALA_MODEL_ID = "Lingalingeswaran/whisper-small-sinhala_v3"

URGENCY_KEYWORDS = [
    "help", "fire", "trapped", "emergency", "blood", "dying", "dead",
    "accident", "hurt", "injured", "smoke", "burning", "flood", "water",
    "ambulance", "police", "gas", "flames", "child", "children", "baby",
    "unconscious", "breathing", "cannot breathe", "stuck", "fast", "quickly",
    # Sinhala keywords (from confirmed transcripts)
    "දුමක්", "ගිනි", "උදව්", "ජලය", "දරුවා", "රෝහල",
    # Romanised Sinhala
    "api", "rush", "inna",
    # Tamil
    "உதவி", "தீ", "வெள்ளம்", "தண்ணீர்", "குழந்தை", "புகை",
]


def asr_quality_score(text: str) -> float:
    """q=1.0 for Latin/Tamil/Sinhala script; q≈0 for Arabic/Cyrillic hallucination."""
    cleaned = text.strip()
    if not cleaned:
        return 0.0
    alpha_chars = [ch for ch in cleaned if ch.isalpha()]
    if not alpha_chars:
        return 0.0
    bad = sum(1 for ch in alpha_chars
              if (0x0600 <= ord(ch) <= 0x06FF) or (0x0400 <= ord(ch) <= 0x04FF))
    return float(1.0 - bad / len(alpha_chars))


def keyword_hits(text: str) -> int:
    t = text.lower()
    return sum(1 for kw in URGENCY_KEYWORDS if kw.lower() in t)


def load_sinhala_calls():
    df = pd.read_csv(LABELS_CSV)
    sinhala = df[df["language"].str.lower() == "sinhala"].copy()
    print(f"Found {len(sinhala)} Sinhala calls in labels.csv\n")
    return sinhala


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    sinhala_df = load_sinhala_calls()
    if sinhala_df.empty:
        print("No Sinhala calls found. Check 'language' column in labels.csv.")
        return

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading Whisper-base ...")
    whisper_base = whisper.load_model("base")

    print("Loading Sinhala fine-tune ...")
    from transformers import pipeline
    sinhala_asr = pipeline("automatic-speech-recognition", model=SINHALA_MODEL_ID)
    print()

    results = []

    for _, row in sinhala_df.iterrows():
        fname  = row["filename"]
        stem   = os.path.splitext(fname)[0]
        wav    = os.path.join(CLEAN_DIR, stem + ".wav")
        label  = row["urgency_label"]

        if not os.path.exists(wav):
            print(f"  [SKIP] {fname} — no clean audio file")
            continue

        print(f"── {fname}  [{label}] ──────────────────────────────")
        y, sr = librosa.load(wav, sr=TARGET_SR, mono=True)

        # Whisper-base
        base_result  = whisper_base.transcribe(y.astype(np.float32), task="transcribe")
        base_text    = base_result.get("text", "").strip()
        base_q       = asr_quality_score(base_text)
        base_kw      = keyword_hits(base_text)

        # Sinhala fine-tune
        sin_result   = sinhala_asr(y.astype(np.float32))
        sin_text     = sin_result["text"].strip()
        sin_q        = asr_quality_score(sin_text)
        sin_kw       = keyword_hits(sin_text)

        print(f"  Whisper-base  q={base_q:.2f}  kw={base_kw}  | {base_text[:80]}")
        print(f"  Sinhala-ft    q={sin_q:.2f}  kw={sin_kw}  | {sin_text[:80]}")
        print()

        results.append({
            "filename":         fname,
            "urgency_label":    label,
            "base_transcript":  base_text,
            "base_quality":     base_q,
            "base_keywords":    base_kw,
            "sin_transcript":   sin_text,
            "sin_quality":      sin_q,
            "sin_keywords":     sin_kw,
            "quality_gain":     sin_q - base_q,
            "keyword_gain":     sin_kw - base_kw,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    df_out = pd.DataFrame(results)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Calls evaluated        : {len(df_out)}")
    print(f"\n  Whisper-base")
    print(f"    Mean quality score   : {df_out['base_quality'].mean():.3f}")
    print(f"    Calls with q=1.0     : {(df_out['base_quality'] == 1.0).sum()}")
    print(f"    Calls with q<0.5     : {(df_out['base_quality'] < 0.5).sum()}  (hallucinating)")
    print(f"    Mean keyword hits    : {df_out['base_keywords'].mean():.2f}")
    print(f"\n  Sinhala fine-tune")
    print(f"    Mean quality score   : {df_out['sin_quality'].mean():.3f}")
    print(f"    Calls with q=1.0     : {(df_out['sin_quality'] == 1.0).sum()}")
    print(f"    Calls with q<0.5     : {(df_out['sin_quality'] < 0.5).sum()}  (hallucinating)")
    print(f"    Mean keyword hits    : {df_out['sin_keywords'].mean():.2f}")
    print(f"\n  Improvement (fine-tune vs base)")
    print(f"    Mean quality gain    : {df_out['quality_gain'].mean():+.3f}")
    print(f"    Mean keyword gain    : {df_out['keyword_gain'].mean():+.2f}")
    print(f"    Calls where fine-tune is better : {(df_out['quality_gain'] > 0).sum()}")
    print(f"    Calls where fine-tune is worse  : {(df_out['quality_gain'] < 0).sum()}")

    # Per urgency class
    print(f"\n  Quality by urgency label (Sinhala fine-tune):")
    for lbl, grp in df_out.groupby("urgency_label"):
        print(f"    {lbl:>6}: mean q={grp['sin_quality'].mean():.3f}  "
              f"mean kw={grp['sin_keywords'].mean():.1f}  n={len(grp)}")

    out_path = os.path.join(REPORTS_DIR, "asr_comparison_sinhala.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
