"""
02_extract_features.py
Extracts acoustic + textual features from each cleaned audio file
and saves the combined feature matrix to features.csv.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import librosa
import whisper

warnings.filterwarnings("ignore")

# Allow importing context_features from the same scripts/ directory
sys.path.insert(0, os.path.dirname(__file__))
from context_features import extract_context_features

CLEAN_DIR   = os.path.join("..", "audio", "clean")
LABELS      = os.path.join("..", "labels.csv")
OUTPUT_CSV  = os.path.join("..", "features.csv")
TARGET_SR   = 16_000
N_MFCC      = 40

URGENCY_KEYWORDS = [
    # ── English ───────────────────────────────────────────────────────────────
    "help", "fire", "inside", "trapped", "quickly", "emergency",
    "blood", "dying", "dead", "accident", "attack", "hurt", "injured",
    "crash", "ambulance", "police", "gas", "smoke", "burning", "flood",
    "workers", "grandmother", "children", "child", "baby", "unconscious",
    "breathing", "cannot breathe", "stuck", "still inside",
    "water", "flames", "fast", "save",

    # ── Sinhala script — extracted from Lingalingeswaran/whisper-small-sinhala_v3
    # transcripts (asr_comparison_sinhala.csv). These are actual output forms.
    "උදව්",          # help
    "උදව් කරන්න",   # give help / please help
    "බේරගන්න",       # save us
    "හදිස්සිය",      # emergency
    "ගිනිගන්නව",     # fire catching / on fire
    "ගිනිඅරගෙන",     # caught fire
    "ගින්නක්",       # a fire
    "ගිනි",          # fire (root form)
    "දුමක්",         # smoke
    "දුම",           # smoke (root)
    "වතුර ඇතුළට",   # water inside
    "ගෙදර ඇතුළට",   # inside the house
    "හිරවෙලා",       # trapped
    "සිරවෙලා",       # trapped (variant)
    "යටවෙලා",        # submerged / flooded
    "ගංවතුරට",       # river flood
    "ළමයා",          # child
    "ළමයිය",         # children
    "ළමේ",           # child (variant)
    "හුස්ම ගන්නත් අමාරු",  # can't breathe
    "හුස්ම",         # breath (breathing distress)
    "ඉක්මනට",        # quickly / hurry
    "ඉක්මනින්",      # quickly (variant)
    "කරුණාකර",       # please (jussive marker)
    "දැන්",          # now (immediacy)

    # ── Tamil script — confirmed from Whisper output ──────────────────────────
    "உதவி", "உதவி செய்யுங்கள்", "உதவுங்கள்",
    "வெள்ளம்", "தீ", "இரத்தம்", "விபத்து",
    "தண்ணீர்", "குழந்தை", "புகை",
]

# Safety qualifiers: presence downgrades urgency signal
SAFETY_QUALIFIERS = [
    # English
    "safe", "okay", "fine", "alright", "upstairs", "upper floor",
    "not sure", "maybe", "neighbour", "neighbor", "next door", "passing by",
    # Sinhala
    "ආරක්ෂිතයි", "ආරක්ෂිතනයි", "ආරක්ෂිත",   # safe
    "උඩ තට්ටු", "උඩ මහළ",                     # upstairs
    "අල්ලපු",                                   # neighbor's / adjacent
    "ළඟ", "ළඟට", "ළඟටම",                       # nearby (not inside)
    "සමහර විට",                                 # maybe / perhaps
    # Tamil
    "பாதுகாப்பாக", "பரவாயில்லை",
]

# Immediacy markers: presence upgrades urgency signal
IMMEDIACY_MARKERS = [
    # English
    "now", "immediately", "hurry", "quickly", "fast", "right now",
    "come now", "send help", "please help",
    # Sinhala
    "දැන්ම", "ඉක්මනට", "ඉක්මනින්", "හදිස්සිය",
    "ඉක්මනකින්", "ඉරවෙලා",
    # Tamil
    "இப்போதே", "சீக்கிரம்",
]

_whisper_model  = None
_sinhala_asr    = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("  Loading Whisper model (base) ...")
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def get_sinhala_asr():
    """Lazy-load Sinhala fine-tune (oracle-language condition for PP2)."""
    global _sinhala_asr
    if _sinhala_asr is None:
        from transformers import pipeline
        print("  Loading Sinhala fine-tune (Lingalingeswaran/whisper-small-sinhala_v3) ...")
        _sinhala_asr = pipeline(
            "automatic-speech-recognition",
            model="Lingalingeswaran/whisper-small-sinhala_v3",
        )
    return _sinhala_asr


# ── Acoustic features ────────────────────────────────────────────────────────

def extract_acoustic(y: np.ndarray, sr: int) -> dict:
    feats = {}

    # MFCCs (40 coefficients — mean + std each = 80 values)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1}_std"]  = float(np.std(mfcc[i]))

    # Pitch (F0) — mean & std of voiced frames
    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
    voiced_f0 = f0[voiced] if voiced is not None and voiced.any() else np.array([0.0])
    feats["pitch_mean"] = float(np.nanmean(voiced_f0))
    feats["pitch_std"]  = float(np.nanstd(voiced_f0))

    # Energy / RMS
    rms = librosa.feature.rms(y=y)[0]
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"]  = float(np.std(rms))

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats["zcr_mean"] = float(np.mean(zcr))

    # Spectral centroid
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats["spec_centroid_mean"] = float(np.mean(sc))

    # Speaking rate — proxy: voiced frames / total duration
    hop = 512
    total_frames = len(y) / hop
    voiced_frames = int(np.sum(voiced)) if voiced is not None else 0
    feats["speaking_rate"] = voiced_frames / max(total_frames, 1)

    # Jitter (mean absolute diff of F0 between consecutive voiced frames)
    if voiced_f0.size > 1:
        feats["jitter"] = float(np.mean(np.abs(np.diff(voiced_f0))))
    else:
        feats["jitter"] = 0.0

    # ── Temporal distress trajectory features ────────────────────────────────
    # Split call into thirds and halves to capture how voice changes over time
    hop = 512
    rms_frames = rms  # already computed above

    n = len(rms_frames)
    if n >= 6:
        t1 = rms_frames[:n//3]
        t3 = rms_frames[2*n//3:]
        feats["rms_slope"] = float(np.mean(t3) - np.mean(t1))

        # Linear regression gradient of RMS over time
        x = np.arange(n, dtype=float)
        feats["rms_gradient"] = float(np.polyfit(x, rms_frames, 1)[0])
    else:
        feats["rms_slope"]    = 0.0
        feats["rms_gradient"] = 0.0

    # Speaking rate change — first half vs second half of audio
    total_frames = len(y) / hop
    if voiced is not None and len(voiced) >= 4:
        mid = len(voiced) // 2
        rate_h1 = float(np.sum(voiced[:mid])) / max(total_frames / 2, 1)
        rate_h2 = float(np.sum(voiced[mid:])) / max(total_frames / 2, 1)
        feats["speaking_rate_change"] = rate_h2 - rate_h1
    else:
        feats["speaking_rate_change"] = 0.0

    return feats


# ── ASR quality score ─────────────────────────────────────────────────────────

def asr_quality_score(text: str) -> float:
    """
    Measures how trustworthy the Whisper transcript is.
    q = 1.0  → Latin or Tamil script  (valid ASR output)
    q ≈ 0.0  → Arabic or Cyrillic     (Sinhala hallucination)

    Only Arabic (U+0600-U+06FF) and Cyrillic (U+0400-U+04FF) are treated
    as hallucination signals. Tamil script (U+0B80-U+0BFF) is valid output
    and must not be penalised.
    """
    cleaned = text.strip()
    if not cleaned:
        return 0.0

    def is_hallucination(ch: str) -> bool:
        code = ord(ch)
        return (0x0600 <= code <= 0x06FF) or (0x0400 <= code <= 0x04FF)

    alpha_chars = [ch for ch in cleaned if ch.isalpha()]
    if not alpha_chars:
        return 0.0

    bad = sum(1 for ch in alpha_chars if is_hallucination(ch))
    return float(1.0 - bad / len(alpha_chars))


# ── Textual features ─────────────────────────────────────────────────────────

def extract_textual(audio: np.ndarray) -> dict:
    from collections import Counter

    model  = get_whisper_model()
    result = model.transcribe(audio.astype(np.float32), task="transcribe")
    raw_text = result.get("text", "")
    text     = raw_text.lower()
    words    = text.split()

    # ASR quality score — how trustworthy is this transcript?
    q = asr_quality_score(raw_text)

    feats = {}
    feats["asr_quality"] = q

    # Raw textual features
    keyword_count     = sum(kw.lower() in text for kw in URGENCY_KEYWORDS)
    word_count        = len(words)
    counts            = Counter(words)
    repeated          = sum(1 for w, c in counts.items() if c > 1)
    repetition_rate   = repeated / max(len(counts), 1)

    positive = {"safe", "okay", "fine", "calm", "alright"}
    negative = {"help", "hurt", "dying", "trapped", "fire", "blood",
                "attack", "crash", "emergency", "pain", "dead"}
    pos = sum(1 for w in words if w in positive)
    neg = sum(1 for w in words if w in negative)
    sentiment_polarity = (pos - neg) / max(len(words), 1)

    # Confidence-weighted textual features
    # When q=1 (good ASR): same as raw. When q=0 (gibberish): zeroed out.
    feats["keyword_count"]      = keyword_count * q
    feats["word_count"]         = word_count    * q
    feats["repetition_rate"]    = repetition_rate   * q
    feats["sentiment_polarity"] = sentiment_polarity * q

    feats["transcript"] = raw_text
    return feats


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    df_labels = pd.read_csv(LABELS)
    rows = []

    for _, row in df_labels.iterrows():
        fname = row["filename"]
        stem  = os.path.splitext(fname)[0]
        wav   = os.path.join(CLEAN_DIR, stem + ".wav")

        if not os.path.exists(wav):
            print(f"  [SKIP — no clean file] {fname}")
            continue

        print(f"  Processing {fname} ...")
        y, sr = librosa.load(wav, sr=TARGET_SR, mono=True)

        acoustic = extract_acoustic(y, sr)
        textual  = extract_textual(y)

        combined = {
            "filename":      fname,
            "urgency_label": row["urgency_label"],
            "language":      row.get("language", ""),
            **acoustic,
            **textual,
        }
        rows.append(combined)

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
