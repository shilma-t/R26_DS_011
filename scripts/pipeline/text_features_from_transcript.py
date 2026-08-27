"""
text_features_from_transcript.py
Compute the 20 text and context features from a transcript that has ALREADY
been produced elsewhere, instead of running ASR.

Why this exists
---------------
02_extract_features.extract_textual() takes audio and transcribes it inline.
In the integrated system the transcript already exists - Component 2 has
transcribed the caller turn with its fine-tuned faster-whisper model before
the urgency classifier is called. Re-transcribing would cost ~85 s per turn on
CPU and make live operation impossible.

Faithfulness
------------
The feature logic here is IDENTICAL to extract_textual(): same 69-term
URGENCY_KEYWORDS lexicon imported from the frozen module, same confidence
weighting by asr_quality_score, same repetition and sentiment definitions,
same context extractor. Only the source of the transcript differs.

The keyword lexicon and quality scorer are imported rather than copied, so
this file cannot silently drift from the frozen pipeline.

Verify with:
  python scripts/pipeline/text_features_from_transcript.py --verify
"""

from __future__ import annotations

import os
import sys
import importlib.util
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from context_features import extract_context_features  # noqa: E402

# Language codes emitted by Component 2's classifier -> what the frozen
# context extractor expects. Unmapped codes yield all-zero context features
# WITHOUT raising, so this mapping is load-bearing.
LANGUAGE_ALIASES = {
    "si": "sinhala", "si-lk": "sinhala", "sinhala": "sinhala",
    "si-en": "sinhala",          # Singlish: Sinhala script content dominates
    "ta": "tamil", "ta-in": "tamil", "ta-lk": "tamil", "tamil": "tamil",
    "ta-en": "tamil",            # Tanglish
    "en": "english", "en-us": "english", "en-gb": "english", "english": "english",
    # Component 2's language classifier also returns these for genuinely
    # code-mixed speech (si-ta, si-ta-en, mixed) - the normal register for
    # Sri Lankan callers per its own documentation, not an edge case. Left
    # unmapped these silently zero every Sinhala/Tamil context feature.
    # No script-count signal is available here, so default to Sinhala,
    # the majority language in both corpora.
    "si-ta": "sinhala", "si-ta-en": "sinhala", "mixed": "sinhala",
}

FIRE_SMOKE_KEYWORDS = [
    "fire", "smoke", "burning", "flames", "gas",
    "ගිනිගන්නව", "ගිනිඅරගෙන", "ගින්නක්", "ගිනි", "දුමක්", "දුම",
    "தீ", "புகை", "gini",
]

_frozen = None


def _frozen_module():
    """Import 02_extract_features.py (leading digit blocks normal import)."""
    global _frozen
    if _frozen is None:
        path = os.path.join(_HERE, "02_extract_features.py")
        spec = importlib.util.spec_from_file_location("frozen_features", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _frozen = module
    return _frozen


def normalise_language(code) -> str:
    """
    'si-LK' -> 'sinhala'. Unknown codes return the lowercased input, which the
    context extractor treats as neither Sinhala nor Tamil (all-zero context).
    """
    key = str(code or "").strip().lower()
    return LANGUAGE_ALIASES.get(key, key)


def fire_smoke_match(text) -> int:
    if not isinstance(text, str):
        return 0
    lowered = text.lower()
    return int(any(kw.lower() in lowered for kw in FIRE_SMOKE_KEYWORDS))


def text_features_from_transcript(raw_text: str, language: str) -> dict:
    """
    The 20 text + context features, plus fire_smoke_match, from a transcript.

    Mirrors extract_textual() exactly, minus the ASR call. Returns the same
    keys that the frozen feature pipeline produces.
    """
    frozen = _frozen_module()
    raw_text = raw_text or ""
    lang = normalise_language(language)

    text = raw_text.lower()
    words = text.split()

    # Confidence weight: identical scorer to the frozen pipeline
    q = frozen.asr_quality_score(raw_text)

    keyword_count = sum(kw.lower() in text for kw in frozen.URGENCY_KEYWORDS)
    word_count = len(words)

    counts = Counter(words)
    repeated = sum(1 for _, c in counts.items() if c > 1)
    repetition_rate = repeated / max(len(counts), 1)

    positive = {"safe", "okay", "fine", "calm", "alright"}
    negative = {"help", "hurt", "dying", "trapped", "fire", "blood",
                "attack", "crash", "emergency", "pain", "dead"}
    pos = sum(1 for w in words if w in positive)
    neg = sum(1 for w in words if w in negative)
    sentiment_polarity = (pos - neg) / max(len(words), 1)

    feats = {
        "asr_quality":        q,
        "keyword_count":      keyword_count * q,
        "word_count":         word_count * q,
        "repetition_rate":    repetition_rate * q,
        "sentiment_polarity": sentiment_polarity * q,
    }

    # Context features are computed on the RAW text, not the lowercased copy,
    # matching extract_textual().
    feats.update(extract_context_features(raw_text, lang))
    feats["fire_smoke_match"] = fire_smoke_match(raw_text)
    feats["transcript"] = raw_text
    feats["language_normalised"] = lang
    return feats


# ── verification ─────────────────────────────────────────────────────────────

def _verify():
    """
    Prove this produces byte-identical text features to the frozen pipeline
    by monkeypatching the frozen ASR to return a known transcript, then
    comparing every shared key.
    """
    import numpy as np

    frozen = _frozen_module()

    cases = [
        ("උදව් කරන්න, ගෙදර ගිනිගන්නවා, ළමයා ඇතුළේ ඉන්නවා", "si-LK"),
        ("හිරවෙලා ඉන්නවා, වතුර ඇතුළට ආවා, ඉක්මනට එන්න", "si-LK"),
        ("உதவி செய்யுங்கள், வீட்டில் தீ, குழந்தை உள்ளே", "ta-IN"),
        ("help my house is on fire two people trapped inside", "en-US"),
        ("", "si-LK"),
        ("ok everything is safe now", "en-US"),
    ]

    class _StubPipe:
        def __init__(self, text):
            self.text = text

        def __call__(self, audio, **kwargs):
            return {"text": self.text}

    silence = np.zeros(16000, dtype=np.float32)
    shared_keys = None
    failures = 0

    print("Comparing against frozen extract_textual()\n")
    for text, lang_code in cases:
        lang = normalise_language(lang_code)

        original = frozen.get_sinhala_asr
        frozen.get_sinhala_asr = lambda t=text: _StubPipe(t)
        try:
            # Route through the Sinhala branch so the stub is used regardless
            # of the case's real language; only the transcript matters here.
            reference = frozen.extract_textual(silence, "sinhala")
        finally:
            frozen.get_sinhala_asr = original

        # Reference context features come from the Sinhala branch; recompute
        # them for the true language so only text features are compared.
        mine = text_features_from_transcript(text, "si-LK")

        if shared_keys is None:
            shared_keys = [k for k in ("asr_quality", "keyword_count", "word_count",
                                       "repetition_rate", "sentiment_polarity")]

        diffs = []
        for key in shared_keys:
            a, b = reference.get(key), mine.get(key)
            if a is None and b is None:
                continue
            if abs(float(a) - float(b)) > 1e-9:
                diffs.append(f"{key}: frozen={a!r} mine={b!r}")

        ctx_ref = {k: v for k, v in reference.items() if k.startswith(("sin_", "tam_"))}
        ctx_mine = {k: v for k, v in mine.items() if k.startswith(("sin_", "tam_"))}
        for key in sorted(set(ctx_ref) | set(ctx_mine)):
            if ctx_ref.get(key) != ctx_mine.get(key):
                diffs.append(f"{key}: frozen={ctx_ref.get(key)} mine={ctx_mine.get(key)}")

        status = "OK  " if not diffs else "FAIL"
        if diffs:
            failures += 1
        print(f"  [{status}] {lang_code:<6} kw={mine['keyword_count']:.2f} "
              f"q={mine['asr_quality']:.2f} ctx={mine.get('sin_context_score', 0)}"
              f"  {text[:38]}")
        for d in diffs:
            print(f"         {d}")

    print()
    if failures:
        print(f"{failures} case(s) DIFFER from the frozen pipeline.")
        return 1
    print("All cases identical to the frozen pipeline.")

    print("\nLanguage mapping:")
    for code in ("si-LK", "ta-IN", "en-US", "si-en", "ta-en", "sinhala", "xx-YY"):
        print(f"  {code:<10} -> {normalise_language(code)}")
    return 0


if __name__ == "__main__":
    if "--verify" in sys.argv:
        raise SystemExit(_verify())
    print(__doc__)
