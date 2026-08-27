"""
safe_refine.py
Wrapper around Component 2's refine_emergency_transcript() that detects when
GPT-4o refuses to correct a draft (returns an apology/refusal instead of a
transcript) and falls back to the raw ASR draft instead of silently baking
the refusal text into downstream features.

Discovered via quick_test_refine_high_misses.py: 3 of 7 tested High-severity
misses got a refusal like "I'm sorry, but the provided draft is too unclear
to accurately correct" returned AS the transcript, which her code accepts
verbatim with no validation. This wrapper is the fix, applied without
touching her file.

Also more efficient than calling transcribe_audio_file(): does ONE ASR pass
(raw) plus one refine call, instead of her function's internal multi-engine
selection running a second time.
"""
import os
import re
import sys

os.environ.setdefault("USE_VERIFIED_REFERENCES", "false")

BACKEND = r"C:\Users\shafr\Desktop\component 2\backend"
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

_REFUSAL_PATTERNS = [
    r"i'?m sorry",
    r"i cannot\b",
    r"i can'?t\b",
    r"unable to (correct|provide|determine)",
    r"too unclear",
    r"as an ai",
    r"cannot provide a corrected transcript",
    r"cannot accurately correct",
    r"insufficient (information|context)",
    r"i apologi[sz]e",
]
_REFUSAL_RE = [re.compile(p, re.I) for p in _REFUSAL_PATTERNS]


def looks_like_refusal(text: str) -> bool:
    """True if GPT-4o declined instead of returning a corrected transcript."""
    if not text or not text.strip():
        return False
    return any(rx.search(text) for rx in _REFUSAL_RE)


def safe_refine(wav_path: str, language_code: str, file_stem: str | None = None) -> dict:
    """
    One ASR pass + refinement, with automatic fallback to the raw draft if
    GPT-4o refuses. Returns:
      text            - refined text, or raw draft if refinement refused/failed
      raw_text        - the unrefined ASR draft, always
      refined         - True if a genuine (non-refusal) refinement was used
      fell_back       - True if refinement was attempted but refused
      refine_engine   - engine string from refine_emergency_transcript()
    """
    from services.whisper_engine import transcribe_with_forced_language
    from services.transcript_refiner import refine_emergency_transcript

    raw = transcribe_with_forced_language(wav_path, language_code)
    raw_text = raw.text or ""

    result = refine_emergency_transcript(
        raw_text,
        language=language_code,
        avg_logprob=raw.avg_logprob,
        file_stem=file_stem,
        candidate_texts=[raw_text],
    )
    refined_text = result.get("text") or raw_text
    engine = result.get("engine")

    if looks_like_refusal(refined_text):
        return {
            "text": raw_text, "raw_text": raw_text,
            "refined": False, "fell_back": True, "refine_engine": engine,
        }

    return {
        "text": refined_text, "raw_text": raw_text,
        "refined": engine == "corpus_guided_refine", "fell_back": False,
        "refine_engine": engine,
    }
