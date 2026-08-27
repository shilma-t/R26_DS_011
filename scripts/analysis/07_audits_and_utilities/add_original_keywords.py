"""
add_original_keywords.py
Adds keyword_count_original to features.csv.

Uses the original keyword list from commit 762041e (before TF-IDF expansion),
with the same asr_quality confidence-weighting as the current pipeline so that
condition B vs C isolates keyword-list expansion only.

Original list: English + Sinhala romanised + 5 Tamil script words (31 terms).
Current list : English + Sinhala script + expanded Tamil (TF-IDF enriched).
"""

import os
import pandas as pd

_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES = os.path.join(_ROOT, "data", "features.csv")

# Original keyword list — exactly as in commit 762041e
ORIGINAL_KEYWORDS = [
    # English
    "help", "emergency", "fire", "accident", "blood", "dying", "dead",
    "trapped", "attack", "knife", "gun", "hurt", "injured", "crash",
    "please", "ambulance", "police", "quickly", "fast", "now",
    # Sinhala romanised (Whisper-base output before fine-tune routing)
    "udaw", "hadisi", "ape", "maru", "wedi", "gini",
    # Tamil script — original 5 terms
    "உதவி", "உதவுங்கள்", "தீ", "விபத்து", "இரத்தம்",
]

def compute_original_keyword_count(transcript: str, asr_quality: float) -> float:
    if not isinstance(transcript, str):
        return 0.0
    text  = transcript.lower()
    count = sum(1 for kw in ORIGINAL_KEYWORDS if kw.lower() in text)
    return count * asr_quality   # same confidence-weighting as current pipeline


def main():
    df = pd.read_csv(FEATURES)

    if "keyword_count_original" in df.columns:
        print("keyword_count_original already present — recomputing.")

    q = df.get("asr_quality", pd.Series(1.0, index=df.index))

    df["keyword_count_original"] = [
        compute_original_keyword_count(t, qi)
        for t, qi in zip(df["transcript"], q)
    ]

    df.to_csv(FEATURES, index=False)

    # Summary
    print(f"Original keyword list : {len(ORIGINAL_KEYWORDS)} terms")
    print(f"Rows processed        : {len(df)}")
    print(f"keyword_count_original stats:")
    print(df["keyword_count_original"].describe().round(4).to_string())
    print()
    print("Comparison — rows where counts differ:")
    diff = df[df["keyword_count_original"] != df["keyword_count"]]
    print(f"  {len(diff)} / {len(df)} rows differ between original and expanded list")
    print()
    print(f"Saved → {FEATURES}")


if __name__ == "__main__":
    main()
