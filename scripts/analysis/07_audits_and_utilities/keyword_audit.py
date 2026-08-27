"""
step4_keyword_audit.py
Keyword match audit by language — compares B (original 31-term list)
vs C (expanded TF-IDF list) to test whether the expanded list adds
signal or noise broken down by language.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES = os.path.join(_ROOT, "data", "features.csv")
REPORTS  = os.path.join(_ROOT, "reports", "pp2")

# Original 31-term list (commit 762041e) — used in condition B
ORIGINAL_KEYWORDS = [
    "help", "emergency", "fire", "accident", "blood", "dying", "dead",
    "trapped", "attack", "knife", "gun", "hurt", "injured", "crash",
    "please", "ambulance", "police", "quickly", "fast", "now",
    "udaw", "hadisi", "ape", "maru", "wedi", "gini",
    "உதவி", "உதவுங்கள்", "தீ", "விபத்து", "இரத்தம்",
]

# Expanded list (condition C/D/E) — from 02_extract_features.py
EXPANDED_KEYWORDS = [
    "help", "fire", "inside", "trapped", "quickly", "emergency",
    "blood", "dying", "dead", "accident", "attack", "hurt", "injured",
    "crash", "ambulance", "police", "gas", "smoke", "burning", "flood",
    "workers", "grandmother", "children", "child", "baby", "unconscious",
    "breathing", "cannot breathe", "stuck", "still inside",
    "water", "flames", "fast", "save",
    "උදව්", "උදව් කරන්න", "බේරගන්න", "හදිස්සිය",
    "ගිනිගන්නව", "ගිනිඅරගෙන", "ගින්නක්", "ගිනි",
    "දුමක්", "දුම", "වතුර ඇතුළට", "ගෙදර ඇතුළට",
    "හිරවෙලා", "සිරවෙලා", "යටවෙලා", "ගංවතුරට",
    "ළමයා", "ළමයිය", "ළමේ", "හුස්ම ගන්නත් අමාරු", "හුස්ම",
    "ඉක්මනට", "ඉක්මනින්", "කරුණාකර", "දැන්",
    "உதவி", "உதவி செய்யுங்கள்", "உதவுங்கள்",
    "வெள்ளம்", "தீ", "இரத்தம்", "விபத்து",
    "தண்ணீர்", "குழந்தை", "புகை",
]


def match_keywords(text, keywords):
    if not isinstance(text, str):
        return []
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lower]


def audit(df, keywords, label):
    rows = []
    for _, row in df.iterrows():
        matches = match_keywords(row.get("transcript", ""), keywords)
        asr_q   = row.get("asr_quality", 1.0)
        rows.append({
            "filename":    row["filename"],
            "language":    row.get("language", ""),
            "urgency":     row["urgency_label"],
            "asr_quality": round(asr_q, 4),
            "n_matches":   len(matches),
            "has_match":   len(matches) > 0,
            "matches":     ", ".join(matches[:5]),
            "transcript_snippet": str(row.get("transcript",""))[:80],
            "condition":   label,
        })
    return pd.DataFrame(rows)


def print_summary(df_b, df_c, lang_filter=None):
    label = lang_filter if lang_filter else "ALL"
    if lang_filter:
        df_b = df_b[df_b["language"] == lang_filter]
        df_c = df_c[df_c["language"] == lang_filter]

    if len(df_b) == 0:
        return

    n        = len(df_b)
    b_match  = df_b["has_match"].sum()
    c_match  = df_c["has_match"].sum()
    b_hits   = df_b["n_matches"].sum()
    c_hits   = df_c["n_matches"].sum()

    # Bad match = match but asr_quality < 0.3 (hallucinated transcript)
    b_bad = df_b[df_b["has_match"] & (df_b["asr_quality"] < 0.3)]["has_match"].sum()
    c_bad = df_c[df_c["has_match"] & (df_c["asr_quality"] < 0.3)]["has_match"].sum()

    print(f"\n  [{label}]  n={n}")
    print(f"  {'':30s}  {'B (original)':>14}  {'C (expanded)':>14}")
    print(f"  {'calls with ≥1 match':30s}  {b_match:>10} ({100*b_match/n:.0f}%)  "
          f"{c_match:>10} ({100*c_match/n:.0f}%)")
    print(f"  {'total keyword hits':30s}  {b_hits:>14}  {c_hits:>14}")
    print(f"  {'matches on bad ASR (q<0.3)':30s}  {b_bad:>14}  {c_bad:>14}")


def main():
    df = pd.read_csv(FEATURES)
    df_b = audit(df, ORIGINAL_KEYWORDS, "B")
    df_c = audit(df, EXPANDED_KEYWORDS, "C")

    print("=" * 65)
    print("KEYWORD MATCH AUDIT — B (original) vs C (expanded)")
    print("=" * 65)

    languages = ["ALL"] + sorted(df["language"].dropna().unique().tolist())
    for lang in languages:
        print_summary(df_b, df_c, None if lang == "ALL" else lang)

    # Per-urgency breakdown
    print("\n" + "=" * 65)
    print("BY URGENCY LABEL (all languages)")
    print("=" * 65)
    for urgency in ["High", "Medium", "Low"]:
        b_u = df_b[df_b["urgency"] == urgency]
        c_u = df_c[df_c["urgency"] == urgency]
        n   = len(b_u)
        print(f"\n  [{urgency}]  n={n}")
        print(f"  B matches: {b_u['has_match'].sum()}  "
              f"C matches: {c_u['has_match'].sum()}")

    # Show sample matches on bad ASR — are they genuine?
    print("\n" + "=" * 65)
    print("SAMPLE MATCHES ON BAD ASR (asr_quality < 0.3)")
    print("Expanded list (C) matches on hallucinated transcripts:")
    print("=" * 65)
    bad_c = df_c[(df_c["has_match"]) & (df_c["asr_quality"] < 0.3)]
    if len(bad_c) == 0:
        print("  None — no expanded-list matches on bad transcripts.")
    else:
        for _, row in bad_c.iterrows():
            print(f"\n  {row['filename']}  [{row['urgency']}]  "
                  f"lang={row['language']}  q={row['asr_quality']}")
            print(f"  Matches  : {row['matches']}")
            print(f"  Transcript: {row['transcript_snippet']}")

    print("\n" + "=" * 65)
    print("Original list (B) matches on hallucinated transcripts:")
    print("=" * 65)
    bad_b = df_b[(df_b["has_match"]) & (df_b["asr_quality"] < 0.3)]
    if len(bad_b) == 0:
        print("  None.")
    else:
        for _, row in bad_b.iterrows():
            print(f"\n  {row['filename']}  [{row['urgency']}]  "
                  f"lang={row['language']}  q={row['asr_quality']}")
            print(f"  Matches  : {row['matches']}")
            print(f"  Transcript: {row['transcript_snippet']}")

    # Save detailed table
    out = pd.concat([df_b, df_c], ignore_index=True)
    path = os.path.join(REPORTS, "keyword_audit.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nDetailed table saved → reports/keyword_audit.csv")


if __name__ == "__main__":
    main()
