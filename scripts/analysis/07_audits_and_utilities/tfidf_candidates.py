"""
10_tfidf_candidates.py
Step 3 — TF-IDF candidate term discovery.

Reads audit_transcripts.xlsx (Sinhala fine-tune, Usable + Partial only)
and features.csv (Tamil Whisper-base transcripts).

For each language, contrasts:
  High vs Low        (clear signal)
  High vs Medium     (hard boundary — most important)
  Medium vs Low

Outputs:
  reports/tfidf_candidates_sinhala.csv
  reports/tfidf_candidates_tamil.csv

Each row: term | freq_high | freq_medium | freq_low | best_contrast | association
Use this output to DISCOVER candidates — do not use it as a word list directly.
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIN_CSV    = os.path.join(_ROOT, "reports", "pp2", "sinhala_transcripts.csv")
FEATURES   = os.path.join(_ROOT, "data", "features.csv")
OUT_DIR    = os.path.join(_ROOT, "reports", "pp2")


# ── helpers ──────────────────────────────────────────────────────────────────

def tokenise(text: str) -> list[str]:
    """Split on whitespace only — preserves non-Latin script tokens."""
    if not isinstance(text, str):
        return []
    return [t for t in text.strip().split() if len(t) >= 2]


def doc_frequencies(docs: list[str]) -> Counter:
    """How many documents contain each token."""
    counts = Counter()
    for doc in docs:
        for tok in set(tokenise(doc)):
            counts[tok] += 1
    return counts


def run_tfidf(texts_high, texts_medium, texts_low, top_n=60):
    """
    Run TF-IDF on three class groups combined, then rank by class association.
    Returns a DataFrame of candidate terms.
    """
    labels = (["High"]   * len(texts_high) +
              ["Medium"] * len(texts_medium) +
              ["Low"]    * len(texts_low))
    texts  = texts_high + texts_medium + texts_low

    if len(texts) < 6:
        print("  Not enough documents — skipping.")
        return pd.DataFrame()

    # Use character-level analyzer fallback won't work for script — use word
    vec = TfidfVectorizer(
        analyzer="word",
        tokenizer=tokenise,
        token_pattern=None,
        min_df=2,        # must appear in at least 2 documents
        max_df=0.95,
        sublinear_tf=True,
    )

    try:
        tfidf = vec.fit_transform(texts)
    except ValueError as e:
        print(f"  TF-IDF error: {e}")
        return pd.DataFrame()

    terms    = vec.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf.toarray(), columns=terms)
    tfidf_df["label"] = labels

    # Mean TF-IDF score per class
    mean_h = tfidf_df[tfidf_df["label"] == "High"].drop(columns="label").mean()
    mean_m = tfidf_df[tfidf_df["label"] == "Medium"].drop(columns="label").mean()
    mean_l = tfidf_df[tfidf_df["label"] == "Low"].drop(columns="label").mean()

    # Raw document frequencies per class
    df_h = doc_frequencies(texts_high)
    df_m = doc_frequencies(texts_medium)
    df_l = doc_frequencies(texts_low)

    rows = []
    for term in terms:
        h  = float(mean_h.get(term, 0))
        m  = float(mean_m.get(term, 0))
        l  = float(mean_l.get(term, 0))
        best_contrast = max(h - l, h - m, m - l, key=abs)
        if h > m and h > l:
            assoc = "High"
        elif m > h and m > l:
            assoc = "Medium"
        elif l > h and l > m:
            assoc = "Low"
        else:
            assoc = "Mixed"
        rows.append({
            "term":          term,
            "tfidf_high":    round(h, 4),
            "tfidf_medium":  round(m, 4),
            "tfidf_low":     round(l, 4),
            "freq_high":     df_h.get(term, 0),
            "freq_medium":   df_m.get(term, 0),
            "freq_low":      df_l.get(term, 0),
            "best_contrast": round(best_contrast, 4),
            "association":   assoc,
        })

    result = (pd.DataFrame(rows)
              .sort_values("best_contrast", ascending=False)
              .head(top_n)
              .reset_index(drop=True))
    return result


# ── Sinhala ───────────────────────────────────────────────────────────────────

def run_sinhala():
    print("=== SINHALA ===")
    df = pd.read_csv(SIN_CSV, encoding="utf-8-sig")

    print(f"  Total Sinhala calls: {len(df)}")
    print(f"  Class distribution:\n{df['urgency_label'].value_counts().to_string()}")

    h = df[df["urgency_label"] == "High"]["transcript"].tolist()
    m = df[df["urgency_label"] == "Medium"]["transcript"].tolist()
    l = df[df["urgency_label"] == "Low"]["transcript"].tolist()

    print(f"  High={len(h)}  Medium={len(m)}  Low={len(l)}")

    result = run_tfidf(h, m, l)
    if result.empty:
        return

    out = os.path.join(OUT_DIR, "tfidf_candidates_sinhala.csv")
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  Top candidates saved to {out}")
    print(result[["term","freq_high","freq_medium","freq_low","association"]].to_string(index=False))


# ── Tamil ─────────────────────────────────────────────────────────────────────

def run_tamil():
    print("\n=== TAMIL ===")
    feat = pd.read_csv(FEATURES)
    tamil = feat[feat["language"] == "tamil"].dropna(subset=["transcript"]).copy()

    # Exclude clearly empty or garbage (quality score < 0.3)
    if "asr_quality" in tamil.columns:
        before = len(tamil)
        tamil  = tamil[tamil["asr_quality"] >= 0.3]
        print(f"  Dropped {before - len(tamil)} low-quality Tamil transcripts")

    # Also drop call55, call57, call40 flagged in audit
    flagged = {"call55.ogg", "call57.ogg", "call40.ogg"}
    tamil   = tamil[~tamil["filename"].isin(flagged)]

    print(f"  Tamil calls used: {len(tamil)}")
    print(f"  Class distribution:\n{tamil['urgency_label'].value_counts().to_string()}")

    h = tamil[tamil["urgency_label"] == "High"]["transcript"].tolist()
    m = tamil[tamil["urgency_label"] == "Medium"]["transcript"].tolist()
    l = tamil[tamil["urgency_label"] == "Low"]["transcript"].tolist()

    print(f"  High={len(h)}  Medium={len(m)}  Low={len(l)}")

    result = run_tfidf(h, m, l)
    if result.empty:
        return

    out = os.path.join(OUT_DIR, "tfidf_candidates_tamil.csv")
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  Top candidates saved to {out}")
    print(result[["term","freq_high","freq_medium","freq_low","association"]].to_string(index=False))


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    run_sinhala()
    run_tamil()
    print("\nDone. Review both CSVs, then move to Step 4 (build word lists).")
