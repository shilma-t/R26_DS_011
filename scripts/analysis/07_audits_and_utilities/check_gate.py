import os
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
df = pd.read_csv(os.path.join(_ROOT, "data", "features.csv"))

gate_mask = (
    df["transcript"].isna() |
    (df["transcript"].astype(str).str.strip() == "") |
    (df.get("asr_quality",    pd.Series(1.0, index=df.index)) < 0.3) |
    (df.get("repetition_rate", pd.Series(0.0, index=df.index)) > 0.8)
)

ctx_cols  = [c for c in df.columns if c.startswith("sin_") or c.startswith("tam_")]
text_cols = ["keyword_count", "word_count", "repetition_rate", "sentiment_polarity"]
suppress_cols = text_cols + ctx_cols

df_gated = df.copy()
df_gated.loc[gate_mask, suppress_cols] = 0.0

for idx in df[gate_mask].index:
    row = df.loc[idx]
    print("=" * 60)
    print(f"FILE:        {row['filename']}")
    print(f"LABEL:       {row['urgency_label']}")
    print(f"LANGUAGE:    {row['language']}")
    print(f"ASR QUALITY: {row['asr_quality']}")
    print(f"TRANSCRIPT:  {str(row['transcript'])[:200]}")
    print()
    print("  FEATURE                    BEFORE       AFTER")
    for col in suppress_cols:
        before  = df.loc[idx, col]
        after   = df_gated.loc[idx, col]
        changed = " <-- CHANGED" if before != after else ""
        print(f"  {col:<30} {before:>8.4f}   {after:>8.4f}{changed}")
    print()
