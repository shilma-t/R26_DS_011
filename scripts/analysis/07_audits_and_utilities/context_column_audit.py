"""
context_column_audit.py
Audit all 16 context columns (sin_* and tam_*).

For each column:
  - How many calls have a non-zero value
  - Mean value per urgency class (High / Medium / Low)
  - Which languages it fires in
  - Whether it fires in bad-ASR calls (asr_quality < 0.3)
  - Whether it helps at the Medium→High boundary
  - Verdict: useful / weak / exploratory / check
"""

import os
import pandas as pd
import numpy as np

_ROOT    = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
FEATURES = os.path.join(_ROOT, "data", "features.csv")
REPORTS  = os.path.join(_ROOT, "reports", "pp2")

CONTEXT_COLS = [
    "sin_danger_count", "sin_immediacy_count", "sin_help_request_count",
    "sin_high_risk_phrase_count", "sin_location_context_count",
    "sin_safety_state_count", "sin_person_at_risk_count", "sin_context_score",
    "tam_danger_count", "tam_immediacy_count", "tam_help_request_count",
    "tam_high_risk_phrase_count", "tam_location_context_count",
    "tam_safety_state_count", "tam_person_at_risk_count", "tam_context_score",
]

SINHALA_LANGS  = {"sinhala", "singlish"}
TAMIL_LANGS    = {"tamil", "tanglish"}

def verdict(row):
    fires        = row["n_nonzero"]
    max_diff     = row["max_class_diff"]
    wrong_lang   = row["fires_wrong_language"] > 0
    bad_asr      = row["fires_on_bad_asr"] > 0

    if fires <= 2:
        return "exploratory"
    if wrong_lang:
        return "check — fires in wrong language"
    if bad_asr:
        return "check — fires on bad ASR"
    if fires <= 5:
        return "weak — too few calls"
    if max_diff < 0.05:
        return "weak — no class separation"
    return "useful"


def main():
    df = pd.read_csv(FEATURES)
    n  = len(df)

    print("=" * 70)
    print("CONTEXT COLUMN AUDIT")
    print(f"Dataset: {n} calls")
    print("=" * 70)

    rows = []

    for col in CONTEXT_COLS:
        if col not in df.columns:
            print(f"  MISSING: {col}")
            continue

        lang_prefix = "sin" if col.startswith("sin_") else "tam"
        expected_langs = SINHALA_LANGS if lang_prefix == "sin" else TAMIL_LANGS
        wrong_langs    = SINHALA_LANGS if lang_prefix == "tam" else TAMIL_LANGS

        series     = df[col].fillna(0)
        nonzero    = df[series > 0]
        n_nonzero  = len(nonzero)
        pct_nonzero = 100 * n_nonzero / n

        # Mean by urgency class
        mean_high   = df[df["urgency_label"] == "High"][col].fillna(0).mean()
        mean_medium = df[df["urgency_label"] == "Medium"][col].fillna(0).mean()
        mean_low    = df[df["urgency_label"] == "Low"][col].fillna(0).mean()
        max_diff    = max(mean_high, mean_medium, mean_low) - min(mean_high, mean_medium, mean_low)

        # Language breakdown of non-zero calls
        lang_counts = nonzero["language"].value_counts().to_dict() if n_nonzero > 0 else {}

        # Fires in wrong language (e.g. sin_* fires in Tamil calls)
        fires_wrong = sum(v for k, v in lang_counts.items()
                          if str(k).lower() in wrong_langs)

        # Fires on bad ASR (quality < 0.3)
        fires_bad_asr = len(nonzero[nonzero.get("asr_quality", 1.0) < 0.3]) \
                        if "asr_quality" in df.columns else 0

        # Medium→High boundary: does it differ?
        mh_mean = df[df["urgency_label"] == "Medium"][col].fillna(0).mean()
        hi_mean = df[df["urgency_label"] == "High"][col].fillna(0).mean()
        mh_boundary_diff = hi_mean - mh_mean

        row = {
            "column":            col,
            "expected_language": "/".join(sorted(expected_langs)),
            "n_nonzero":         n_nonzero,
            "pct_nonzero":       round(pct_nonzero, 1),
            "mean_High":         round(mean_high, 3),
            "mean_Medium":       round(mean_medium, 3),
            "mean_Low":          round(mean_low, 3),
            "max_class_diff":    round(max_diff, 3),
            "mh_boundary_diff":  round(mh_boundary_diff, 3),
            "lang_breakdown":    str(lang_counts),
            "fires_wrong_language": fires_wrong,
            "fires_on_bad_asr":  fires_bad_asr,
        }
        row["verdict"] = verdict(row)
        rows.append(row)

        # Print
        print(f"\n  {col}")
        print(f"    Non-zero calls : {n_nonzero}/{n} ({pct_nonzero:.1f}%)")
        print(f"    Mean by class  : High={mean_high:.3f}  Medium={mean_medium:.3f}  Low={mean_low:.3f}")
        print(f"    Max class diff : {max_diff:.3f}   Med→High diff: {mh_boundary_diff:+.3f}")
        print(f"    Language breakdown: {lang_counts}")
        print(f"    Fires wrong lang  : {fires_wrong}   Fires bad ASR: {fires_bad_asr}")
        print(f"    VERDICT: {row['verdict']}")

    # Summary
    out = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("VERDICT SUMMARY")
    print("=" * 70)
    print(out.groupby("verdict")["column"].apply(list).to_string())

    print("\n" + "=" * 70)
    print("COLUMNS SORTED BY CLASS SEPARATION (max_class_diff)")
    print("=" * 70)
    print(out[["column","n_nonzero","max_class_diff","mh_boundary_diff","verdict"]]
          .sort_values("max_class_diff", ascending=False)
          .to_string(index=False))

    path = os.path.join(REPORTS, "context_column_audit.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nSaved → reports/pp2/context_column_audit.csv")


if __name__ == "__main__":
    main()
