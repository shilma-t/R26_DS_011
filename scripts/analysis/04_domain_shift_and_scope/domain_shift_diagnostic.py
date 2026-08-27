"""
domain_shift_diagnostic.py
Why did v1 score 0.2524 on DMC against 0.6208 on the simulated corpus?

Compares the distribution of every one of the 111 frozen features between the
267 simulated training calls and the 59 DMC recordings, and reports which
features have moved far enough that the model's learned thresholds no longer
apply.

The distinction that matters:
  - if ACOUSTIC features have shifted, the failure is domain shift and better
    ASR will not fix it
  - if only TEXT/CONTEXT features have shifted, the ASR swap should recover
    most of the loss

Standardised mean difference (Cohen's d) is used rather than a p-value: with
n=267 vs n=59 almost everything is "significant", but effect size says how far
apart the distributions actually sit.

Usage:
  python scripts/analysis/domain_shift_diagnostic.py
"""

import os
import sys

import numpy as np
import pandas as pd

_ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
PIPE = os.path.join(_ROOT, "scripts", "pipeline")
if PIPE not in sys.path:
    sys.path.insert(0, PIPE)

FEATURES = os.path.join(_ROOT, "data", "features.csv")
DMC_EVAL = os.path.join(_ROOT, "reports", "pp2", "dmc_evaluation.csv")
REPORTS = os.path.join(_ROOT, "reports", "pp2")

ACOUSTIC = (
    [f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")] +
    ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
     "spec_centroid_mean", "speaking_rate", "jitter",
     "rms_slope", "rms_gradient", "speaking_rate_change"]
)
TEXT = ["asr_quality", "keyword_count", "word_count",
        "repetition_rate", "sentiment_polarity"]
CONTEXT = [
    "sin_context_score", "sin_danger_count", "sin_help_request_count",
    "sin_high_risk_phrase_count", "sin_location_context_count",
    "sin_person_at_risk_count", "sin_safety_state_count",
    "tam_context_score", "tam_danger_count", "tam_help_request_count",
    "tam_high_risk_phrase_count", "tam_location_context_count",
    "tam_person_at_risk_count", "tam_safety_state_count",
]
GROUPS = {"acoustic": ACOUSTIC, "text": TEXT, "context": CONTEXT,
          "fire_smoke": ["fire_smoke_match"]}


def cohens_d(a, b):
    """Standardised mean difference. |d|>0.8 is a large shift."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                     / max(na + nb - 2, 1))
    if pooled == 0:
        return 0.0 if a.mean() == b.mean() else np.inf
    return (b.mean() - a.mean()) / pooled


def coverage(train, test):
    """Fraction of test values falling inside the training min/max range."""
    train, test = np.asarray(train, float), np.asarray(test, float)
    train, test = train[np.isfinite(train)], test[np.isfinite(test)]
    if not len(train) or not len(test):
        return np.nan
    lo, hi = train.min(), train.max()
    return float(((test >= lo) & (test <= hi)).mean())


def main():
    for path in (FEATURES, DMC_EVAL):
        if not os.path.exists(path):
            raise SystemExit(f"Not found: {path}")

    sim = pd.read_csv(FEATURES)
    dmc = pd.read_csv(DMC_EVAL)

    print("=" * 70)
    print("DOMAIN SHIFT DIAGNOSTIC  -  simulated (train) vs DMC (test)")
    print("=" * 70)
    print(f"simulated calls : {len(sim)}")
    print(f"DMC calls       : {len(dmc)}")

    rows = []
    for group, cols in GROUPS.items():
        for col in cols:
            if col not in sim.columns or col not in dmc.columns:
                continue
            d = cohens_d(sim[col], dmc[col])
            rows.append({
                "feature": col,
                "group": group,
                "train_mean": sim[col].mean(),
                "dmc_mean": dmc[col].mean(),
                "abs_d": abs(d) if np.isfinite(d) else np.inf,
                "d": d,
                "in_range": coverage(sim[col], dmc[col]),
            })

    df = pd.DataFrame(rows)

    # ── group summary ────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("SHIFT BY FEATURE GROUP")
    print("-" * 70)
    print(f"{'group':<12} {'n':>4} {'median |d|':>11} {'large(>0.8)':>12} "
          f"{'mean in-range':>14}")
    for group in GROUPS:
        g = df[df["group"] == group]
        if g.empty:
            continue
        large = int((g["abs_d"] > 0.8).sum())
        print(f"{group:<12} {len(g):>4} {g['abs_d'].median():>11.2f} "
              f"{large:>5}/{len(g):<6} {g['in_range'].mean():>13.1%}")

    # ── the headline question ────────────────────────────────────────────────
    ac = df[df["group"] == "acoustic"]
    tx = df[df["group"].isin(["text", "context", "fire_smoke"])]
    ac_large = (ac["abs_d"] > 0.8).mean()
    tx_large = (tx["abs_d"] > 0.8).mean()

    print("\n" + "-" * 70)
    print("VERDICT")
    print("-" * 70)
    print(f"acoustic features with a large shift : {ac_large:.0%} "
          f"({int((ac['abs_d']>0.8).sum())} of {len(ac)})")
    print(f"text/context features with a large shift: {tx_large:.0%} "
          f"({int((tx['abs_d']>0.8).sum())} of {len(tx)})")
    print()
    if ac_large > 0.30:
        print("  ACOUSTIC DOMAIN SHIFT IS SUBSTANTIAL.")
        print("  Better ASR alone will not recover this. The 91 acoustic")
        print("  features - 82% of the model - describe a different signal")
        print("  distribution on real telephony audio than on the simulated")
        print("  corpus the thresholds were learned from.")
    else:
        print("  Acoustic features largely transfer.")
        print("  The failure is concentrated in text/context, which the")
        print("  ASR swap should substantially recover.")

    # ── worst offenders ──────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("20 MOST SHIFTED FEATURES")
    print("-" * 70)
    print(f"{'feature':<24} {'group':<10} {'train':>10} {'dmc':>10} "
          f"{'|d|':>7} {'in-range':>9}")
    worst = df.sort_values("abs_d", ascending=False).head(20)
    for r in worst.itertuples():
        print(f"{r.feature:<24} {r.group:<10} {r.train_mean:>10.3f} "
              f"{r.dmc_mean:>10.3f} {r.abs_d:>7.2f} {r.in_range:>8.0%}")

    # ── duration-sensitive features called out explicitly ────────────────────
    print("\n" + "-" * 70)
    print("DURATION-SENSITIVE AND CONCATENATION-SENSITIVE FEATURES")
    print("-" * 70)
    watch = ["word_count", "keyword_count", "speaking_rate",
             "speaking_rate_change", "rms_slope", "rms_gradient",
             "repetition_rate", "asr_quality"]
    print(f"{'feature':<24} {'train':>10} {'dmc':>10} {'|d|':>7} {'in-range':>9}")
    for name in watch:
        r = df[df["feature"] == name]
        if r.empty:
            continue
        r = r.iloc[0]
        print(f"{name:<24} {r['train_mean']:>10.3f} {r['dmc_mean']:>10.3f} "
              f"{r['abs_d']:>7.2f} {r['in_range']:>8.0%}")

    # ── prediction skew ──────────────────────────────────────────────────────
    if "predicted_label" in dmc.columns:
        print("\n" + "-" * 70)
        print("PREDICTION SKEW")
        print("-" * 70)
        counts = dmc["predicted_label"].value_counts()
        for label in ("High", "Medium", "Low"):
            n = int(counts.get(label, 0))
            print(f"  predicted {label:<7} {n:>3} / {len(dmc)}  "
                  f"({n/len(dmc):.0%})")
        truth = dmc["true_label"].value_counts()
        print("  true distribution: " + ", ".join(
            f"{k} {int(v)}" for k, v in truth.items()))

    out = os.path.join(REPORTS, "domain_shift_diagnostic.csv")
    df.sort_values("abs_d", ascending=False).to_csv(out, index=False)
    print(f"\nFull table -> reports/pp2/domain_shift_diagnostic.csv")


if __name__ == "__main__":
    main()
