"""
dmc_error_analysis.py
Per-call failure categorization for the v1 DMC evaluation, covering all
three classes but with High given the deepest treatment (panel priority:
"identifying critical condition").

For every misclassified call, checks four candidate failure modes using
signals already present in dmc_evaluation.csv, and picks the one the
evidence most supports:

  ASR_FAILURE        - text pipeline found nothing usable (no keywords, no
                        context, and/or the transcript itself looks broken:
                        very short, low asr_quality, or a repetition loop)
  ACOUSTIC_MISMATCH  - text signal WAS present and pointed toward the true
                        class, but this call's acoustic features sit closer
                        to the training distribution of the class it got
                        predicted as, instead of its true class
  MODEL_OVERRIDE     - text signal was present and pointed correctly, and
                        acoustics didn't clearly point elsewhere either -
                        the model got it wrong despite adequate signal
  UNCLEAR            - no single dominant explanation from available signals

Acoustic comparison uses the 267-call simulated training distribution
(mean/std per class) for a curated set of interpretable distress features:
pitch_mean, pitch_std, rms_mean, rms_std, jitter, speaking_rate.

Usage:
  python scripts/analysis/dmc_error_analysis.py
"""
import os
import numpy as np
import pandas as pd

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
DMC_EVAL = os.path.join(ROOT, "reports", "pp2", "dmc_evaluation.csv")
TRAIN_FEATURES = os.path.join(ROOT, "data", "features.csv")
OUT = os.path.join(ROOT, "reports", "pp2", "dmc_error_analysis.csv")

ACOUSTIC_PROBE = ["pitch_mean", "pitch_std", "rms_mean", "rms_std",
                  "jitter", "speaking_rate"]


def build_class_stats(train_df):
    stats = {}
    for cls in ("High", "Medium", "Low"):
        sub = train_df[train_df["urgency_label"] == cls]
        stats[cls] = {
            col: (sub[col].mean(), sub[col].std() or 1e-6)
            for col in ACOUSTIC_PROBE
        }
    return stats


def acoustic_distance(row, cls, stats):
    """Mean absolute z-score across the probe features vs a class's training dist."""
    zs = []
    for col in ACOUSTIC_PROBE:
        mean, std = stats[cls][col]
        zs.append(abs((row[col] - mean) / std))
    return np.mean(zs)


def text_signal_present(row):
    return (row.get("keyword_count", 0) or 0) > 0 or any(
        (row.get(c, 0) or 0) > 0 for c in
        ["sin_danger_count", "sin_help_request_count", "sin_high_risk_phrase_count",
         "sin_person_at_risk_count", "tam_danger_count", "tam_help_request_count",
         "tam_high_risk_phrase_count", "tam_person_at_risk_count"]
    )


def transcript_looks_broken(row):
    reasons = []
    wc = row.get("word_count", 0) or 0
    aq = row.get("asr_quality", 1) or 1
    rep = row.get("repetition_rate", 0) or 0
    if wc < 3:
        reasons.append(f"very short (word_count={wc:.1f})")
    if aq < 0.5:
        reasons.append(f"low asr_quality ({aq:.2f})")
    if rep > 0.3:
        reasons.append(f"repetitive/degenerate (repetition_rate={rep:.2f})")
    return reasons


def categorize(row, stats):
    true_cls, pred_cls = row["true_label"], row["predicted_label"]
    if true_cls == pred_cls:
        return "CORRECT", ""

    broken = transcript_looks_broken(row)
    has_text_signal = text_signal_present(row)

    if not has_text_signal or broken:
        detail = "; ".join(broken) if broken else "no keywords or context features fired"
        return "ASR_FAILURE", detail

    d_true = acoustic_distance(row, true_cls, stats)
    d_pred = acoustic_distance(row, pred_cls, stats)
    if d_pred < d_true - 0.3:
        return "ACOUSTIC_MISMATCH", (
            f"acoustics closer to {pred_cls} training dist (z={d_pred:.2f}) "
            f"than {true_cls} (z={d_true:.2f})"
        )

    if d_true < d_pred:
        return "MODEL_OVERRIDE", (
            f"text signal present, acoustics also closer to true class "
            f"(z={d_true:.2f} vs {d_pred:.2f} for predicted) - model still got it wrong"
        )

    return "UNCLEAR", f"z_true={d_true:.2f} z_pred={d_pred:.2f}, no dominant signal"


def main():
    dmc = pd.read_csv(DMC_EVAL)
    train = pd.read_csv(TRAIN_FEATURES)
    stats = build_class_stats(train)

    results = []
    for _, row in dmc.iterrows():
        cat, detail = categorize(row, stats)
        results.append({
            "recording_id": row.get("recording_id", row.get("filename", "")),
            "true_label": row["true_label"],
            "predicted_label": row["predicted_label"],
            "category": cat,
            "detail": detail,
            "keyword_count": row.get("keyword_count", 0),
            "asr_quality": row.get("asr_quality", None),
            "word_count": row.get("word_count", None),
            "repetition_rate": row.get("repetition_rate", None),
            "transcript": str(row.get("transcript", ""))[:80],
        })

    out = pd.DataFrame(results)
    out.to_csv(OUT, index=False)

    print("=" * 78)
    print("DMC ERROR ANALYSIS (v1, primary model)")
    print("=" * 78)

    errors = out[out["category"] != "CORRECT"]
    print(f"\nTotal calls: {len(out)}  |  Correct: {(out['category']=='CORRECT').sum()}  "
          f"|  Misclassified: {len(errors)}")

    print("\n--- Failure category counts, by TRUE class ---")
    pivot = errors.groupby(["true_label", "category"]).size().unstack(fill_value=0)
    print(pivot.to_string())

    print("\n--- Failure category counts, overall ---")
    print(errors["category"].value_counts().to_string())

    for cls, label in [("High", "HIGH (panel priority - full detail)"),
                       ("Medium", "MEDIUM"), ("Low", "LOW")]:
        sub = errors[errors["true_label"] == cls]
        print(f"\n{'='*78}\n{label} misses: {len(sub)}\n{'='*78}")
        for _, r in sub.iterrows():
            print(f"  [{r['category']:<18}] {r['recording_id']:<28} "
                  f"-> predicted {r['predicted_label']}")
            print(f"      {r['detail']}")
            if cls == "High":
                print(f"      kw={r['keyword_count']}  asrq={r['asr_quality']}  "
                      f"wc={r['word_count']}  transcript: {r['transcript']}")

    print(f"\nSaved -> reports/pp2/dmc_error_analysis.csv")


if __name__ == "__main__":
    main()
