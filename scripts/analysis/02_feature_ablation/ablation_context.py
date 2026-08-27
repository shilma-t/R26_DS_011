"""
ablation_context.py  —  v2
5-condition fold-locked ablation for the context-aware keyword algorithm.

BASE  = acoustic + asr_quality + expanded text features
         (matches the 96-column pre-context features.csv)

Conditions (explicit set additions on top of acoustic):
  A — Acoustic only            (no text at all)
  B — Acoustic + asr_quality + keyword_count   (original text baseline)
  C — BASE = A + asr_quality + all 4 text cols (pre-context 96-col baseline)
  D — BASE + 16 context cols   (sin_* / tam_*)
  E — D with ASR gate applied  (text+context zeroed for unusable transcripts)

All conditions share identical 5-fold assignments.
Every run saves a manifest (SHA-256 + full feature lists) and fold_assignments.csv.
Cite only results tied to the manifest hash.
"""

import hashlib
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES = os.path.join(_ROOT, "data", "features.csv")
REPORTS  = os.path.join(_ROOT, "reports", "pp2")

# ── Experiment constants ──────────────────────────────────────────────────────

FOLD_SEED   = 42
N_FOLDS     = 5
RF_PARAMS   = dict(n_estimators=200, min_samples_leaf=2,
                   class_weight="balanced", random_state=FOLD_SEED, n_jobs=-1)
SMOTE_SEED  = FOLD_SEED

ASR_GATE_QUALITY    = 0.3
ASR_GATE_REPETITION = 0.8

LABEL_COL    = "urgency_label"
EXCLUDE_COLS = {"filename", "urgency_label", "language", "transcript"}


# ── Column group builder ──────────────────────────────────────────────────────

def build_groups(df_cols):
    """
    Derive named column groups from the actual CSV columns.

    asr_quality is included only if present in the CSV, which confirms it was
    produced by the pre-context extraction pipeline (02_extract_features.py).
    It is computed from transcript character-set alone — no label information.
    """
    acoustic = (
        [f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")] +
        ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
         "spec_centroid_mean", "speaking_rate", "jitter",
         "rms_slope", "rms_gradient", "speaking_rate_change"]
    )
    acoustic = [c for c in acoustic if c in df_cols]

    asr_quality   = ["asr_quality"] if "asr_quality" in df_cols else []

    # keyword_count_original: original 31-term list (commit 762041e), same asr_quality weighting
    # keyword_count          : expanded TF-IDF list — what is currently in features.csv
    orig_kw       = ["keyword_count_original"] if "keyword_count_original" in df_cols else []
    old_text      = orig_kw + [c for c in ["word_count", "repetition_rate",
                                            "sentiment_polarity"] if c in df_cols]
    expanded_text = [c for c in ["keyword_count", "word_count",
                                  "repetition_rate", "sentiment_polarity"]
                     if c in df_cols]
    context       = sorted(c for c in df_cols
                           if c.startswith("sin_") or c.startswith("tam_"))

    return dict(acoustic=acoustic, asr_quality=asr_quality,
                old_text=old_text, expanded_text=expanded_text,
                context=context)


def build_conditions(g):
    """
    BASE = acoustic + asr_quality + expanded_text  (pre-context 96-col model).
    Each condition is an explicit union; no implicit inheritance.
    """
    A = g["acoustic"]
    B = g["acoustic"] + g["asr_quality"] + g["old_text"]
    C = g["acoustic"] + g["asr_quality"] + g["expanded_text"]   # BASE
    D = C + g["context"]
    E = D   # same column list as D; gate is applied to the data frame, not columns
    return {"A_acoustic": A, "B_old_keyword": B, "C_base": C,
            "D_context": D, "E_context_gate": E}


# ── Hashing ───────────────────────────────────────────────────────────────────

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Validation ────────────────────────────────────────────────────────────────

def validate(df, conditions):
    errors = []
    for name, cols in conditions.items():
        if LABEL_COL in cols:
            errors.append(f"{name}: urgency_label in feature list")
        overlap = set(cols) & EXCLUDE_COLS
        if overlap:
            errors.append(f"{name}: excluded metadata in feature list: {overlap}")
        missing = [c for c in cols if c not in df.columns]
        if missing:
            errors.append(f"{name}: columns absent from CSV: {missing}")
        from collections import Counter
        dupes = [c for c, n in Counter(cols).items() if n > 1]
        if dupes:
            errors.append(f"{name}: duplicate columns: {dupes}")

    # Confirm all conditions share the same row ordering (same df used)
    # — guaranteed because we slice X from the same df with the same index.

    return errors


# ── ASR gate ──────────────────────────────────────────────────────────────────

def apply_gate(df, g):
    df = df.copy()
    suppress = [c for c in (g["asr_quality"] + g["expanded_text"] + g["context"])
                if c in df.columns]
    mask = (
        df["transcript"].isna() |
        (df["transcript"].astype(str).str.strip() == "") |
        (df.get("asr_quality",     pd.Series(1.0, index=df.index)) < ASR_GATE_QUALITY) |
        (df.get("repetition_rate", pd.Series(0.0, index=df.index)) > ASR_GATE_REPETITION)
    )
    df.loc[mask, suppress] = 0.0
    return df, int(mask.sum())


# ── Pipeline ──────────────────────────────────────────────────────────────────

def make_pipeline():
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=SMOTE_SEED)),
        ("clf",    RandomForestClassifier(**RF_PARAMS)),
    ])


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_condition(name, X, y, le, skf):
    fold_records = []
    all_true, all_pred = [], []

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        pipe = make_pipeline()
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        all_true.extend(y[te])
        all_pred.extend(pred)
        mf1 = f1_score(y[te], pred, average="macro")
        for cls_idx, cls_name in enumerate(le.classes_):
            fold_records.append({
                "condition": name, "fold": fold, "class": cls_name,
                "f1":       f1_score(y[te], pred, labels=[cls_idx], average="macro"),
                "recall":   recall_score(y[te], pred, labels=[cls_idx], average="macro"),
                "macro_f1": mf1,
            })

    return fold_records, np.array(all_true), np.array(all_pred)


# ── Manifest ──────────────────────────────────────────────────────────────────

def build_manifest(df, conditions, g, n_suppressed):
    lines = []
    w = lines.append

    w("=" * 70)
    w("ABLATION MANIFEST")
    w(f"  Run timestamp  : {datetime.now().isoformat(timespec='seconds')}")
    w(f"  Script SHA-256 : {file_sha256(os.path.abspath(__file__))}")
    w("")
    w("── Feature matrix ──")
    w(f"  Path    : {FEATURES}")
    w(f"  Shape   : {df.shape[0]} rows × {df.shape[1]} columns")
    w(f"  SHA-256 : {file_sha256(FEATURES)}")
    w("")
    w("── Excluded columns (metadata — never in X) ──")
    for c in sorted(EXCLUDE_COLS):
        w(f"  {c}")
    w("")
    w("── Column groups ──")
    for gname, cols in g.items():
        w(f"  {gname} ({len(cols)} cols)")
        for c in cols:
            w(f"    {c}")
    w("")
    w("── Conditions (exact feature names) ──")
    for cname, cols in conditions.items():
        w(f"  {cname}  →  {len(cols)} features")
        for c in cols:
            w(f"    {c}")
    w("")
    w("── Cross-validation ──")
    w(f"  StratifiedKFold  n_splits={N_FOLDS}  shuffle=True  random_state={FOLD_SEED}")
    w("")
    w("── SMOTE ──")
    w(f"  Method     : SMOTE  random_state={SMOTE_SEED}")
    w("  Placement  : inside each fold, fit on train split only, after StandardScaler")
    w("")
    w("── Random Forest ──")
    for k, v in RF_PARAMS.items():
        w(f"  {k}: {v}")
    w("")
    w("── Scaler ──")
    w("  StandardScaler — fit on train split only; applied before SMOTE inside pipeline")
    w("")
    w("── ASR gate (Condition E only) ──")
    w(f"  asr_quality     < {ASR_GATE_QUALITY}  → zero all text + context features")
    w(f"  repetition_rate > {ASR_GATE_REPETITION} → zero all text + context features")
    w(f"  Calls suppressed: {n_suppressed}")
    w("")
    w("── ASR models ──")
    w("  Sinhala : Lingalingeswaran/whisper-small-sinhala_v3  (oracle-language routing)")
    w("  Other   : openai/whisper-base")
    w("")
    w("── asr_quality provenance ──")
    w("  Computed by asr_quality_score() in 02_extract_features.py")
    w("  Input: transcript text only. Logic: fraction of alpha chars that are")
    w("  NOT Arabic (U+0600-U+06FF) or Cyrillic (U+0400-U+04FF).")
    w("  No label information used. Present in original 96-col pre-context CSV.")
    w("  Included in B, C, D, E to match that baseline; excluded from A (acoustic-only).")
    w("")
    w("── Keyword versions ──")
    w("  keyword_count_original : 31-term list (commit 762041e)")
    w("    English + Sinhala romanised + 5 Tamil script words")
    w("    Applied with same asr_quality confidence-weighting as current pipeline")
    w("    Used in condition B only (pre-context baseline)")
    w("  keyword_count          : expanded TF-IDF list (~70 terms)")
    w("    English + Sinhala script + expanded Tamil (TF-IDF enriched)")
    w("    Used in conditions C, D, E")
    w("  Computed by: scripts/add_original_keywords.py")
    w("")
    w("── Context feature rule version ──")
    w("  File   : scripts/context_features.py")
    w("  8 features per language (sin_* / tam_*), 16 total")
    w("  Word lists: TF-IDF candidates (all Sinhala/Tamil calls) + manual audit")
    w("  context_score = min(danger,1)+min(immediacy,1)+min(help,1)")
    w("                  +min(phrase,1)+min(person,1) - min(safety,1)  [max=5]")
    w("")
    w("── Validation checks (all must pass before run proceeds) ──")
    w("  urgency_label not in any feature list              : PASS")
    w("  metadata cols not in any feature list              : PASS")
    w("  all feature cols present in features.csv           : PASS")
    w("  no duplicate cols within any condition             : PASS")
    w("  all conditions use identical row ordering          : PASS (same df)")
    w("  all conditions use identical fold indices          : PASS (same SKF object)")
    w("  SMOTE fit on train split only                      : PASS (ImbPipeline)")
    w("  scaler fit on train split only                     : PASS (ImbPipeline)")
    w("=" * 70)

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(REPORTS, exist_ok=True)

    df_raw = pd.read_csv(FEATURES)
    g      = build_groups(df_raw.columns)
    conds  = build_conditions(g)

    # Validate before doing anything
    errors = validate(df_raw, conds)
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    print("Validation: all checks passed.\n")

    df_gated, n_suppressed = apply_gate(df_raw, g)

    le  = LabelEncoder()
    y   = le.fit_transform(df_raw[LABEL_COL].values)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)

    # Save fold assignments (proves identical folds across all conditions)
    fold_rows = []
    for fold_id, (_, te) in enumerate(skf.split(df_raw, y), 1):
        for i in te:
            fold_rows.append({
                "filename":      df_raw.iloc[i]["filename"],
                "fold_id":       fold_id,
                "urgency_label": df_raw.iloc[i][LABEL_COL],
                "language":      df_raw.iloc[i].get("language", ""),
            })
    (pd.DataFrame(fold_rows)
       .sort_values(["fold_id", "filename"])
       .to_csv(os.path.join(REPORTS, "fold_assignments.csv"), index=False))
    print("Fold assignments → reports/fold_assignments.csv")

    # Print + save manifest
    manifest = build_manifest(df_raw, conds, g, n_suppressed)
    print(manifest)
    with open(os.path.join(REPORTS, "ablation_manifest.txt"), "w",
              encoding="utf-8") as f:
        f.write(manifest)
    print("Manifest → reports/ablation_manifest.txt\n")

    # Run conditions
    data_for = {
        "A_acoustic":     df_raw,
        "B_old_keyword":  df_raw,
        "C_base":         df_raw,
        "D_context":      df_raw,
        "E_context_gate": df_gated,
    }

    all_fold_records = []
    summary_rows     = []

    for cname, cols in conds.items():
        df_use = data_for[cname]
        present = [c for c in cols if c in df_use.columns]
        n_feat  = len(present)
        print(f"── Condition {cname}  ({n_feat} features)")
        X = df_use[present].fillna(0).values

        fold_records, y_true, y_pred = evaluate_condition(cname, X, y, le, skf)
        all_fold_records.extend(fold_records)

        macro_f1s = [r["macro_f1"] for r in fold_records
                     if r["class"] == le.classes_[0]]
        mean_mf1  = float(np.mean(macro_f1s))
        std_mf1   = float(np.std(macro_f1s))

        report  = classification_report(y_true, y_pred,
                                        target_names=le.classes_, output_dict=True)
        cm      = confusion_matrix(y_true, y_pred, labels=le.transform(le.classes_))
        cls_lst = list(le.classes_)
        med_idx = cls_lst.index("Medium")
        hi_idx  = cls_lst.index("High")

        summary_rows.append({
            "condition":        cname,
            "n_features":       n_feat,
            "macro_f1_mean":    round(mean_mf1, 4),
            "macro_f1_std":     round(std_mf1,  4),
            "medium_recall":    round(report["Medium"]["recall"],   4),
            "medium_f1":        round(report["Medium"]["f1-score"], 4),
            "high_recall":      round(report["High"]["recall"],     4),
            "high_f1":          round(report["High"]["f1-score"],   4),
            "low_recall":       round(report["Low"]["recall"],      4),
            "low_f1":           round(report["Low"]["f1-score"],    4),
            "medium_as_high":   int(cm[med_idx, hi_idx]),
            "calls_suppressed": n_suppressed if "gate" in cname else 0,
        })

        print(f"  Macro F1 : {mean_mf1:.3f} ± {std_mf1:.3f}")
        print(f"  Medium   recall={report['Medium']['recall']:.3f}"
              f"  f1={report['Medium']['f1-score']:.3f}")
        print(f"  High     recall={report['High']['recall']:.3f}"
              f"  f1={report['High']['f1-score']:.3f}")
        print(f"  Low      recall={report['Low']['recall']:.3f}"
              f"  f1={report['Low']['f1-score']:.3f}")
        print(f"  Medium→High errors: {int(cm[med_idx, hi_idx])}")
        if "gate" in cname:
            print(f"  Calls suppressed by gate: {n_suppressed}")
        print()

    summary_df = pd.DataFrame(summary_rows)
    folds_df   = pd.DataFrame(all_fold_records)
    summary_df.to_csv(os.path.join(REPORTS, "ablation_context_summary.csv"), index=False)
    folds_df.to_csv(  os.path.join(REPORTS, "ablation_context_folds.csv"),   index=False)

    print("=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print()
    print("Outputs:")
    print("  reports/ablation_manifest.txt")
    print("  reports/fold_assignments.csv")
    print("  reports/ablation_context_summary.csv")
    print("  reports/ablation_context_folds.csv")


if __name__ == "__main__":
    main()
