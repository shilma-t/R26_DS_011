"""
freeze_candidate_pipeline.py
Generate the final candidate pipeline manifest.

This freezes the DEVELOPMENT-SELECTED candidate feature set.
The 0.621 macro-F1 is a development estimate, not a validated final result.
External evaluation on DMC recordings is required before any final claim.
"""

import hashlib
import json
import os
import datetime
import pandas as pd

_ROOT    = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
REPORTS  = os.path.join(_ROOT, "reports", "pp2")

def sha256(path):
    if not os.path.exists(path):
        return "FILE NOT FOUND"
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

def main():
    now = datetime.datetime.now().isoformat(timespec="seconds")

    # ── File hashes ───────────────────────────────────────────────────────────
    labels_path   = os.path.join(_ROOT, "data", "labels_frozen_20260817.csv")
    features_path = os.path.join(_ROOT, "data", "features.csv")
    folds_path    = os.path.join(REPORTS, "fold_assignments.csv")

    # ── Exact feature list ────────────────────────────────────────────────────
    acoustic = (
        [f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")] +
        ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
         "spec_centroid_mean", "speaking_rate", "jitter",
         "rms_slope", "rms_gradient", "speaking_rate_change"]
    )
    text = ["asr_quality", "keyword_count", "word_count",
            "repetition_rate", "sentiment_polarity"]
    context_retained = [
        "sin_context_score", "sin_danger_count", "sin_help_request_count",
        "sin_high_risk_phrase_count",
        "sin_location_context_count", "sin_person_at_risk_count",
        "sin_safety_state_count",
        "tam_context_score", "tam_danger_count", "tam_help_request_count",
        "tam_high_risk_phrase_count",
        "tam_location_context_count", "tam_person_at_risk_count",
        "tam_safety_state_count",
    ]
    context_removed = ["sin_immediacy_count", "tam_immediacy_count"]
    fire_smoke_feature = ["fire_smoke_match"]

    final_features = acoustic + text + context_retained + fire_smoke_feature
    n_features = len(final_features)

    # ── Development results ───────────────────────────────────────────────────
    dev_results = {
        "A_acoustic_91":           {"macro_f1": 0.5919, "High_recall": 0.7615, "Medium_recall": 0.4390, "Low_recall": 0.6000},
        "B_old_keyword_96":        {"macro_f1": 0.6081, "High_recall": 0.7538, "Medium_recall": 0.4390, "Low_recall": 0.6545},
        "C_base_96":               {"macro_f1": 0.5934, "High_recall": 0.7615, "Medium_recall": 0.4390, "Low_recall": 0.6000},
        "D_context_112":           {"macro_f1": 0.6039, "High_recall": 0.7462, "Medium_recall": 0.4390, "Low_recall": 0.6545},
        "E_context_gate_112":      {"macro_f1": 0.6047, "High_recall": 0.7385, "Medium_recall": 0.4512, "Low_recall": 0.6545},
        "D_no_immediacy_110":      {"macro_f1": 0.6091, "High_recall": 0.7538, "Medium_recall": 0.4634, "Low_recall": 0.6364,
                                    "note": "Condition D minus sin_immediacy_count and tam_immediacy_count"},
        "D_candidate_111":         {"macro_f1": 0.6208, "High_recall": 0.7923, "Medium_recall": 0.4634, "Low_recall": 0.6364,
                                    "note": "D_no_immediacy + fire_smoke_match. DEVELOPMENT ESTIMATE — may be optimistic."},
    }

    # ── Fire/smoke feature definition ─────────────────────────────────────────
    fire_smoke_keywords = [
        "fire", "smoke", "burning", "flames", "gas",
        "ගිනිගන්නව", "ගිනිඅරගෙන", "ගින්නක්", "ගිනි", "දුමක්", "දුම",
        "தீ", "புகை",
        "gini",
    ]

    # ── Build manifest ────────────────────────────────────────────────────────
    manifest = {
        "title": "CANDIDATE PIPELINE FREEZE MANIFEST",
        "status": "DEVELOPMENT CANDIDATE — not a validated final result",
        "frozen_at": now,
        "methodological_caveat": (
            "fire_smoke_match was selected after inspecting development errors "
            "on the same simulated corpus used for cross-validation. "
            "The macro-F1 of 0.621 is a development estimate and may be optimistic. "
            "External evaluation on independent DMC recordings is required "
            "before any final performance claim."
        ),

        "data": {
            "labels_file":         "data/labels_frozen_20260817.csv",
            "labels_sha256":       sha256(labels_path),
            "labels_frozen_date":  "2026-08-17",
            "features_file":       "data/features.csv",
            "features_sha256":     sha256(features_path),
            "fold_assignments":    "reports/pp2/fold_assignments.csv",
            "fold_assignments_sha256": sha256(folds_path),
            "n_calls":             267,
            "class_distribution":  {"High": 130, "Medium": 82, "Low": 55},
            "languages":           ["sinhala", "tamil", "singlish", "tanglish", "english"],
        },

        "asr": {
            "sinhala_model":  "Lingalingeswaran/whisper-small-sinhala_v3",
            "other_model":    "openai/whisper-base",
            "routing":        "oracle-language label used to route each call",
            "asr_quality_metric": (
                "Fraction of alpha characters that are not Arabic (U+0600-U+06FF) "
                "or Cyrillic (U+0400-U+04FF). Computed from transcript text only."
            ),
        },

        "feature_engineering": {
            "n_features_final":    n_features,
            "acoustic_features":   acoustic,
            "text_features":       text,
            "context_features_retained": context_retained,
            "context_features_removed": {
                "columns":  context_removed,
                "reason":   (
                    "Immediacy features (sin_immediacy_count, tam_immediacy_count) "
                    "were found to fire more in Medium and Low calls than High, "
                    "actively harming Medium recall. Removing them improved "
                    "macro-F1 from 0.604 to 0.609 and Medium recall from 0.439 to 0.463."
                ),
            },
            "fire_smoke_match": {
                "definition":  "Binary: 1 if any fire/smoke keyword matches transcript (case-insensitive substring), else 0",
                "keywords":    fire_smoke_keywords,
                "languages_covered": ["english", "sinhala_script", "tamil_script", "romanised_sinhala"],
                "fires_in_n_calls": 29,
                "fires_by_class":   {"High": 20, "Medium": 7, "Low": 2},
                "observed_high_precision": 0.69,
                "selection_basis": (
                    "Selected after keyword concept audit showed FIRE/SMOKE "
                    "has the strongest High-class association of any concept group. "
                    "DEVELOPMENT SELECTION — not pre-registered."
                ),
            },
            "keyword_count": {
                "list": "expanded TF-IDF enriched list (~70 terms)",
                "note": "TF-IDF candidates selected from full corpus before folds — exploratory, not fold-safe",
            },
            "context_feature_version": "scripts/pipeline/context_features.py",
            "context_score_formula": (
                "min(danger,1)+min(immediacy,1)+min(help,1)"
                "+min(phrase,1)+min(person,1)-min(safety,1)  [max=5]"
                " — NOTE: immediacy component still in score formula; "
                "only the raw count column was removed from the feature matrix."
            ),
        },

        "ml_pipeline": {
            "cross_validation": "StratifiedKFold  n_splits=5  shuffle=True  random_state=42",
            "scaler":           "StandardScaler — fit on train split only",
            "smote":            "SMOTE(random_state=42) — fit on train split only, after scaler, inside ImbPipeline",
            "classifier":       "RandomForestClassifier",
            "rf_params": {
                "n_estimators":    200,
                "min_samples_leaf": 2,
                "class_weight":    "balanced",
                "random_state":    42,
                "n_jobs":          -1,
            },
        },

        "development_results": dev_results,

        "comparator_note": (
            "Medium recall comparisons in the paper must state the comparator explicitly. "
            "D_candidate vs D_full: Medium recall 0.463 vs 0.439 (+0.024). "
            "D_candidate vs D_no_imm: Medium recall 0.463 vs 0.463 (no change — "
            "the fire_smoke feature did NOT improve Medium recall). "
            "The Medium recall improvement is attributable to immediacy removal, not fire_smoke."
        ),

        "frozen_outputs": {
            "ablation_summary":       "reports/pp2/ablation_context_summary.csv",
            "ablation_folds":         "reports/pp2/ablation_context_folds.csv",
            "ablation_manifest":      "reports/pp2/ablation_manifest.txt",
            "fold_assignments":       "reports/pp2/fold_assignments.csv",
            "medium_error_analysis":  "reports/pp2/medium_error_analysis.csv",
            "context_column_audit":   "reports/pp2/context_column_audit.csv",
            "keyword_concept_audit":  "reports/pp2/keyword_concept_audit.csv",
            "acoustic_group_ablation":"reports/pp2/acoustic_group_ablation.csv",
            "immediacy_removal_test": "reports/pp2/immediacy_removal_test.csv",
            "fire_smoke_test":        "reports/pp2/fire_smoke_test.csv",
        },

        "next_steps": [
            "1. Evaluate frozen candidate pipeline on DMC recordings WITHOUT any tuning",
            "2. Report simulated and DMC results in separate tables",
            "3. Record whether DMC recordings contain enough genuine Medium cases for three-class evaluation",
            "4. Do NOT add more features, keywords, or classifier changes before DMC evaluation",
            "5. Begin writing Methods and Results sections using frozen development results",
        ],
    }

    # ── Save as JSON ──────────────────────────────────────────────────────────
    json_path = os.path.join(REPORTS, "candidate_pipeline_freeze.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ── Save readable text version ────────────────────────────────────────────
    txt_path = os.path.join(REPORTS, "candidate_pipeline_freeze.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("CANDIDATE PIPELINE FREEZE MANIFEST\n")
        f.write(f"Frozen: {now}\n")
        f.write("STATUS: DEVELOPMENT CANDIDATE — external DMC evaluation required\n")
        f.write("=" * 70 + "\n\n")

        f.write("METHODOLOGICAL CAVEAT\n")
        f.write(manifest["methodological_caveat"] + "\n\n")

        f.write("DATA\n")
        f.write(f"  Labels (frozen): {manifest['data']['labels_file']}\n")
        f.write(f"  Labels SHA-256 : {manifest['data']['labels_sha256']}\n")
        f.write(f"  Features       : {manifest['data']['features_file']}\n")
        f.write(f"  Features SHA   : {manifest['data']['features_sha256']}\n")
        f.write(f"  Calls          : {manifest['data']['n_calls']}\n")
        f.write(f"  Classes        : {manifest['data']['class_distribution']}\n\n")

        f.write("ASR\n")
        f.write(f"  Sinhala : {manifest['asr']['sinhala_model']}\n")
        f.write(f"  Other   : {manifest['asr']['other_model']}\n")
        f.write(f"  Routing : {manifest['asr']['routing']}\n\n")

        f.write(f"FINAL FEATURE SET ({n_features} features)\n")
        f.write(f"  Acoustic ({len(acoustic)}): mfcc_1-40_mean/std + 11 prosodic/energy/spectral\n")
        f.write(f"  Text ({len(text)}): {text}\n")
        f.write(f"  Context retained ({len(context_retained)}): {context_retained}\n")
        f.write(f"  Context REMOVED : {context_removed}\n")
        f.write(f"  Added           : fire_smoke_match (binary)\n\n")

        f.write("FIRE/SMOKE KEYWORDS\n")
        for kw in fire_smoke_keywords:
            f.write(f"  {kw}\n")
        f.write("\n")

        f.write("ML PIPELINE\n")
        f.write(f"  {manifest['ml_pipeline']['cross_validation']}\n")
        f.write(f"  {manifest['ml_pipeline']['scaler']}\n")
        f.write(f"  {manifest['ml_pipeline']['smote']}\n")
        f.write(f"  RF params: {manifest['ml_pipeline']['rf_params']}\n\n")

        f.write("DEVELOPMENT RESULTS\n")
        for cond, res in dev_results.items():
            f.write(f"  {cond:<30} macro_f1={res['macro_f1']:.4f}  "
                    f"High={res['High_recall']:.4f}  "
                    f"Med={res['Medium_recall']:.4f}  "
                    f"Low={res['Low_recall']:.4f}\n")

        f.write("\nCOMPARATOR NOTE\n")
        f.write(manifest["comparator_note"] + "\n\n")

        f.write("NEXT STEPS\n")
        for step in manifest["next_steps"]:
            f.write(f"  {step}\n")

    print("=" * 70)
    print("CANDIDATE PIPELINE FROZEN")
    print("=" * 70)
    print(f"  Frozen at      : {now}")
    print(f"  Features       : {n_features}")
    print(f"  Labels SHA-256 : {manifest['data']['labels_sha256'][:16]}...")
    print(f"  Features SHA   : {manifest['data']['features_sha256'][:16]}...")
    print()
    print("  Development results:")
    for cond, res in dev_results.items():
        note = f"  ← {res['note']}" if "note" in res else ""
        print(f"    {cond:<30} macro_f1={res['macro_f1']:.4f}{note}")
    print()
    print(f"  Saved → reports/pp2/candidate_pipeline_freeze.json")
    print(f"  Saved → reports/pp2/candidate_pipeline_freeze.txt")
    print()
    print("  STATUS: DEVELOPMENT CANDIDATE")
    print("  Next: evaluate on DMC recordings without any tuning.")


if __name__ == "__main__":
    main()
