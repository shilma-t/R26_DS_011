# Emergency Call Urgency Detection System
### R26_DS_011 — PP2

Multimodal urgency classification of Sri Lankan emergency calls (Sinhala / Tamil / Singlish / Tanglish / English) into **High / Medium / Low** urgency using acoustic and linguistic/context features, validated against real Disaster Management Centre (DMC) recordings and integrated live into the AI call-handling and operator-dashboard components.

---

## Folder Structure

```
R26_DS_011/
├── audio/
│   ├── *.ogg                  ← raw labelled call recordings
│   └── clean/                 ← preprocessed .wav files (auto-generated)
├── data/
│   ├── labels.csv              ← labelled samples (simulated corpus)
│   └── features.csv            ← extracted feature matrix (auto-generated)
├── frozen_v1/                 ← frozen snapshot of the v1 baseline's data/pipeline
├── models/                    ← saved model artefacts (auto-generated)
│   ├── urgency_rf_v1.joblib          ← early-fusion baseline (frozen v1)
│   ├── urgency_text_only_v1.joblib   ← text/context-only branch
│   └── urgency_acoustic_only_v1.joblib ← acoustic-only branch
├── reports/                   ← evaluation plots and CSVs
├── archive/reports/pp2_finalized_reports/SUMMARY.md ← curated PP2 results record
├── integration/                     ← the locked model wired into the live system
│   └── urgency_bridge.py            ← classify_urgency_late_fusion(), used by Component 2
├── templates/index.html       ← web demo UI (upload file or record live)
├── app.py                     ← Flask web interface — same locked pipeline as production
├── requirements.txt
├── archive/                    ← superseded scripts/results, kept for reference (not deleted)
│   ├── scripts/                     ← retired pipeline scripts (05_predict.py onward — superseded
│   │                                   by scripts/analysis/) and the one-time DMC m4a→wav converter
│   ├── reports/                     ← superseded result sets (e.g. the old flood25 DMC subset,
│   │                                   pre-diarization-fix manifests)
│   └── paper/                       ← ICAC2026 draft + supervisor notes
└── scripts/
    ├── pipeline/                    ← the active core pipeline
    │   ├── 01_preprocess.py             ← noise reduction, resample, trim
    │   ├── 02_extract_features.py       ← acoustic + ASR textual features
    │   ├── 03_train.py                  ← Random Forest + SMOTE training
    │   ├── 04_evaluate.py               ← confusion matrix, F1, feature importance
    │   ├── context_features.py          ← imported live by integration/urgency_bridge.py
    │   ├── text_features_from_transcript.py ← imported live by integration/urgency_bridge.py
    │   ├── caller_turns.py              ← imported by evaluate_dmc.py
    │   └── evaluate_dmc.py / _v2 / _v3  ← DMC evaluation, produce the CSVs SUMMARY.md cites
    └── analysis/               ← 55 reproducible PP2 experiment scripts
        ├── 01_error_diagnosis/
        ├── 02_feature_ablation/
        ├── 03_fusion_experiments/
        ├── 04_domain_shift_and_scope/
        ├── 05_ruled_out_negative_results/
        ├── 06_final_candidate_and_tuning/
        └── 07_audits_and_utilities/
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running the Full Pipeline

Run the core pipeline scripts from `scripts/pipeline/`:

```bash
cd scripts/pipeline

# Step 1 — Preprocess audio
python 01_preprocess.py

# Step 2 — Extract features
python 02_extract_features.py

# Step 3 — Train classifier
python 03_train.py

# Step 4 — Evaluate and generate plots
python 04_evaluate.py
```

The rest of `scripts/pipeline/` is the DMC-specific one-time prep (diarization,
caller-audio extraction, verification worksheets) — already run; kept for
reproducibility, not something you re-run for a normal pipeline pass.

## PP2 Experiment Record

```bash
cd scripts/analysis
```

`scripts/analysis/` holds the full PP2 experiment record — all 55 reproducible
scripts behind every tuning attempt, negative result, and diagnostic that led to the
locked model below (error diagnosis, feature ablation, fusion experiments,
domain-shift/scope analysis, ruled-out negative results, final candidate tuning, and
audit utilities). See `archive/reports/pp2_finalized_reports/SUMMARY.md` for the
curated write-up. Older single-file scripts (SHAP analysis, per-language breakdown, keyword
discovery, model comparison) have been superseded by this analysis suite and are
kept in `archive/scripts/` for reference.

## Web Demo

```bash
python app.py
# Open http://localhost:5000
```

Upload an audio file, or record live from the mic, to get an urgency classification
with confidence scores and transcript — using the exact same locked model and ASR
pipeline as the live integrated system (byte-identical, verified side-by-side).

---

## System Architecture

Late fusion of two independently-trained models — not one model on concatenated
features. Sinhala is transcribed with the multilingual ASR the live call-handling
integration already uses (Component 2), not this project's own fine-tuned Sinhala
model; Tamil and English use base Whisper.

```
Raw call audio
      │
      ▼
  Preprocessing
  (noise reduction → 16kHz resample → silence trim)
      │
      ├─────────────────────────────┬─────────────────────────────┐
      │                             │                             │
  Acoustic branch               Linguistic/context branch         │
  91 features                   19-20 features                    │
  MFCCs, pitch/F0, RMS energy   ASR transcript (Sinhala: multi-    │
  ZCR, spectral centroid,       lingual ASR · Tamil/English:       │
  speaking rate, jitter         base Whisper)                      │
      │                         keyword/context score, repetition  │
      │                         rate, ASR quality                  │
      ▼                             ▼                             │
  Acoustic-only RF              Text/context-only RF               │
  (StandardScaler + SMOTE)      (StandardScaler + SMOTE)           │
      │                             │                             │
      └──────────────┬──────────────┘                             │
                      ▼                                            │
        Late fusion: 0.5 × P(acoustic) + 0.5 × P(text)              │
                      │                                            │
                      ▼                                            │
          High / Medium / Low ───────────────────────────────────┘
```

---

## Results — locked model (2026-08-30)

Chosen over the project's own-ASR configuration because it improves on **all three**
evaluation views, not just one — the first tuning candidate in the project to clear
that bar on simulated data too, not only on DMC.

| Evaluation view | Accuracy | Macro F1 |
|---|---|---|
| Simulated (5-fold CV, 267 calls) | 66.4% | 0.639 |
| DMC — scope-matched (fire/flood only, n=11) | 80.0% | 0.753 |
| DMC — full real corpus (n=59) | 44.1% | 0.425 |

High-urgency recall on the full DMC set — the operationally critical metric, since a
missed High call is the dangerous failure direction — is 60.9%, the strongest result
among every configuration evaluated. See `archive/paper/urgency_detection_solo_paper_v3.md`
Table 2 for the full results, including 95% bootstrap confidence intervals.

For reference, the same architecture using this project's own fine-tuned Sinhala ASR
instead scored 63.9% / 0.613, 67.3% / 0.630, and 40.7% / 0.398 respectively on the
same three views — worse on every one.

**Key finding:** the external-validation gap between simulated (66.4%) and full DMC
(44.1%) is substantially explained by emergency-type coverage, not a failure of the
multimodal design — the scope-matched subset (fire/flood only, the scenarios the
simulated corpus was actually built around) recovers to 80.0%. ASR quality on real
telephony audio remains the dominant remaining limitation; see
`archive/reports/pp2_finalized_reports/SUMMARY.md` §5 for a transcript-level root-cause
breakdown of the specific failure modes.

Late fusion beat 7+ other fusion strategies tested (early fusion, CV-tuned weights,
per-class weights, confidence gating) and 11+ other tuning approaches were tried and
rejected with documented reasons (hyperparameter tuning, probability calibration,
audio augmentation, stacking, gradient boosting, and more) — see SUMMARY.md §6 for
the full negative-result record.

---

## Live Integration

This model is not a standalone demo — it runs inside the live 4-component system:

- **Component 2** (AI call agent) calls `integration/urgency_bridge.py` directly on
  each confirmed caller turn, aggregating turn-level results into a call-level
  verdict when the call ends (including disconnected/abandoned calls, via an
  idle-timeout safety net).
- **Component 4** (operator dashboard) receives the classified report and displays
  it with urgency badges, alongside SMS reports from Component 3, with cross-source
  incident matching.

## PP2 → Next Steps

| Remaining limitation | Plan |
|---|---|
| ASR quality on real telephony audio (dominant DMC gap) | Domain-fine-tuned ASR substitution across the pipeline, re-validate on DMC |
| DMC scope coverage (non-fire/flood emergency types) | Expand simulated training corpus beyond fire/flood scenarios |
| Concurrent multi-caller load | Not yet supported — needs measurement before real deployment |
| "Unclear" DMC categorization (~50% of scope-matching) | Native-speaker re-review to strengthen the scope-matched n=11 result |
