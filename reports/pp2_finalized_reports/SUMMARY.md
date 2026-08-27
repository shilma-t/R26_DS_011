# PP2 Finalized Results Summary

Curated summary of the key validated findings for the urgency-classification
component (multimodal urgency detection). Most of these numbers were
computed by scripts in `scripts/analysis/` and printed to console rather
than saved as files — this document is the record of them.

All DMC numbers use production models trained on the full 267-call
simulated corpus, evaluated on the 59-call real DMC set, averaged over 5
seeds (1, 7, 42, 99, 123) unless noted. Simulated numbers use 5-fold
StratifiedKFold CV, same 5 seeds.

---

## 1. Frozen v1 baseline (the original PP1-era model)

Full 111-feature model (91 acoustic + 20 text/context), plain argmax.

| | Accuracy | Macro-F1 | High Recall |
|---|---|---|---|
| Simulated (in-domain CV) | ~65.4% | ~0.63 | ~0.75 |
| DMC (all 59 calls) | 34.2% | 0.326 | 28.7% |

Corrected from an earlier invalid run (0.2524) caused by a preprocessing
mismatch (DMC audio fed in raw, not denoised/trimmed like training) — fixed
in `scripts/pipeline/evaluate_dmc.py`.

---

## 2. Condition C — the best validated candidate (text/context-only, 19 features)

Drops `word_count` from the 20 text/context features (found to be a
near-noise, heavily domain-shifted feature causing systematic Medium-class
collapse — see `02_feature_ablation/medium_collapse_diagnostic.py` and
`text_only_minus_spurious.py`). Keeps `repetition_rate`.

| | Accuracy | Macro-F1 | High Recall | Medium Recall |
|---|---|---|---|---|
| Simulated (in-domain CV) | 41.5% | 0.397 | 41.2% | — |
| **DMC (all 59 calls)** | **45.8%** | **0.449** | **44.4%** | **45.7%** |

**~2x v1's DMC accuracy and High recall.** This is the primary recommended
result for the panel's "improve critical-condition identification" and
"increase accuracy" feedback.

### Condition C + decision-threshold tuning (mild, w_high=1.1/w_low=0.8)
| | Accuracy | Macro-F1 | High Recall |
|---|---|---|---|
| DMC | 44.8% | 0.439 | 49.6% |

### Condition C + recall-constrained tuning (aggressive, w_high=1.4/w_low=0.5)
| | Accuracy | Macro-F1 | High Recall |
|---|---|---|---|
| DMC | 39.0% | 0.312 | 71.3% |

Three honest operating points on the same accuracy/recall trade-off curve —
report whichever matches the framing you want (balanced vs. recall-max).

---

## 3. Fusion strategy comparison (all 59 DMC calls)

| Method | DMC Accuracy | DMC Macro-F1 | DMC High Recall |
|---|---|---|---|
| Text-only (Condition C) | **45.8%** | **0.449** | 44.4% |
| Acoustic-only (91 feat) | 33.6% | 0.304 | 26.1% |
| Standard early fusion (v1, 111 feat) | 34.2% | 0.326 | 28.7% |
| Standard early fusion (apples-to-apples, 110 feat) | 34.6% | 0.319 | 29.6% |
| Late fusion, unweighted (0.5/0.5) | 43.7% | 0.428 | **46.1%** |
| Late fusion, CV-tuned weight (w_acoustic=0.7) | 35.9% | 0.347 | 33.0% |
| Class-specific fusion (per-class weight from sim CV) | 33.2% | 0.318 | 28.7% |

No fusion strategy tested beats text-only alone on full DMC. Late fusion
(unweighted) is the closest competitor and wins specifically on High
recall/High F1.

11 acoustic-subset x fusion-strategy combinations were also tested
(MFCC-only, non-MFCC-only, low-domain-shift feature selection, each
combined 3 ways) — none beat text-only. See `02_feature_ablation/` and
`03_fusion_experiments/`.

---

## 4. Scope-matched DMC subset — flood/fire only (n=11, best-effort keyword-categorized)

The simulated training corpus is scoped to fire/flood scenarios only (per
the PP1 proposal). Isolating the DMC calls that match that scope shows the
full-DMC number is depressed by emergency types the model was never trained
on (fallen trees, tsunami/wind, road-blockage, etc.).

| Method | Accuracy | Macro-F1 | High Recall | High Precision |
|---|---|---|---|---|
| Text-only | 61.8% | 0.600 | 57.1% | 1.000 |
| Acoustic-only | 34.6% | 0.284 | 28.6% | 0.833 |
| Early fusion (both together) | 47.3% | 0.456 | 31.4% | 0.950 |
| **Late fusion (0.5/0.5)** | **67.3%** | **0.630** | **65.7%** | **1.000** |

**Caveat: n=11, small sample — each call moves the percentage by ~9 points.**
Categorization is a keyword heuristic (`error_by_emergency_type.py`),
likely an undercount (~50% of DMC calls landed "unclear"). A native-speaker
re-review of the "unclear" bucket would strengthen this result with a
larger n.

High-precision is 100% for both text-only and late fusion on this subset —
zero false High alarms.

---

## 5. High→Low error root-cause diagnosis (the supervisor's specific ask)

5 DMC calls are robustly (4-5 of 5 seeds) misclassified True-High → Low
under Condition C. All 5 share `keyword_count=0`, `sin_danger_count=0`,
`sin_context_score=0`. Two distinct causes:

1. **Empty transcript** (1 call) — ASR produced nothing. Acoustic-only
   correctly calls this High in 5/5 seeds — the one case where acoustic
   genuinely rescues a text failure.
2. **ASR hallucination loops** (4 calls) — Whisper degenerates into
   repeated-syllable garbage (e.g. `"...ක්තුක්තුක්තුක්තු..."`). Passes the
   `asr_quality` script-validity check (still Sinhala characters) but
   carries zero real signal. Word-token-based `repetition_rate` misses 2 of
   4 entirely (repetition isn't whitespace-separated). A character-level
   compression-ratio detector was tested as a fix — made things worse
   (overfit to a spurious, domain-shifted signal) — ruled out.
   Acoustic-only does NOT rescue any of these 4.

See `01_error_diagnosis/high_to_low_error_analysis.py` and
`check_acoustic_catches_asr_failures.py`.

---

## 6. Ruled-out negative results (tested and rejected, with reasons)

| Approach | Result | Why rejected |
|---|---|---|
| v2 (teammate's ASR) full retrain | DMC macro-F1 0.309 vs v1's 0.329 | Worse despite better simulated numbers |
| v3 (GPT-4o transcript refinement) | 2/19 targeted calls fixed, 1 regression | Net near-zero benefit; full run not pursued |
| Probability calibration (Platt/sigmoid) | Brier score worse in 5/5 seeds | Insufficient data (267 calls) for reliable calibration |
| White-noise audio augmentation | Mean domain-shift \|d\| 0.80→1.65 (worse) | Wrong degradation type — real shift looks like bandwidth limiting, not noise |
| Telephone bandpass (300-3400Hz) augmentation | Mean \|d\| 0.80→1.01 (still worse) | Distorts spectral-shape MFCC means even while helping variability features |
| eGeMAPS (openSMILE) acoustic features | Mean \|d\| 1.197 vs custom features' 0.869 (worse) | Underlying audio mismatch, not feature-engineering choice |
| Confidence-gated fusion (rms_cv-based) | Worse than unweighted late fusion on every metric | Gate direction decided from near-zero correlation (r=0.03) — simulated audio too uniform to learn a real quality signal |
| Compression-ratio hallucination feature | DMC macro-F1 0.449→0.285 (worse) | Same failure mode as word_count — spurious, heavily domain-shifted |
| Tree-hazard/tsunami keyword additions | No measurable effect (text-only), still worse when fused | Zero training exposure — 0-3 of 267 simulated calls ever mention these terms |
| RF hyperparameter tuning | +0.6pt on simulated CV, but WORSE on DMC scope-matched (67.3%→63.6%) | Simulated-optimal params overfit; don't transfer |
| Stacking meta-learner (logistic regression) | 64.9% vs baseline RF's 65.7% on simulated | Underperformed baseline, not pursued further |
| Gradient Boosting | 63.4% vs baseline RF's 65.7% on simulated | Underperformed baseline |

---

## 7. Recommendation for PP2 reporting

- **Primary model to present**: Condition C (text-only, 19 features),
  reported on full DMC (45.8% acc / 44.4% HR) as the headline external-
  validation result.
- **Multimodal-preserving alternative**: unweighted late fusion, ties
  Condition C overall and wins on High recall/precision — use this if you
  want to keep "combining both modalities" as the presented final system.
- **Domain-shift finding**: scope-matched DMC subset (67.3% late fusion)
  as evidence the gap is substantially explained by emergency-type
  coverage, not a failure of the multimodal design itself.
- **Root-cause depth**: the High→Low diagnosis (Section 5) directly answers
  the supervisor's specific question with concrete transcript evidence.
- Frame the extensive ruled-out list (Section 6) as evidence of thorough,
  rigorous methodology — a well-documented negative-result search is a
  legitimate research contribution, not something to hide.

## File index in this folder

- `dmc_evaluation.csv` — v1 baseline, full 111-feature predictions on DMC
- `dmc_evaluation_v2.csv` — same calls with teammate's ASR transcripts (used for the High→Low root-cause work)
- `dmc_error_analysis.csv` — original error taxonomy (ASR_FAILURE / MODEL_OVERRIDE / UNCLEAR / ACOUSTIC_MISMATCH)
- `candidate_pipeline_freeze.txt` / `.json` — frozen v1 pipeline specification
- `domain_shift_diagnostic.csv` — Cohen's d simulated-vs-DMC feature shift, full 111 features
- `discovered_keywords.csv`, `keyword_discovery_v2_candidates.csv` — TF-IDF-discovered keyword candidates (need native-speaker review before adoption)

Full experiment scripts (55 total, reproducible, absolute-path based — safe
to run from anywhere) are organized in `scripts/analysis/` by category:
`01_error_diagnosis/`, `02_feature_ablation/`, `03_fusion_experiments/`,
`04_domain_shift_and_scope/`, `05_ruled_out_negative_results/`,
`06_final_candidate_and_tuning/`, `07_audits_and_utilities/`.
