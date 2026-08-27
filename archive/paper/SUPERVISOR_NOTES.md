# ICAC 2026 submission — notes for supervisor review

**Paper:** A Multilingual Multi-Channel Emergency Intake and Triage Pipeline for Sri Lanka
**Authors:** Thanseel, Edirisinghe, Karunarathna, Siriwardena
**Draft date:** 25 August 2026

## Status

Draft is complete except one results subsection. Approximately 4,700 words, estimated 5.0 pages before formatting, against a 6-page limit including references.

## Items requiring attention before submission

**1. Section IV-E — DMC results are missing.** The locked external evaluation is still running. The frozen pipeline is being evaluated on 59 authentic DMC recordings under three pre-registered conditions (turn-level caller audio, whole-call caller audio, whole-call mixed audio). Numbers will be inserted on completion. No other section has placeholders.

**2. Reference [11] is unverified.** Cited as a multilingual cross-document event coreference dataset in support of Section VII. The citation could not be confirmed from the source draft and should be supplied by A. N. Siriwardena.

**3. Claim about co-authors' validation.** Section VIII states that only the urgency component has been evaluated on authentic recordings, and that the other three components' figures are development estimates. This is accurate as far as the source drafts show, but it characterises co-authors' work and they should confirm before submission.

**4. Figure 1 is supplied separately** as `fig1_architecture.png` (600 dpi, single-column width). Insertion point is marked in Section III.

## Provenance of reported figures

All numbers were taken from the four source drafts and from frozen experimental records, not re-derived:

| Section | Source |
|---|---|
| IV (urgency) | `candidate_pipeline_freeze.txt`, `ablation_manifest.txt`, diarization worksheets, this project's records |
| V (SMS triage) | Karunarathna, `paper.pdf` |
| VI (dialogue) | Edirisinghe, `Adaptive_Multilingual_Emergency_Clarification_IEEE.docx` |
| VII (correlation) | Siriwardena, `C4_Joint_Incident_History_Matching_PP2_Paper.docx` |

## Points where the draft is deliberately self-critical

These are stated plainly rather than omitted, on the view that a reviewer will find them anyway:

- The 0.621 development macro-F1 is flagged as potentially optimistic, because the fire/smoke feature was selected after inspecting errors on the same corpus used for cross-validation.
- The Medium-recall improvement is attributed to immediacy-feature removal, not to the added fire/smoke feature, which left Medium recall unchanged.
- Automatic speech recognition fails on authentic telephony audio, producing in-script but semantically wrong Sinhala and disabling 20 of 111 features. This is reported as the dominant limitation.
- Speaker diarization failed on four 8 kHz recordings and could not be recovered by resampling.
- Medium remains the weakest class at F1 0.463 throughout.

If the supervisor prefers a less self-critical framing, the substance can be retained while moving these points into the limitations section only.

## Known gaps

- Integration between components is described architecturally but has not yet been implemented or measured end to end. Section VIII presents it as remaining work rather than a result.
- No end-to-end latency figures.
- The Tamil corpus is thin: one authentic Tamil recording in the DMC set.
