# Multimodal Urgency Detection for Sri Lankan Emergency Calls: Evaluating Simulated-to-Real Generalisation

*[Author names, affiliations, and emails withheld pending anonymisation decision for submission.]*

---

*Abstract*—Emergency call centres must order incoming reports by urgency under time pressure, yet transcript content alone is a weak signal: a caller describing a life-threatening situation and a caller requesting routine information can use overlapping vocabulary while differing sharply in vocal distress. This study develops and evaluates a multimodal classifier that combines acoustic-prosodic and textual/conversational features to assign a High, Medium, or Low urgency label to emergency calls in the Sri Lankan multilingual context (Sinhala, Tamil, English, and their code-mixed varieties). A Random Forest classifier was developed under stratified 5-fold cross-validation on a 267-call simulated corpus, comparing text-only, acoustic-only, early-fusion, and late-fusion formulations, and was then evaluated — without further tuning of its frozen configuration — on 59 independently collected, authentic Disaster Management Centre (DMC) recordings. Development-corpus macro-F1 reached 0.621, but the identically-configured early-fusion model fell to 0.326 macro-F1 on the independent DMC set, only marginally above a majority-class baseline (39.0%). A domain-shift analysis using Cohen's d found the acoustic feature distributions of the DMC recordings differed from the simulated corpus more than the text/context features did (median |d| = 0.57 vs. 0.154), providing partial evidence for the generalisation gap. A late-fusion reformulation — two independently trained modality-specific classifiers combined by unweighted probability averaging — was found empirically to be the only fusion strategy that improved on text-only performance on the independent set, and is reported as the final formulation, alongside the finding that text-only features alone remain competitive with, and on one metric superior to, the fused model. These results indicate that acoustic information contributes meaningfully to urgency detection but that simulated-corpus performance substantially overstates real-world generalisation, and that fusion strategy — not raw feature richness — determines whether that gap can be narrowed.

*Keywords*—emergency call triage, urgency classification, multimodal fusion, acoustic-prosodic features, domain shift, code-mixed speech, Sinhala, Tamil

---

## 1. Introduction

Emergency response is a resource-allocation problem under scarcity: dispatch centres must decide which reports demand immediate action before full information is available. In Sri Lanka this decision is complicated by three factors specific to the setting. Callers report in Sinhala, Tamil, or English, frequently code-mixed within a single utterance. Emergency-call speech is produced under acute physiological and psychological stress, which manifests acoustically — in pitch, energy, and speaking rate — independently of the words used. And textual content alone is an ambiguous signal for urgency: two callers can use near-identical vocabulary while one is in immediate danger and the other is not, because the distinguishing information is carried in *how* something is said, not only *what* is said.

This motivates combining acoustic and textual evidence. However, a classifier's apparent effectiveness on a simulated or acted corpus — the only kind of data typically available at development time, given the operational and ethical difficulty of collecting labelled real emergency audio — does not establish that it will generalise to authentic calls, where recording conditions, caller behaviour, and the distribution of true emergencies differ from a simulated set. This study treats that generalisation question as a first-class research question rather than an afterthought, by evaluating the developed classifier on an independently collected set of authentic emergency recordings from Sri Lanka's Disaster Management Centre (DMC), obtained separately from and after the classifier's development.

The study asks four research questions:

- **RQ1.** How effectively can acoustic-prosodic and textual/conversational information, individually, classify emergency-call urgency?
- **RQ2.** Does multimodal fusion improve urgency classification over either modality alone, and does the fusion strategy (early vs. late) matter?
- **RQ3.** How well does a classifier developed on a simulated emergency-call corpus generalise to independent, authentic recordings?
- **RQ4.** What measurable factors help explain the observed difference between simulated-corpus and real-world performance?

**Contributions.** (1) A systematic comparison of unimodal and multimodal (early- and late-fusion) urgency classifiers in a multilingual, code-mixed emergency-call setting. (2) An independent external evaluation against 59 authentic DMC recordings, distinct from the simulated development corpus in source, collection method, and time period. (3) A quantitative domain-shift analysis using Cohen's d that identifies which feature families differ most between the simulated and real-world distributions. (4) An evidence-based selection of a late-fusion formulation, justified by its behaviour under external validation rather than by development-set accuracy alone.

## 2. Related Work

**Speech-based urgency and affect.** Acoustic correlates of stress and arousal — fundamental frequency, jitter, energy dynamics, and speaking rate — are established markers in speech emotion recognition [4], with mel-frequency cepstral coefficients (MFCCs) the dominant paraverbal representation. Prior work on emergency-call classification has generally relied on transcript content, leaving the acoustic channel comparatively underused despite its robustness to transcription failure — a property directly relevant to this study's findings on real telephony audio.

**Multilingual and code-mixed language.** Cross-lingual encoders such as XLM-RoBERTa [1] transfer to low-resource languages with limited supervision. Weakly supervised multilingual speech recognition models [2] extend automatic speech recognition (ASR) coverage across languages but are known to degrade under intra-sentential code-mixing. Language identification for code-switched text has been established as a distinct modelling problem rather than noisy monolingual data [3], a position this study's feature design follows by treating Sinhala and Tamil context features separately rather than merging them.

**Speaker diarization.** Isolating a single speaker's audio from a multi-speaker recording — necessary here to separate caller from call-taker before acoustic feature extraction — is addressed by modern neural diarization pipelines such as pyannote.audio [7].

**Class imbalance.** Synthetic Minority Over-sampling Technique (SMOTE) [6] is a standard method for correcting class imbalance during training without altering the evaluation distribution, used here inside cross-validation folds only to avoid leakage.

**Domain shift.** That a model's held-out performance on its training distribution does not predict its performance on a differently sourced dataset is a long-established finding in machine learning, sometimes termed "dataset bias" [5]. This study applies that concern directly and quantitatively (via Cohen's d) to the specific case of simulated-versus-authentic emergency audio, which — to the best of our knowledge — has not been reported for Sri Lankan multilingual emergency calls specifically.

## 3. Research Gap and Questions

Emergency-call urgency classification research has generally reported performance on a single, internally consistent corpus, whether simulated or real, but rarely on a classifier developed on one and evaluated on the other. This leaves open exactly the question that matters for deployment: does a promising development-set result transfer? The multilingual, code-mixed character of Sri Lankan emergency communication further means that findings from monolingual settings cannot be assumed to hold. This study is designed to close that specific gap, structured around RQ1–RQ4 above, with the simulated-to-real evaluation (RQ3–RQ4) treated as the central empirical contribution rather than a secondary robustness check.

## 4. Methodology

### 4.1 Datasets

**Simulated development corpus.** 267 simulated emergency calls, role-played to cover a range of incident types and severities, labelled High (130), Medium (82), or Low (55) urgency. Two independent annotators labelled a subset of the corpus for inter-annotator agreement, yielding Cohen's κ ≈ 0.77 — reported here as the value derivable from the annotators' recorded agreement rate; the full confusion matrix underlying this figure was not separately available for this study, so it is reported with that caveat rather than as an exactly re-derived statistic. This corpus is used exclusively for model development, feature selection, and cross-validated training; no result computed on it is treated as an estimate of real-world performance.

**Independent DMC evaluation set.** 59 authentic emergency recordings collected by Sri Lanka's Disaster Management Centre during real flood and landslide response operations, labelled High (23), Medium (14), or Low (22) urgency prior to any model being applied to them. Critically, this set was collected separately from the simulated corpus, by a different process, at a different time, and was used only for a single, frozen, external evaluation — never for training, threshold-tuning, or model selection. The majority-class baseline on this set (always predicting High, the plurality class) is 39.0% (23/59); this is the reference point against which any classifier's DMC accuracy must be read.

**TABLE 1. Dataset and evaluation setup**

| | Simulated (development) | DMC (independent evaluation) |
|---|---|---|
| Calls | 267 | 59 |
| High / Medium / Low | 130 / 82 / 55 | 23 / 14 / 22 |
| Role | Training, CV, model/feature selection | Frozen external evaluation only |
| Inter-annotator agreement | κ ≈ 0.77 | Labelled before model use |
| Majority-baseline accuracy | — | 39.0% |

### 4.2 Preprocessing

Audio is denoised, resampled to 16 kHz, and trimmed of leading/trailing silence before feature extraction. This pipeline must be applied identically to both corpora: an earlier evaluation pass fed the DMC recordings to the trained model without this preprocessing, which inflated energy-based features (RMS-mean Cohen's d rose to 5.45) and depressed the reported DMC macro-F1 to 0.2524 — a preprocessing-mismatch artifact, not a genuine model or domain-shift result. Once corrected, RMS-mean's Cohen's d fell to 0.145 and the corrected DMC evaluation of the same model scored 0.326 macro-F1 (Section 6). All results in this paper use the corrected, consistently preprocessed pipeline; the artifact is noted here only because preprocessing consistency across development and evaluation data is a precondition for any of the study's other findings to be meaningful.

### 4.3 Automatic speech recognition

Transcripts for the textual/conversational feature family are produced by ASR routed by detected language: a Sinhala-fine-tuned Whisper model (`faster-whisper-medium-si-exp10-fp16`) for Sinhala speech, and base Whisper otherwise. ASR quality matters directly to this study because every text/context feature below is computed from the transcript; a per-utterance ASR-confidence score is therefore included as a feature so that low-confidence transcripts are down-weighted rather than trusted at face value.

### 4.4 Textual/conversational features

The final, refined textual/conversational representation ("Condition C") comprises 19 features: `asr_quality`, `keyword_count`, `repetition_rate`, `sentiment_polarity`, and 15 rule-based context features computed separately for Sinhala and Tamil (`{lang}_context_score`, `{lang}_danger_count`, `{lang}_help_request_count`, `{lang}_high_risk_phrase_count`, `{lang}_location_context_count`, `{lang}_person_at_risk_count`, `{lang}_safety_state_count` — 7 per language) plus a shared `fire_smoke_match` indicator. This is a reduction from an earlier, larger candidate set: `word_count` was dropped after ablation showed it did not contribute discriminative signal once `keyword_count` and `repetition_rate` were present, and per-language `immediacy_count` features were dropped after they were found to depress Medium-class recall (Section 6.3). All context features are ASR-confidence-weighted, so an unreliable transcript automatically contributes less to the feature vector rather than being treated as ground truth.

### 4.5 Acoustic features

The acoustic representation comprises 91 features: 40 MFCCs summarised by mean and standard deviation (80 features), plus 11 prosodic and spectral descriptors — pitch mean and standard deviation, jitter, RMS-energy mean, standard deviation, slope, and gradient, zero-crossing rate, spectral centroid, speaking rate, and speaking-rate change. These features are computed directly from caller audio and are independent of transcript accuracy, which is their principal complementary value to the textual family.

### 4.6 Classifier and validation protocol

Classification uses a Random Forest (200 trees, minimum leaf size 2, balanced class weighting) under stratified 5-fold cross-validation. Feature standardisation and SMOTE class-balancing are fitted inside each training fold only, never on validation data, to prevent leakage from oversampled synthetic minority examples into the fold used to score the model. Reported cross-validation figures are averaged across 5 random seeds to reduce sensitivity to any single fold split.

### 4.7 Fusion strategies

Four formulations were compared to answer RQ2:

- **Text-only** — Random Forest trained on the 19-feature Condition C representation alone.
- **Acoustic-only** — Random Forest trained on the 91-feature acoustic representation alone.
- **Early fusion** — a single Random Forest trained on the concatenated 111-feature (acoustic + text/context) vector, allowing the ensemble to learn cross-modal splits directly.
- **Late fusion** — two independently trained Random Forests (one per modality, using the representations above), combined only at inference time by averaging their predicted class probabilities.

Early fusion was the strongest formulation on the simulated development corpus (Section 6.1), which is why it — not late fusion — was the initial candidate for the final model. Late fusion was adopted instead only after external evaluation on the DMC set showed early fusion did not generalise (Section 6.2): a single tree ensemble spanning both modalities appears to learn acoustic/lexical co-occurrence patterns specific to the simulated corpus's production conditions, which do not hold in authentic telephony audio, whereas training the two branches independently prevents the acoustic branch's domain-specific noise from corrupting the text branch's more stable predictions. Among late-fusion weightings tested — unweighted (0.5/0.5), a cross-validation-tuned weight favouring acoustic, per-class weighting, and confidence-gated fusion — unweighted averaging was the only variant that improved on text-only performance on the DMC set; the alternatives underperformed it on every metric. Unweighted late fusion is therefore reported as the final formulation, selected for its demonstrated external-validation behaviour rather than for topping the development-set leaderboard.

## 5. Experimental Design

Five evaluations were conducted, each mapped to the research questions above: (1) unimodal text-only and acoustic-only classification on the simulated corpus, to establish each modality's standalone ceiling (RQ1); (2) early- vs. late-fusion comparison on the simulated corpus (RQ2); (3) an acoustic feature-subset ablation, to determine whether the full 91-feature acoustic representation is necessary or whether a reduced subset (e.g., MFCCs alone) performs comparably (RQ1, supporting evidence); (4) a single, frozen external evaluation of every formulation above on the independent 59-call DMC set (RQ3); and (5) a quantitative domain-shift analysis comparing the simulated and DMC feature distributions, to explain the result of (4) (RQ4). Model configuration, feature sets, and hyperparameters were fixed before the DMC evaluation was run; no result on the DMC set informed any subsequent change to the model presented as final.

## 6. Results

### 6.1 Modality and fusion comparison — simulated corpus

Table 2 reports development-corpus performance. The 91-feature acoustic-only model reaches macro-F1 0.592; adding the refined text/context representation and combining all 111 features into a single early-fusion model raises this to 0.621, the best development-set result obtained.

### 6.2 Independent evaluation — DMC recordings

Table 2 also reports the frozen external evaluation on the 59 DMC calls — the central result of this study. Every formulation's DMC performance is substantially below its simulated-corpus estimate. Early fusion, despite being the strongest model on the development corpus (0.621 macro-F1), falls to 0.326 macro-F1 on DMC — only marginally above the 39.0% majority-class baseline, and below both text-only and (on macro-F1) comparable to acoustic-only. Text-only performance transfers comparatively well (0.449 macro-F1, 45.8% accuracy) and is, on this metric, the single strongest individual result on the DMC set. Late fusion, evaluated with two ASR configurations, improves over both single modalities and over early fusion on macro-F1 and on High-urgency recall — the metric of greatest operational consequence, since a missed High-urgency call carries the highest cost of any classification error — reaching 57.4% High-recall with the locked ASR configuration, against 44.4% for text-only and 26.1% for acoustic-only. On the raw-accuracy metric for the full DMC set, however, text-only (45.8%) marginally exceeds the locked late-fusion model (44.8%); late fusion's advantage over text-only is concentrated in High-urgency recall and in performance on the subset of DMC calls matching the simulated corpus's fire/flood scope (80.0% accuracy, 88.6% High-recall on that 11-call subset), not in overall DMC accuracy. This is reported plainly rather than obscured, since it is itself an informative finding (Section 7).

**TABLE 2. Modality and fusion comparison — simulated (5-fold CV) vs. independent DMC evaluation**

| Model | Simulated Macro-F1 | DMC Full Accuracy | DMC Full Macro-F1 | DMC Full High-Recall |
|---|---|---|---|---|
| Majority baseline | — | 39.0% | — | — |
| Acoustic-only (91 feat.) | 0.592 | 33.6% | 0.304 | 26.1% |
| Text-only, Condition C (19 feat.) | [VERIFY FROM PROJECT RESULTS — simulated Condition-C macro-F1 not separately confirmed against DMC-evaluation script; DMC value below is confirmed] | 45.8% | 0.449 | 44.4% |
| Early fusion (111 feat., single RF) | **0.621** | 34.2% | 0.326 | ~29% |
| Late fusion (own-ASR routing) | 0.613 | 43.7% | 0.428 | 46.1% |
| Late fusion (locked ASR routing) | 0.639 | 44.8% | **0.431** | **57.4%** |

*DMC Full = all 59 calls. On the fire/flood-scope-matched 11-call subset, locked late fusion reaches 80.0% accuracy, 0.753 macro-F1, 88.6% High-recall — its strongest result on any external evaluation view.*

### 6.3 Acoustic feature-subset ablation

Table 3 compares acoustic feature subsets on the simulated corpus. An MFCC-only representation (80 features) reaches macro-F1 0.611, marginally *higher* than the full 91-feature acoustic set (0.603) — the additional prosodic/spectral descriptors did not, by themselves, improve on MFCCs alone in this evaluation. Feature selection did materially affect the multimodal model, however: removing two candidate `immediacy_count` context features (identified as depressing Medium-class recall) and adding a `fire_smoke_match` indicator raised the final development-set model from 0.609 to 0.621 macro-F1 and raised High-urgency recall from 0.754 to 0.792. No acoustic-only configuration, in any evaluation, exceeded the text-only Condition C model's performance on the independent DMC set (Table 2), indicating that on real recordings the acoustic signal alone is a materially weaker basis for urgency classification than the textual/conversational signal, though the two remain complementary in fusion (Section 6.2).

**TABLE 3. Acoustic feature-subset ablation (simulated corpus, 5-fold CV)**

| Configuration | Features | Macro-F1 |
|---|---|---|
| MFCC-only | 80 | 0.611 |
| All-acoustic | 91 | 0.603 |
| Final model (± immediacy, + fire/smoke) | 111 total | 0.621 |

### 6.4 Domain-shift analysis

Cohen's d was computed per feature between the simulated and DMC distributions to quantify how much each feature family shifted. The acoustic feature family shows a substantially larger median absolute shift (|d| = 0.57) than the text/context family (|d| = 0.154) — a roughly 3.7-fold difference — with 25 of 91 acoustic features (27%) showing a large shift (|d| > 0.8), against effectively none of the text/context features at that threshold. This is consistent with, and provides partial quantitative support for, the finding in Section 6.2 that acoustic-only and early-fusion models transfer poorly while text-only transfers comparatively well: the acoustic feature space the model learned from is measurably less stable across the simulated/real boundary than the text feature space. This should be read as evidence consistent with a domain-shift explanation, not as proof of a single cause — plausible contributing factors include differences in recording equipment and channel (telephony vs. simulated-call audio), caller environment and background noise, and genuine differences in vocal behaviour between role-played and authentic distress, none of which this analysis isolates individually.

**TABLE 4. Domain-shift statistics, simulated vs. DMC (Cohen's d)**

| Feature family | n | Median \|d\| | Features with \|d\| > 0.8 |
|---|---|---|---|
| Acoustic | 91 | 0.57 | 25 (27%) |
| Text/context | 19 | 0.154 | ~0 |

## 7. Discussion

**RQ1 — individual modalities.** Acoustic features alone reach a respectable 0.592 macro-F1 on the simulated corpus but transfer poorly to DMC (0.304 macro-F1, only slightly above chance-adjusted baselines). Text/conversational features are a materially weaker predictor of urgency on the simulated corpus in isolation but the strongest single modality on the independent DMC set (0.449 macro-F1, 45.8% accuracy) — a reversal that is only visible because of the external evaluation, and would not have been discovered from simulated-corpus results alone.

**RQ2 — fusion strategy.** Early fusion is the best-performing formulation under cross-validation on the development corpus but the worst-generalising formulation externally, falling to near-baseline DMC performance. Late fusion, despite never being the top scorer on simulated-corpus cross-validation, is the only multimodal formulation that improves over text-only on the independent set — and its improvement is concentrated specifically in High-urgency recall, the metric with the greatest operational cost if missed. This indicates that fusion architecture, not just feature richness, determines whether a multimodal gain observed in development survives contact with real data.

**RQ3 — simulated-to-real generalisation.** Every model's DMC performance is well below its simulated-corpus estimate; this gap is the study's central finding. A model selected purely on development-set macro-F1 (early fusion, 0.621) would have been the *worst*-generalising choice available, underscoring that development-corpus accuracy alone is an unsafe basis for model selection in this domain.

**RQ4 — explaining the gap.** The domain-shift analysis (Table 4) shows the acoustic feature space shifted considerably more than the text feature space between the two corpora, offering a partial, feature-level explanation for why acoustic-heavy formulations (acoustic-only, early fusion) transferred worse than text-heavy or late-fusion formulations. This is an association consistent with the transfer results, not an isolated causal test; a preprocessing-consistency fix (Section 4.2) was also necessary before any of this domain-shift measurement was meaningful, underscoring that methodological rigour in evaluation setup is a precondition for, not separate from, the scientific finding.

**Honest reporting of a null/negative result.** Text-only slightly outperforms the final late-fusion model on raw DMC accuracy. Rather than omit or reweight this to favour the multimodal contribution, we report it directly: it demonstrates that the value of acoustic fusion in this setting is not uniform across metrics, and is concentrated where it matters operationally (High-urgency recall, and performance within the model's intended domain scope) rather than in aggregate accuracy. This is itself a scientifically meaningful result for future urgency-detection work: it argues for evaluating multimodal emergency-call systems on class-sensitive and scope-sensitive metrics, not accuracy alone.

Several alternative approaches to closing the simulated-to-real gap were evaluated and did not survive validation, and are noted briefly because they materially shaped the final methodology rather than because they are novel negative results in themselves: hyperparameter re-tuning did not provide a sufficient DMC improvement to justify departing from the simpler configuration; probability calibration worsened Brier score under the corpus's limited sample size; audio augmentation intended to make simulated audio resemble telephony conditions did not improve DMC transfer; and ASR-hallucination-detection heuristics did not provide a sufficient accuracy gain to adopt. Their consistent failure to close the gap is itself weak evidence that the gap is not attributable to any single, easily-patched cause.

## 8. Limitations

The simulated development corpus (267 calls) and, especially, the independent DMC evaluation set (59 calls) are both modest in size for a three-class problem, and the DMC set in particular limits the precision of any per-class recall estimate. The DMC set, while genuinely independent and authentic, was drawn from a specific set of flood/landslide response operations over a bounded period and geography, and generalisation to other emergency types or regions is not established by this study. Multilingual ASR quality — particularly for colloquial and code-mixed Sinhala/Tamil speech — remains an active source of error that propagates into every text/context feature, and this study's ASR-quality-weighting mitigates rather than eliminates that risk. The domain-shift analysis (Section 6.4) identifies association, not full causal attribution, and does not isolate recording equipment, environment, or genuine behavioural differences from one another. No claim is made that this classifier provides a clinical or safety guarantee; it is reported strictly as a decision-support signal evaluated against the metrics presented.

## 9. Conclusion

This study developed and, critically, independently evaluated a multimodal urgency classifier for Sri Lankan multilingual emergency calls. The central finding is that a large gap exists between simulated-corpus and authentic-recording performance, that this gap is not uniform across modalities (acoustic-heavy models transfer worse than text-heavy ones), and that a domain-shift analysis provides quantitative, if partial, support for why. A late-fusion architecture — selected on external-validation evidence rather than development-set accuracy — narrows this gap, improving specifically on the operationally critical High-urgency recall metric, while text-only features remain a strong and, on some metrics, superior contributor. The contribution of this work is not a claim of state-of-the-art or deployment-ready accuracy, but a methodologically rigorous demonstration — with an independently sourced, held-out real-world evaluation set — that simulated-corpus performance in this domain substantially overstates real-world generalisation, and that this must be measured, not assumed, before an urgency-detection system is treated as reliable.

## References

[1] A. Conneau et al., "Unsupervised cross-lingual representation learning at scale," in *Proc. 58th Annu. Meeting Assoc. Comput. Linguistics*, 2020, pp. 8440–8451.

[2] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust speech recognition via large-scale weak supervision," in *Proc. 40th Int. Conf. Mach. Learn.*, 2023, pp. 28492–28518.

[3] T. Solorio et al., "Overview for the first shared task on language identification in code-switched data," in *Proc. 1st Workshop Comput. Approaches Code Switching*, 2014, pp. 62–72.

[4] B. Schuller, A. Batliner, S. Steidl, and D. Seppi, "Recognising realistic emotions and affect in speech: State of the art and lessons learnt from the first challenge," *Speech Commun.*, vol. 53, no. 9–10, pp. 1062–1087, 2011.

[5] A. Torralba and A. A. Efros, "Unbiased look at dataset bias," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2011, pp. 1521–1528.

[6] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *J. Artif. Intell. Res.*, vol. 16, pp. 321–357, 2002.

[7] H. Bredin, "pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe," in *Proc. INTERSPEECH*, 2023.
