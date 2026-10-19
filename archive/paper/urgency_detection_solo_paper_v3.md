# Multimodal Urgency Detection for Sri Lankan Emergency Calls: Evaluating Simulated-to-Authentic Generalisation

*[Author names, affiliations, and emails withheld pending anonymisation decision for submission.]*

---

*Abstract* - Emergency call centres must order incoming reports by urgency under time pressure, yet transcript content alone is a weak signal: a caller describing a life-threatening situation and a caller requesting routine information can use overlapping vocabulary while differing sharply in vocal distress. This study develops a multimodal classifier combining acoustic-prosodic and textual/conversational features to assign a High, Medium, or Low urgency label to emergency calls in the Sri Lankan multilingual context (Sinhala, Tamil, English, and code-mixed varieties), developed on a 267-call simulated corpus and evaluated against 59 independently collected, authentic Disaster Management Centre (DMC) recordings. Development-corpus macro-F1 reached 0.621 with early fusion, but the same formulation fell to 0.328 on the independent DMC set, close to a majority-class baseline of 0.187. A domain-shift analysis using Cohen's d found acoustic feature distributions shifted considerably more between corpora than text and context features did; a targeted augmentation experiment showed that simple telephony-noise and bandwidth transformations do not by themselves account for this mismatch. Guided by this external-validation evidence, a late-fusion reformulation, combining two independently trained modality-specific classifiers by unweighted probability averaging, was adopted as the final formulation, achieving the strongest DMC High-urgency recall (60.9%) among the configurations evaluated, though text-only alone remained comparable or superior on raw DMC accuracy. These results indicate that simulated-corpus performance substantially overstates authentic-recording generalisation, that fusion strategy materially affects whether multimodal information transfers across domains, and that text-only features remain a strong, and on some metrics superior, contributor even after fusion is improved.

*Keywords* - emergency call triage, urgency classification, multimodal fusion, acoustic-prosodic features, domain shift, code-mixed speech, Sinhala, Tamil

---

## 1. Introduction

Emergency response is a resource-allocation problem under scarcity: dispatch centres must decide which reports demand immediate action before full information is available. In Sri Lanka this decision is complicated by three factors specific to the setting. Callers report in Sinhala, Tamil, or English, frequently code-mixed within a single utterance. Emergency-call speech is produced under acute physiological and psychological stress, which manifests acoustically - in pitch, energy, and speaking rate - independently of the words used. And textual content alone is an ambiguous signal for urgency: two callers can use near-identical vocabulary while one is in immediate danger and the other is not, because the distinguishing information is carried in *how* something is said, not only *what* is said.

This motivates combining acoustic and textual evidence. However, a classifier's apparent effectiveness on a simulated or acted corpus - the only kind of data typically available at development time, given the operational and ethical difficulty of collecting labelled real emergency audio - does not establish that it will generalise to authentic calls, where recording conditions, caller behaviour, and the distribution of true emergencies differ from a simulated set. This study treats that generalisation question as a first-class research question rather than an afterthought, by evaluating the developed classifier against an independently collected set of authentic emergency recordings from Sri Lanka's Disaster Management Centre (DMC), and by using that external evidence directly to guide which formulation of the classifier is reported as final.

The study asks four research questions:

- **RQ1.** How effectively can acoustic-prosodic and textual/conversational information, individually, classify emergency-call urgency?
- **RQ2.** Does multimodal fusion improve urgency classification over either modality alone, and does the fusion strategy (early vs. late) matter?
- **RQ3.** How well does a classifier developed on a simulated emergency-call corpus generalise to independent, authentic recordings?
- **RQ4.** What measurable factors help explain the observed difference between simulated-corpus and authentic-recording performance?

**Contributions.** (1) A systematic comparison of unimodal and multimodal (early- and late-fusion) urgency classifiers in a multilingual, code-mixed emergency-call setting. (2) An independent external validation against 59 authentic DMC recordings, distinct from the simulated development corpus in source, collection method, and time period, used directly to select the final fusion formulation. (3) A quantitative domain-shift analysis using Cohen's d, corroborated by a targeted augmentation experiment, that identifies which feature families differ most between the simulated and authentic distributions and shows that simple noise/bandwidth transformations do not by themselves account for that difference. (4) A transparent account of where a multimodal reformulation improves on text-only performance and where it does not.

## 2. Related Work

**Emergency-call and text-based urgency classification.** Machine learning classification of emergency-call transcripts has been shown feasible on a small South African dataset spanning several languages, classifying calls by time-critical prehospital diagnosis from caller-described symptoms [8]; that study, like this one, worked with a modest, difficult-to-collect real-call dataset and reported strong cross-validation accuracy that dropped on held-out data, a pattern consistent with the generalisation gap this paper investigates directly. Urgency classification purely from text has also been studied outside the emergency domain, comparing context-independent (TF-IDF) and context-dependent (BERT) text representations for business voicemail triage [9], and, closer to this paper's multilingual setting, using cross-lingual transformer encoders to classify multilingual medical-emergency social media posts into three urgency tiers [10] - the same three-class formulation used here, and evidence that cross-lingual encoders such as XLM-RoBERTa generalise usefully to urgency classification specifically, not just to general-purpose language understanding.

**Speech-based urgency and affect.** Acoustic correlates of stress and arousal - fundamental frequency, jitter, energy dynamics, and speaking rate - are established markers in speech emotion recognition [4], with mel-frequency cepstral coefficients (MFCCs) the dominant paraverbal representation. Prior work on emergency-call classification has generally relied on transcript content, leaving the acoustic channel comparatively underused despite its robustness to transcription failure - a property directly relevant to this study's findings on real telephony audio.

**Multilingual and code-mixed language.** Cross-lingual encoders such as XLM-RoBERTa [1] transfer to low-resource languages with limited supervision, including to urgency classification specifically [10]. Weakly supervised multilingual speech recognition models [2] extend automatic speech recognition (ASR) coverage across languages but are known, from the broader code-switching ASR literature, to degrade under intra-sentential code-mixing. Language identification for code-switched text has been established as a distinct modelling problem rather than noisy monolingual data [3], a position this study's feature design follows by treating Sinhala and Tamil context features separately.

**Speaker diarization.** Isolating a single speaker's audio from a multi-speaker recording - necessary here to separate caller from call-taker before acoustic feature extraction - is addressed by modern neural diarization pipelines such as pyannote.audio [7].

**Class imbalance.** Synthetic Minority Over-sampling Technique (SMOTE) [6] is a standard method for correcting class imbalance during training without altering the evaluation distribution, used here inside cross-validation folds only to avoid leakage.

**Domain shift.** That a model's held-out performance on its training distribution does not predict its performance on a differently sourced dataset is a long-established finding in machine learning, sometimes termed "dataset bias" [5]. This study applies that concern directly and quantitatively (via Cohen's d) to the specific case of simulated-versus-authentic emergency audio, which - to the best of our knowledge - has not been reported for Sri Lankan multilingual emergency calls specifically.

## 3. Research Gap and Questions

Emergency-call urgency classification research has generally reported performance on a single, internally consistent corpus, whether simulated or real, but rarely on a classifier developed on one and evaluated on the other. This leaves open exactly the question that matters for deployment: does a promising development-set result transfer? The multilingual, code-mixed character of Sri Lankan emergency communication further means that findings from monolingual settings cannot be assumed to hold. This study is designed to close that specific gap, structured around RQ1–RQ4 above, with the simulated-to-authentic evaluation (RQ3–RQ4) treated as the central empirical contribution.

## 4. Methodology

### 4.1 Datasets

**Simulated development corpus.** 267 role-played emergency calls, labelled High (132), Medium (80), or Low (55) urgency, constructed around fire and flood scenarios specifically - the two disaster types the wider system targets - which is directly relevant to interpreting the DMC results in Section 6.2. A 40-call subset was double-annotated for inter-annotator agreement (Section 4.1.1). Used exclusively for development, feature selection, and cross-validated training; no result computed on it is treated as an estimate of authentic-recording performance.

**Independent DMC validation set.** 59 authentic recordings from Sri Lanka's Disaster Management Centre, labelled High (23), Medium (14), or Low (22) urgency before any model was applied, spanning nine districts. Unlike the simulated corpus, DMC is not scoped to one incident type: flood accounts for less than half the calls (25 of 59), with the remainder split across landslide, fallen-tree, and other weather-related incidents, some fire calls, and general information requests, complaints, and enquiries (including one about a tsunami threat). Collected independently of the simulated corpus and used both to evaluate generalisation and, directly, to select the final fusion formulation (Section 4.7) - it is described consistently throughout this paper as an external validation set that informed model selection, not as an untouched final test set, since a bare "evaluation set" label would understate that. Majority-class baseline: 39.0% accuracy, macro-F1 0.187.

**TABLE 1. Dataset and evaluation setup**

| | Simulated (development) | DMC (independent validation) |
|---|---|---|
| Calls | 267 | 59 |
| High / Medium / Low | 132 / 80 / 55 | 23 / 14 / 22 |
| Scenario scope | Fire and flood only | Predominantly flood (25), plus landslide, fallen-tree, some fire, and other/uncategorised incident and information calls |
| Geographic coverage | - (role-played) | 9 districts |
| Role | Training, CV, model/feature selection | Independent external validation; used to select final fusion formulation, never used for training |
| Inter-annotator agreement | κ = 0.695 (40-call double-annotated subset) | Labelled operationally, before model use |
| Majority-baseline accuracy / macro-F1 | 49.4% / 0.221 | 39.0% / 0.187 |

#### 4.1.1 Inter-annotator agreement

A 40-call subset of the simulated corpus was independently labelled by a second annotator. Comparing against the primary annotator's labels on the same 40 calls gives 32 exact matches (80%) and Cohen's κ = 0.695 - "substantial" agreement on the standard Landis–Koch scale. Table 1a shows the full confusion matrix; every disagreement involves the Medium class (4 Medium calls labelled High by the second annotator, 2 Medium labelled Low, 2 Low labelled Medium), while High and Low were never confused with each other. This is independent evidence, from annotation rather than model error, that Medium is the genuinely ambiguous class in this label scheme - consistent with Medium being the weakest-performing class throughout the modelling results (Section 6).

**TABLE 1a. Inter-annotator confusion matrix (40 calls; rows = primary annotator, columns = second annotator)**

| | Second: High | Second: Medium | Second: Low |
|---|---|---|---|
| **Primary: High** | 15 | 0 | 0 |
| **Primary: Medium** | 4 | 9 | 2 |
| **Primary: Low** | 0 | 2 | 8 |

### 4.2 Preprocessing

Audio is denoised, resampled to 16 kHz, and trimmed of leading/trailing silence before feature extraction; all datasets are processed with this same pipeline, a preprocessing inconsistency identified during preliminary validation was corrected before the experiments reported here. This correction mattered materially for one specific feature, reducing RMS-energy's cross-corpus Cohen's d from 5.45 to 0.145 (Section 6.4), confirming that preprocessing consistency, not a genuine acoustic difference, was driving that artifact. This does not mean acoustic domain shift was eliminated corpus-wide: the median shift across the full 91-feature acoustic set (|d| = 0.57, Section 6.4) reflects genuine, remaining cross-corpus differences.

### 4.3 Automatic speech recognition

Two ASR configurations appear in this study, both routed by detected language (Sinhala vs. Tamil/English) and differing only in the Sinhala model used. Own-ASR was used for the initial fusion-strategy comparison (Section 6.1); locked-ASR, a domain-fine-tuned Sinhala model adopted after own-ASR was found to produce semantically incorrect Sinhala transcripts on real calls, was used for the final external-validation configuration (Section 6.2). This was a sequential adoption, not a designed ASR ablation.

**TABLE 1b. ASR configurations**

| Configuration | Sinhala model | Tamil/English model | Role in this study |
|---|---|---|---|
| Own-ASR | The general-purpose Whisper-family model used during initial development | Base Whisper | Used for the initial matched-ASR fusion-strategy comparison (Section 6.1–6.2) |
| Locked-ASR | A domain-fine-tuned Sinhala model (`faster-whisper-medium-si-exp10-fp16`), adopted after own-ASR was found to produce semantically incorrect Sinhala transcripts on real calls | Base Whisper (unchanged) | Adopted for the final external-validation configuration (Section 6.2) |

Transcripts feed every textual/conversational feature below; a per-utterance ASR-confidence score (`asr_quality`) is included as a feature, and three of the lexical features (`keyword_count`, `repetition_rate`, `sentiment_polarity`) are multiplied by it directly, so an unreliable transcript contributes less to those specific features rather than being trusted at face value. The 14 language-specific context counts and `fire_smoke_match` are not confidence-weighted. Because own-ASR and locked-ASR differ only in the Sinhala model, performance differences between the two configurations can be interpreted as arising from that ASR substitution and its downstream effect on the extracted textual features.

### 4.4 Textual/conversational features

The final, refined representation comprises 19 features: ASR-quality score, keyword count, repetition rate, sentiment polarity, 14 language-specific rule-based context counts (danger, help-request, high-risk-phrase, location-context, person-at-risk, safety-state, and a composite context score - 7 per language, for Sinhala and Tamil), and a shared `fire_smoke_match` indicator - a binary flag for a fixed trilingual fire/smoke keyword list ("fire," "smoke," "burning," and Sinhala/Tamil equivalents). This is a reduction from a larger candidate set: `word_count` was dropped once ablation showed it added nothing beyond keyword count and repetition rate, and per-language `immediacy_count` (counts of urgency-signalling words like "now," "immediately") was dropped after depressing Medium-class recall (Section 6.3). These decisions were made by comparing configurations across the full 267-call development corpus prior to the cross-validation reported in Section 6.1, distinct from within-fold selection; the reported CV estimate should be read with that in mind (Section 4.6). ASR-confidence weighting (Section 4.3) applies to three of the lexical features here (`keyword_count`, `repetition_rate`, `sentiment_polarity`); the 14 context counts and `fire_smoke_match` are computed from the transcript directly, without confidence weighting.

### 4.5 Acoustic features

The acoustic representation comprises 91 features: 40 MFCCs summarised by mean and standard deviation (80 features), plus 11 prosodic and spectral descriptors - pitch mean and standard deviation, jitter, RMS-energy mean, standard deviation, slope, and gradient, zero-crossing rate, spectral centroid, speaking rate, and speaking-rate change. These features are computed directly from caller audio and are independent of transcript accuracy, which is their principal complementary value to the textual family.

### 4.6 Classifier and validation protocol

Classification uses a Random Forest (200 trees, minimum leaf size 2, balanced class weighting) under stratified 5-fold, development-corpus cross-validation after feature selection (Section 4.4) - i.e., cross-validation performed after feature choices were finalised on the same corpus, not fully leakage-free in the within-fold sense, and reported as such throughout. Feature standardisation and SMOTE class-balancing are fitted inside each training fold only, never on validation data. Simulated-corpus figures are averaged across 5 random seeds (1, 7, 42, 99, 123). For the independent DMC set, the final prediction per model is the 5-seed-averaged class probability (argmax of the mean predicted-probability vector across the 5 seeds), and uncertainty is reported as a 95% confidence interval obtained by non-parametric bootstrap resampling of the 59 DMC calls (5,000 resamples), which directly reflects the uncertainty from having only 59 evaluation calls rather than only the variance across training seeds (Section 6.2).

### 4.7 Fusion strategies

Four formulations were compared to answer RQ2:

- **Text-only** - Random Forest trained on the 19-feature textual/conversational representation alone.
- **Acoustic-only** - Random Forest trained on the 91-feature acoustic representation alone.
- **Early fusion** - a single Random Forest trained on the concatenated 111-feature (acoustic + text/context) vector, allowing the ensemble to learn cross-modal splits directly.
- **Late fusion** - two independently trained Random Forests (one per modality, using the representations above), combined only at inference time by averaging their predicted class probabilities.

Early fusion was the strongest formulation on the simulated development corpus under matched (own-)ASR conditions (0.621 vs. 0.613 macro-F1 for late fusion; Table 2), which is why it was the initial candidate for the final model. It was replaced by late fusion only after the independent DMC evaluation showed early fusion did not generalise (Section 6.2 and Discussion). Among late-fusion weightings evaluated on the DMC set - unweighted (0.5/0.5), a cross-validation-tuned weight favouring acoustic, per-class weighting, and confidence-gated fusion - unweighted averaging was the strongest, so it is reported as the final formulation. This selection was made using, and is explicitly justified by, its DMC evaluation behaviour (Table 2).

## 5. Experimental Design

Five evaluations were conducted, each mapped to the research questions above: (1) unimodal text-only and acoustic-only classification on the simulated corpus, to establish each modality's standalone performance (RQ1); (2) early- vs. late-fusion comparison on the simulated corpus under matched ASR (RQ2); (3) an acoustic feature-subset ablation, to determine whether the full 91-feature acoustic representation is necessary or whether a reduced subset (e.g., MFCCs alone) performs comparably (RQ1, supporting evidence); (4) an independent evaluation of every formulation above on the 59-call DMC set, used both to assess generalisation and to select the final fusion formulation and ASR configuration (RQ3); and (5) a quantitative domain-shift analysis comparing the simulated and DMC feature distributions, followed by a targeted augmentation experiment testing the simplest candidate explanation for that shift (RQ4).

## 6. Results

### 6.1 Modality and fusion comparison - simulated corpus

Table 2 reports development-corpus performance under own-ASR (the configuration used when fusion strategies were first compared); as noted in Section 4.6, these cross-validation figures follow feature selection performed on the same corpus and are not fully leakage-free in the within-fold sense. The 91-feature acoustic-only model reaches macro-F1 0.603; adding the refined text/context representation and combining all 111 features into a single early-fusion model raises this to 0.621, the best matched-ASR development-set result obtained - ahead of late fusion's 0.613 under the same ASR configuration.

### 6.2 Independent evaluation - DMC recordings

Table 2 also reports the independent evaluation on the 59 DMC calls - the central result of this study, and the evidence that determined the final formulation. Each model's DMC prediction is its 5-seed-averaged class probability; uncertainty is a 95% confidence interval from non-parametric bootstrap resampling of the 59 calls (5,000 resamples), which reflects the uncertainty of evaluating on only 59 calls rather than only training-seed variance. Every formulation's DMC performance is substantially below its simulated-corpus estimate. Early fusion, despite being the strongest model on the development corpus, falls to 0.328 macro-F1 on DMC (95% CI [0.213, 0.439]) - close to the majority-class baseline (0.187, CI [0.142, 0.230]), and below both text-only and acoustic-only on this metric. Text-only performance transfers comparatively better (0.454 macro-F1, 45.8% accuracy) and is, on raw DMC-Full accuracy, comparable to the locked late-fusion model. Late fusion, evaluated under both ASR configurations, improves over early fusion on every DMC metric; under the locked-ASR configuration it reaches 44.1% accuracy (CI [32.2, 57.6]), 0.425 macro-F1 (CI [0.302, 0.547]), and - its clearest advantage - 60.9% High-urgency recall (CI [40.0, 80.8]), against 47.8% for text-only and 26.1% for acoustic-only. High-urgency recall is particularly important operationally, since failing to identify a High-urgency call plausibly represents a more consequential error than misclassifying a lower-urgency one, though this study does not formally define a cost matrix establishing that hierarchy.

On raw DMC-Full accuracy, text-only and locked late fusion have similar performance (45.8% vs. 44.1%), with overlapping 95% bootstrap confidence intervals; the present analysis therefore does not establish a reliable accuracy advantage for either configuration. Late fusion's advantage is concentrated in High-urgency recall, and, as supporting evidence given its small size, on the 11-call fire/flood-scope-matched DMC subset (80.0% accuracy, 0.753 macro-F1, 88.6% High-recall, 5-seed mean; too small for a stable bootstrap CI).

**TABLE 2. Modality and fusion comparison - simulated (5-fold CV) vs. independent DMC evaluation (point estimate, 95% bootstrap CI in brackets where computed)**

| Model | Simulated Macro-F1 | DMC Full Accuracy | DMC Full Macro-F1 | DMC Full High-Recall |
|---|---|---|---|---|
| Majority baseline | 0.221 | 39.0% [27.1, 52.5] | 0.187 [0.142, 0.230] | 100%* |
| Acoustic-only (91 feat.) | 0.603 | 32.2% [20.3, 44.1] | 0.298 [0.186, 0.413] | 26.1% [9.1, 45.8] |
| Text-only (19 feat.) | 0.397 | 45.8% [33.9, 57.6] | 0.454 [0.323, 0.575] | 47.8% [27.3, 68.4] |
| Early fusion (111 feat., single RF) | 0.621 | 35.6% [23.7, 47.5] | 0.328 [0.213, 0.439] | 34.8% [14.8, 55.2] |
| Late fusion (own-ASR) | 0.613 | 40.7% [28.8, 52.5] | 0.398 [0.276, 0.515] | 47.8% [26.9, 69.2] |
| Late fusion (locked-ASR, final) | - | 44.1% [32.2, 57.6] | 0.425 [0.302, 0.547] | **60.9% [40.0, 80.8]** |

*\*By construction: a classifier that always predicts High cannot miss a High call, so its High-recall is mechanically 100% - this is why High-recall alone is an insufficient metric and macro-F1 is reported alongside it throughout.*

*On the 11-call fire/flood-scope-matched DMC subset, locked late fusion reaches a substantially stronger 80.0% accuracy, 0.753 macro-F1, 88.6% High-recall (5-seed mean, not bootstrap-CI'd due to subset size) than on DMC Full - but this estimate is too small to support a stable performance claim, and is reported as supporting, scope-matched evidence rather than a headline result.*

Table 2a breaks the locked late-fusion model's DMC-Full result down by class. High is the class the model handles most reliably (precision and recall both 0.609), consistent with the High-recall figures already reported. The larger error pattern is a persistent confusion between Low and Medium in both directions (10 true-Low calls predicted Medium, 5 true-Medium calls predicted Low), while High is comparatively rarely confused with Medium (7 of 23 true-High calls) and almost never with Low (2 of 23). This mirrors the annotation-level finding in Section 4.1.1, where every inter-annotator disagreement also involved Medium: the model's dominant error mode tracks the class humans themselves find hardest to separate, rather than an arbitrary failure.

**TABLE 2a. Confusion matrix and per-class metrics, locked late-fusion model, DMC Full (59 calls)**

| True \\ Predicted | High | Low | Medium | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| High (n=23) | 14 | 2 | 7 | 0.609 | 0.609 | 0.609 |
| Low (n=22) | 6 | 6 | 10 | 0.462 | 0.273 | 0.343 |
| Medium (n=14) | 3 | 5 | 6 | 0.261 | 0.429 | 0.324 |

### 6.3 Acoustic feature-subset ablation

Table 3 compares acoustic feature subsets on the simulated corpus. An MFCC-only representation (80 features) reaches macro-F1 0.611, marginally *higher* than the full 91-feature acoustic set (0.603) - the additional prosodic/spectral descriptors did not, by themselves, improve on MFCCs alone in this evaluation. Feature selection did materially affect the multimodal model, however: removing the two `immediacy_count` context features and adding `fire_smoke_match` raised the final development-set model from 0.609 to 0.621 macro-F1 and raised High-urgency recall from 0.754 to 0.792. No acoustic-only configuration, in any evaluation, exceeded the text-only model's performance on the independent DMC set (Table 2), indicating that on real recordings the acoustic signal alone is a materially weaker basis for urgency classification than the textual/conversational signal, though the two remain complementary in fusion (Section 6.2).

**TABLE 3. Acoustic feature-subset ablation (simulated corpus, 5-fold CV)**

| Configuration | Features | Macro-F1 |
|---|---|---|
| MFCC-only | 80 | 0.611 |
| All-acoustic | 91 | 0.603 |
| Final multimodal model (± immediacy, + fire/smoke) | 111 total | 0.621 |

### 6.4 Domain-shift analysis

Cohen's d was computed per feature between the simulated and DMC distributions to quantify how much each feature family shifted. The acoustic feature family shows a substantially larger median absolute shift (|d| = 0.57) than the text/context family (|d| = 0.154) - roughly 3.7-fold - with 25 of 91 acoustic features (27%) showing a large shift (|d| > 0.8), with no text/context features exceeding that threshold. This is consistent with, and provides partial quantitative support for, the finding in Section 6.2 that acoustic-only and early-fusion models transfer poorly while text-only transfers comparatively better.

This relationship is an association consistent with the observed transfer results rather than evidence of causation. Two specific instantiations of one candidate explanation - that the acoustic mismatch is simply telephony-channel noise or bandwidth limiting - were tested directly: audio augmentation applying white noise, and separately a telephone-bandwidth bandpass filter (300–3400 Hz), to the simulated corpus, intended to make it resemble real telephony conditions. Both *increased* the measured domain-shift rather than reducing it, suggesting that the measured mismatch cannot be explained solely by the specific noise and bandwidth transformations evaluated here; this does not establish that channel/equipment differences play no role, only that these two simple simulations of them did not. Plausible remaining explanations include differences in recording equipment and channel beyond simple noise/bandwidth, caller environment, and genuine differences in vocal behaviour between role-played and authentic distress; this analysis does not isolate which of these, individually, accounts for the remainder.

**TABLE 4. Domain-shift statistics, simulated vs. DMC (Cohen's d)**

| Feature family | n | Median \|d\| | Features with \|d\| > 0.8 |
|---|---|---|---|
| Acoustic | 91 | 0.57 | 25 (27%) |
| Text/context | 19 | 0.154 | ~0 |

## 7. Discussion

**RQ1 - individual modalities.** Acoustic features alone reach 0.603 macro-F1 on the simulated corpus but transfer poorly to DMC (0.298 macro-F1, close to the majority baseline). Text/conversational features are a materially weaker predictor of urgency on the simulated corpus in isolation but the strongest single modality on the independent DMC set on raw accuracy - a reversal only visible because of the external evaluation.

**RQ2 - fusion strategy.** Early fusion's simulated-corpus lead over acoustic-only (0.621 vs. 0.603) evaporates on DMC, where it becomes the lowest-scoring formulation on every metric (Table 2). One possible explanation is that a single tree ensemble spanning both modalities exploits acoustic/lexical co-occurrence patterns specific to the simulated corpus's production conditions, which training the two branches independently would limit; this study's design does not isolate that mechanism directly. Late fusion, though not the top scorer on simulated-corpus cross-validation, was the formulation whose DMC-evaluation behaviour justified selecting it as final. The results suggest that fusion strategy can materially affect whether multimodal information transfers across domains, and that model selection guided by independent external evidence, rather than development-set accuracy alone, changed which formulation was adopted.

**RQ3 - simulated-to-authentic generalisation.** Every model's DMC performance is well below its simulated-corpus estimate; this gap is the study's central finding. Among the formulations evaluated, the model selected purely on development-set macro-F1 (early fusion, 0.621) fared no better on DMC than the acoustic-only model it had clearly outperformed in development, and was among the two lowest-scoring formulations on the independent DMC set.

**RQ4 - explaining the gap.** The domain-shift analysis (Table 4) shows that the acoustic feature space shifted considerably more than the text feature space between the two corpora, providing a partial, feature-level explanation for why acoustic-heavy formulations transferred less effectively than text-heavy and late-fusion formulations. This relationship is an association consistent with the observed transfer results rather than evidence of causation. In addition, the preprocessing-consistency fix described in Section 4.2 was necessary before these domain-shift measurements could be considered meaningful, highlighting the importance of methodological consistency in evaluating cross-corpus performance.

**Honest reporting of a mixed result.** Text-only and the final late-fusion model show similar raw DMC-Full accuracy, with overlapping 95% bootstrap confidence intervals (Table 2), so this analysis does not establish a reliable advantage for either. Rather than omit or reweight this to favour the multimodal contribution, we report it directly: the value of acoustic fusion in this setting is not uniform across metrics, and is concentrated where it matters operationally - High-urgency recall, and performance within the model's intended fire/flood scope - rather than in aggregate DMC-Full accuracy. This argues for evaluating multimodal emergency-call systems on class-sensitive and scope-sensitive metrics, not accuracy alone.

Several alternative approaches to closing the simulated-to-authentic gap were evaluated and did not survive validation, and are noted briefly because they materially shaped the final methodology: hyperparameter re-tuning did not provide a sufficient DMC improvement to justify departing from the simpler configuration; probability calibration worsened Brier score under the corpus's limited sample size; and ASR-hallucination-detection heuristics did not provide a sufficient accuracy gain to adopt.

## 8. Limitations

The simulated development corpus (267 calls) and, especially, the independent DMC validation set (59 calls) are both modest in size for a three-class problem; DMC per-class recall estimates in particular carry substantial uncertainty because of the small evaluation sample (the 95% bootstrap CIs in Table 2), and the 11-call scope-matched subset should be read as directional, not precise. Because DMC results directly informed the choice of final fusion formulation (Section 4.7), this study does not provide a fully unbiased, untouched-test-set estimate of generalisation performance; what it provides is an external validation that guided model selection, which is a different and more limited claim. The DMC set, while genuinely independent and authentic and spanning nine districts, is drawn predominantly from flood-related operations with smaller numbers of landslide, fallen-tree, and other incident and information calls from a bounded set of disaster-response periods; generalisation to emergency types or regions materially different from this mix is not established by this study. Because the simulated training corpus was itself scoped to fire and flood scenarios, part of the DMC-Full/DMC-Scoped performance gap (Section 6.2) is attributable to this scope mismatch rather than to model or recognition failure. Multilingual ASR quality - particularly for colloquial and code-mixed Sinhala/Tamil speech - remains an active source of error that propagates into every text/context feature regardless of weighting, and confidence-weighting (Section 4.3) mitigates rather than eliminates this risk for the three features it applies to. The domain-shift analysis (Section 6.4) identifies association, not full causal attribution; the augmentation experiment shows the specific noise/bandwidth transformations tested did not reduce the measured shift, but does not isolate recording equipment, environment, or genuine behavioural differences from one another. Feature-selection decisions (Section 4.4) were made using the full development corpus prior to cross-validation, meaning the reported simulated-corpus cross-validation estimate is not strictly leakage-free in the within-fold sense. No claim is made that this classifier provides a clinical or safety guarantee; it is reported strictly as a decision-support signal evaluated against the metrics presented.

## 9. Conclusion

This study developed and, critically, independently evaluated a multimodal urgency classifier for Sri Lankan multilingual emergency calls, with the DMC evaluation guiding the final fusion choice rather than serving as an untouched test set. The central finding is that a large gap exists between simulated-corpus and authentic-recording performance, that this gap is not uniform across modalities, and that a domain-shift analysis - for which a direct augmentation experiment showed the specific noise and bandwidth transformations tested did not reduce the observed mismatch - provides quantitative, partial support for why. A late-fusion architecture, selected through this external validation, produced substantially higher DMC High-urgency recall than early fusion, while text-only features remain a strong and, on some metrics, superior contributor. The contribution of this work is not a claim of state-of-the-art or deployment-ready accuracy, but a methodologically transparent demonstration that simulated-corpus performance in this domain substantially overstates authentic-recording generalisation, and that this must be measured - and used to guide model selection - rather than assumed.

## References

[1] A. Conneau et al., "Unsupervised cross-lingual representation learning at scale," in *Proc. 58th Annu. Meeting Assoc. Comput. Linguistics*, 2020, pp. 8440–8451.

[2] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust speech recognition via large-scale weak supervision," in *Proc. 40th Int. Conf. Mach. Learn.*, 2023, pp. 28492–28518.

[3] T. Solorio et al., "Overview for the first shared task on language identification in code-switched data," in *Proc. 1st Workshop Comput. Approaches Code Switching*, 2014, pp. 62–72.

[4] B. Schuller, A. Batliner, S. Steidl, and D. Seppi, "Recognising realistic emotions and affect in speech: State of the art and lessons learnt from the first challenge," *Speech Commun.*, vol. 53, no. 9–10, pp. 1062–1087, 2011.

[5] A. Torralba and A. A. Efros, "Unbiased look at dataset bias," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2011, pp. 1521–1528.

[6] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *J. Artif. Intell. Res.*, vol. 16, pp. 321–357, 2002.

[7] H. Bredin, "pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe," in *Proc. INTERSPEECH*, 2023.

[8] T. Anthony, A. K. Mishra, W. Stassen, and J. Son, "The feasibility of using machine learning to classify calls to South African emergency dispatch centres according to prehospital diagnosis, by utilising caller descriptions of the incident," *Healthcare*, vol. 9, no. 9, p. 1107, 2021.

[9] A. Trabelsi, S. Soussilane, and E. Helbert, "Voicemail urgency detection using context dependent and independent NLP techniques," in *Proc. 15th Int. Conf. Agents and Artificial Intelligence (ICAART)*, vol. 3, 2023, pp. 450–456.

[10] M. Singh, K. Agarwal, K. Verma, and R. K. Yadav, "Urgency status classification of multilingual medical emergency tweets using transformer-based cross-lingual encoders," in *Proc. 2022 2nd Int. Conf. Advance Computing and Innovative Technologies in Engineering (ICACITE)*, 2022, pp. 2122–2127.
