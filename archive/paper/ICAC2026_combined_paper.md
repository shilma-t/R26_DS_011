# A Multilingual Multi-Channel Emergency Intake and Triage Pipeline for Sri Lanka

**S. Thanseel**, Faculty of Computing, Sri Lanka Institute of Information Technology (SLIIT), Malabe, Sri Lanka — it22644480@my.sliit.lk

**E. A. K. Edirisinghe**, Faculty of Computing, SLIIT, Malabe, Sri Lanka — it22129758@my.sliit.lk

**D. M. S. G. Karunarathna**, Faculty of Computing, SLIIT, Malabe, Sri Lanka — it22263476@my.sliit.lk

**A. N. Siriwardena**, Faculty of Computing, SLIIT, Malabe, Sri Lanka — it22504418@my.sliit.lk

---

*Abstract*—Emergency intake in Sri Lanka arrives over multiple channels, in three official languages that callers freely mix, and under acute stress. This paper presents an integrated pipeline that accepts both short message service (SMS) and voice reports, converts each into a structured incident record, assigns an urgency level, and correlates records that describe the same real-world event. Four components are contributed. A hybrid triage component restricts large language model (LLM) use to stages where rule-based extraction is demonstrably brittle, wraps every generative call in an evidence-grounding guard, and assigns priority using the Medical Priority Dispatch System; an adaptive router reduces LLM calls by 58.3% without accuracy loss. A closed-loop dialogue component decides, turn by turn, which single missing field to request, using an auditable finite-state clarification policy rather than a generative selector; it raises entity F1 from 0.278 to 0.429 and spoken-response language match from 41.0% to 100%. A multimodal urgency component fuses 91 acoustic, 5 lexical, and 15 context features and attains a macro-F1 of 0.621 on a 267-call simulated corpus, and is then evaluated without tuning on 59 authentic Disaster Management Centre (DMC) recordings, for which caller speech is isolated by verified speaker diarization. A correlation component jointly encodes accumulated incident history rather than scoring report pairs, raising area under the receiver operating characteristic curve from 0.8442 to 0.9638 on a geographically confusable split. We report the external-validation gap on real recordings honestly, including a documented degradation of automatic speech recognition on telephony audio, and identify it as the dominant limitation of the deployed system.

*Keywords*—emergency response, multilingual natural language processing, code-mixing, speech urgency detection, speaker diarization, dialogue management, event correlation, Sinhala, Tamil

---

## I. Introduction

Sri Lanka experiences recurrent floods, landslides, and fires in which the public reports incidents through whichever channel is available: a text message, a telephone call, or both. A national emergency service must therefore accept heterogeneous input, understand it in Sinhala, Tamil, or English — frequently mixed within a single utterance — and convert it into an actionable dispatch decision.

Three difficulties compound. First, the input is linguistically irregular. Sinhala and Tamil are both official languages and English serves as a link language, and callers do not keep them separate; the romanized varieties colloquially termed Singlish and Tanglish are the ordinary register of urban reporters rather than an edge case. Second, the input is incomplete. A first message or a first utterance rarely contains everything needed to dispatch, so the system must decide what to ask next rather than merely transcribe. Third, the same event generates several reports across channels and languages, while distinct events at the same landmark share almost identical vocabulary; naive deduplication by lexical similarity is actively misleading.

A further requirement cuts across all three. Dispatch is a resource-allocation decision under scarcity, so reports must be ordered by urgency. Textual content alone is a weak signal for this: a caller reporting a life-threatening situation and a caller requesting information may use overlapping vocabulary while differing sharply in vocal distress.

This paper presents an integrated pipeline addressing these requirements through four components, and contributes the following:

1. A hybrid SMS triage architecture in which LLM use is confined to stages with demonstrated rule-based failure, every generative call is verified against its source text, and priority is assigned on an established remote-triage standard rather than an ad hoc scale (Section V).
2. A closed-loop voice dialogue system whose clarification policy is an explicit affect- and uncertainty-aware finite-state decision maker, so that every question asked is auditable and bounded (Section VI).
3. A multimodal urgency classifier combining acoustic, lexical, and rule-based context features, frozen prior to evaluation and validated externally on authentic emergency recordings with speaker-isolated caller audio (Section IV).
4. A joint incident-history matcher that scores a candidate report against the whole accumulated history rather than against individual earlier reports, operationalized through a cost-sensitive merge/separate/defer policy (Section VII).

Section II reviews related work, Section III describes the integrated architecture, Sections IV–VII present the four components, Section VIII discusses integration and limitations, and Section IX concludes.

## II. Related Work

**Multilingual and code-mixed understanding.** Cross-lingual encoders such as XLM-RoBERTa [1] transfer to low-resource targets with modest supervision, suiting domains with only a few hundred annotated utterances. Weakly supervised multilingual speech models [2] extend recognition coverage but degrade on intra-sentential code-mixing. Language identification for code-switched text has been treated as a shared task [3], establishing that mixed varieties are better modelled as distinct classes than as noisy monolingual instances; we follow that position throughout.

**Speech-based urgency and affect.** Acoustic correlates of stress — fundamental frequency, jitter, energy dynamics, speaking rate — are established in emotion recognition [4], with mel-frequency cepstral coefficients the dominant paraverbal representation. Prior emergency call classification has relied largely on transcript content; the acoustic channel is comparatively underused despite being robust to transcription failure, a property that proves decisive in Section IV.

**Extraction and dialogue.** Conditional random fields [5] and bidirectional LSTM-CRF architectures [6] preceded transformer fine-tuning for span extraction and remain informative small-corpus baselines. Task-oriented dialogue systems maintain a belief over slot values and select a system act [7], while clarification question generation has mainly targeted open-domain information seeking, ranking candidates by expected informational value [8]. Emergency call-taking differs: the slot inventory is fixed by protocol, the cost of a slow question is measured in response time, and a distressed caller may be unable to answer a well-formed but tonally inappropriate question.

**Protocols and correlation.** Operational call-handling guidance converges on a small set of information categories — where, what, who, life safety — and on a minimum data set required before committing resources. MPDS [9] is designed for remote triage from a reported description, unlike field protocols such as START [10] whose core checks require physical assessment. For correlation, multilingual cross-document event coreference is established as a meaningful task [11] and pairwise cross-encoders are the default unit of comparison, with wider context typically introduced around the pair or after pairwise scoring; Section VII replaces the pair as the evidence unit rather than augmenting it.

## III. System Architecture

The pipeline accepts reports on two intake channels and produces a ranked, deduplicated dispatch queue (Fig. 1).

*[Insert fig1_architecture.png here — Fig. 1. Four-component pipeline. Solid arrows carry structured records; dashed arrows carry signals that inform but do not alter a record.]*

The **SMS channel** (Section V) receives text messages, extracts four structured fields, and conducts a deterministic follow-up dialogue until the record is complete. The **voice channel** (Section VI) receives telephone audio and conducts a spoken clarification loop, emitting one structured emergency report per call. Both channels produce records with a common schema: incident type, location, people count, and life-safety status.

The **urgency component** (Section IV) attaches to the voice channel and consumes caller audio together with the transcript already produced by that channel's recognition stage. It emits a High, Medium, or Low label per caller turn, which is aggregated across the call. Because the dialogue component supplies the transcript, the urgency component performs no recognition of its own at inference time.

The **correlation component** (Section VII) receives completed records from both channels and decides, for each, whether it extends a known incident or opens a new one.

The components hold disjoint authority. No component both produces content and approves it: recognition cannot decide relevance, the readiness assessor cannot choose question wording, and the urgency classifier cannot alter the extracted record. This separation is what makes the pipeline auditable end to end.

## IV. Multimodal Urgency Detection

### A. Feature design

Transcript content alone is an incomplete basis for severity ordering, so the component fuses three families over caller speech. The **acoustic** family (91 features) comprises 40 mel-frequency cepstral coefficients summarized by mean and standard deviation, plus 11 prosodic and spectral descriptors — pitch mean and deviation, jitter, energy mean, deviation, slope and gradient, zero-crossing rate, spectral centroid, speaking rate, and speaking-rate change — capturing vocal distress independently of what was said. The **lexical** family (5 features) comprises a transcript-quality score, an urgency-keyword count over a 69-term trilingual lexicon, word count, repetition rate, and sentiment polarity, each multiplied by the quality score so that unreliable recognition attenuates its own contribution. The **context** family (15 features) applies curated per-language rules yielding counts for danger, help request, safety state, location context, high-risk phrase, and person at risk, plus an aggregate score, for Sinhala and Tamil separately, with a binary fire-or-smoke indicator. Word lists were derived by TF-IDF ranking followed by manual audit; high-risk phrases match as ordered word pairs within a six-token window, so that *house* and *burning* count jointly.

### B. Corpus, model, and freezing

The development corpus comprises 267 simulated calls (High 130, Medium 82, Low 55) annotated by two annotators at Cohen's κ = 0.77, median duration 6.9 s. External validation uses 59 authentic Disaster Management Centre recordings from the November–December 2025 floods and landslides across nine districts (High 23, Low 22, Medium 14), annotated before any model was applied.

Recognition is routed by oracle language label: Sinhala-fine-tuned Whisper-small for Sinhala, Whisper-base otherwise. Classification uses a random forest (200 trees, minimum leaf 2, balanced weights) under stratified five-fold cross-validation; standardization and SMOTE [14] are fitted inside each training fold only. Seven configurations were compared (Table I); the selected one removes the two immediacy counts, which degraded Medium recall, and adds the fire-or-smoke indicator.

**TABLE I. Feature ablation, simulated corpus (macro-F1, 5-fold CV)**

| Configuration | Features | Macro-F1 | High | Medium | Low |
|---|---|---|---|---|---|
| Acoustic only | 91 | 0.592 | 0.762 | 0.439 | 0.600 |
| + original keywords | 96 | 0.608 | 0.754 | 0.439 | 0.655 |
| + expanded keywords | 96 | 0.593 | 0.762 | 0.439 | 0.600 |
| + context | 112 | 0.604 | 0.746 | 0.439 | 0.655 |
| + context, quality-gated | 112 | 0.605 | 0.739 | 0.451 | 0.655 |
| − immediacy | 110 | 0.609 | 0.754 | 0.463 | 0.636 |
| **Selected** | **111** | **0.621** | **0.792** | **0.463** | **0.636** |

The fire-or-smoke indicator was chosen after inspecting development errors on the corpus used for cross-validation, so 0.621 is a development estimate and may be optimistic; the Medium-recall gain is attributable to immediacy removal, not the added indicator. Features, lexicons, hyperparameters, and data hashes were frozen in a manifest before the DMC recordings were processed.

### C. Caller isolation

All 59 DMC recordings are single-channel with caller and operator mixed, which disqualifies the acoustic family — 82% of the feature set — because an operator trained to remain calm contributes a near-constant signal that dilutes the distress variation the model depends on, and corrupts the lexical family because operator keywords would be credited to the caller.

Caller speech was isolated by speaker diarization [13], producing 2,620 segments across 41 two-speaker, 14 three-speaker, and 4 four-speaker recordings. Because diarization labels are arbitrary, one excerpt per speaker per call was generated and all 140 classified by a human annotator, yielding 63 caller, 53 operator, 9 and 9 with cross-speaker bleed, 5 noise, and 1 unclear. Ten calls had the caller split across two labels and were merged with overlapping intervals unioned, giving 37.4 min of caller speech, median 32.7 s per call.

Two heuristics were evaluated against the verified labels. Turn order identified the caller in 85% of calls; speaking time in only 44%, being anti-correlated because operators ask question sequences while the caller holds a median 41% of speech. Neither is safe to automate, so manual verification was retained.

### D. Turn-level inference and results

Isolated caller audio has median duration 32.7 s against 6.9 s for training clips, and duration-scaling features saturate rather than extrapolate under a tree ensemble. More importantly, whole-call scoring does not match deployment, since the dialogue component classifies one caller turn at a time. Turns were reconstructed by merging segments separated by under one second and discarding those under two, yielding 312 turns of median 5.2 s. Predictions are aggregated by averaging probability vectors, peak severity reported alongside. All three conditions were fixed before any result was computed.

*[DMC results pending; to be inserted on completion of the locked evaluation run.]*

One finding already visible warrants report: on authentic telephony audio the Sinhala recognition model produces fluent-looking but semantically incorrect Sinhala, yielding an urgency-keyword count of zero on every call processed. The quality metric cannot detect this, identifying only script-level hallucination. The 20 lexical and context features therefore contribute nothing on real recordings, leaving classification to the acoustic family alone.

## V. Hybrid SMS Triage

### A. Architecture

An earlier rule-based design using fixed keyword lists and classical classifiers proved brittle across six language styles. The redesign confines LLM use to stages where rule-based extraction demonstrably fails — free-text extraction from noisy code-mixed input, distress detection, and location-specificity judgments — while question generation, memory tracking, answer parsing, escalation, and priority scoring remain deterministic.

### B. Hallucination guards

Every stage permitting the model to rewrite or extract text is wrapped in verification. The extraction guard requires a verbatim quote justifying each field and independently confirms the quote exists in the source. The grammar guard separates grammatical glue from fact-bearing content words, rejecting a correction only when it introduces an ungrounded content word. Once corrected, the guard caught four distinct live failure modes: fabricated facts, unwanted translation, meta-commentary leakage, and semantic corruption.

### C. Adaptive routing and detection

An LLM-free heuristic predicts which cleaning stages a message requires before any call is spent, using established libraries where they exist and a corpus-bootstrapped fuzzy fallback for Singlish and Tanglish. It reduces LLM calls from 2,592 to 1,080, a 58.3% reduction, at grammar-stage F1 of 0.85.

Noise tagging over the corpus shows broken grammar affects 83.7% of messages, substantially exceeding spelling errors (37.2%) and code-mixing (40.4%); English messages showed the highest grammar-noise rate at 93.1%, contradicting the assumption that multilingual cleanup is primarily a typo problem.

A 500-example benchmark compared three hallucination detectors: content-word overlap (F1 0.92), embedding similarity (0.77), and natural language inference entailment (0.97).

Independent extraction with two model sizes revealed a reliability gap invisible to single-model evaluation: field agreement was 80.6% for incident and 76.9% for resources, but only 4.5% for location, the smaller model systematically over-extracting generic nouns as place names. A cross-model verification step was added in which the larger model's verdict prevails.

### D. Priority assignment

MPDS was adopted over START because its assessment is designed for remote triage from a reported description. One chief-complaint protocol is implemented per incident type, assigning codes on the MPDS scale with the triggering evidence recorded.

An initial 26-example benchmark scored 100% but was circular, sharing vocabulary with the keyword lists. An adversarial benchmark rewording the same scenarios without literal keywords dropped accuracy to 57.7% with 11 dangerous under-triage cases. Adding an independent LLM severity check, combined by taking the more urgent verdict, raised accuracy to 73.1% and reduced dangerous misses to 1 (Table II).

**TABLE II. MPDS adversarial validation**

| Configuration | Exact match | Dangerous misses |
|---|---|---|
| Circular benchmark | 100% | 0 |
| Paraphrase, keyword only | 57.7% | 11 |
| Paraphrase, keyword + LLM | 73.1% | 1 |

### E. Fine-tuning and human factors

A parallel mT5-small fine-tune reached 82.76% incident accuracy but 40% location recall, tracing to data scarcity: approximately 5% of the 864-message corpus carries a labelled location even after eightfold oversampling of location-positive examples. Strong performance on the well-represented field alongside weak performance on the sparse one is treated as a coherent result, and the model is logged as a parallel comparison rather than the authoritative extractor. A second fine-tuning target — a binary emergency-relevance classifier — rejects unrelated traffic without a per-message LLM call, motivated by the empirical finding that field emptiness alone cannot distinguish a vague real emergency from an irrelevant message.

Two mechanisms address the reporter's state rather than their language. Fatigue detection combines reply length, urgency-word presence, and consecutive failed parses, switching questions from open text to numbered multiple choice once sustained struggle is detected. Distress detection runs a grounded LLM check before any reply is parsed as a field answer, so that panicked or off-topic replies are never silently recorded as literal values; it reached F1 = 1.00 on a 24-example benchmark after an initial version that over-flagged short-but-correct answers was corrected. Live testing further showed that substring keyword matching recorded a wrong headcount by matching the English *we* inside the unrelated Singlish word *wela*, fixed by moving to whole-word matching.

## VI. Adaptive Clarification Dialogue

### A. Agent decomposition

The voice channel executes a closed loop once per caller turn over seven stages: hybrid recognition, language and code-mix identification, relevance-gated extraction, dispatch-readiness assessment, clarification policy, constrained realization with safety validation, and language-routed synthesis.

Six agents hold disjoint authority. The perception agent decides what was said but not what to do about it. The trust agent holds a veto over whether extracted spans enter the report. The readiness agent nominates a target field but cannot choose how it is requested. The affect agent owns distress and situation signals. The policy agent is the sole component permitted to choose an action. The realization agent converts that action into one validated sentence and may veto surface forms violating protocol constraints, including forms the policy agent requested.

### B. Relevance-gated extraction

A fine-tuned XLM-RoBERTa token classifier extracts seven span types: exact address, area, place type, incident, people count, injury, and trapped status. A separately fine-tuned utterance classifier decides whether spans from a given turn may be trusted, preventing off-topic speech from contaminating the report. Session state merges newly trusted spans non-destructively, so a location given in turn two survives an unintelligible turn three, and each field is timestamped on arrival.

### C. Readiness assessment and clarification policy

The readiness agent maps extracted slots onto established call-taking information categories and scores how close the accumulated report is to being actionable, rather than performing a null check. It nominates a single target field — the one whose absence most delays dispatch — but holds no authority over how that field is requested.

The policy agent selects one of seven strategies from the fused dialogue state, comprising the readiness nomination, the trust verdict, and the affect agent's distress and situation signals. It applies a calming preface under detected distress, enforces a bounded wait so that caller silence is treated as a channel failure rather than a refusal, and terminates uncooperative calls within a finite number of redirects. Because it is an explicit finite-state decision maker rather than a generative selector, every decision is inspectable, the strategy space is bounded by construction, and no question can reach the caller without passing an independent safety check by the realization agent, which did not select it. Synthesis is routed by detected language to engine-voice pairs verified for that script, since an English voice reading Sinhala text produces speech a caller cannot act on.

### D. Results

On 510 simulated multilingual calls (Table III), the fine-tuned extractor raises entity F1 from 0.278 to 0.429 over classical sequence-labelling baselines and relevance F1 from 0.783 to 0.819. The readiness agent reduces the share of calls left in a critical information state from 91.4% to 74.1%. The policy raises question diversity from 0.040 to 0.222 at a 100% safety-validation pass rate. Language routing raises spoken-response language match from 41.0% to 100%.

**TABLE III. Dialogue component, 510-call corpus**

| Module | Baseline | Proposed |
|---|---|---|
| Entity extraction (F1) | 0.278 | 0.429 |
| Utterance relevance (F1) | 0.783 | 0.819 |
| Calls in critical state | 91.4% | 74.1% |
| Question diversity | 0.040 | 0.222 |
| Response language match | 41.0% | 100% |

## VII. Joint Incident-History Matching

### A. Problem and evidence unit

During a disaster, several reports in different languages and over different channels may describe one incident while exposing different evidence fragments: one names a place, another mentions casualties, a third describes severity, and a fourth may be extremely short or noisy. Two properties make correlation hard. The evidence needed to decide is fragmented across earlier reports, so comparison against any single earlier report may be uninformative even when the incident is already known; and different incidents at the same landmark use nearly identical disaster vocabulary, so lexical or geographic similarity is actively misleading.

The controlled question addressed is therefore narrow and stated explicitly: given the same chronologically available evidence for a candidate incident, does jointly encoding the whole accumulated history outperform pairwise report matching followed by late aggregation, and how does that comparison change as evidence accumulates? Neither cross-document event coreference nor cross-encoders nor the use of cluster context is claimed as new; what is contributed is the replacement of the report pair as the unit of evidence.

### B. Benchmark and method

A 300-incident, 1,546-report Sri Lankan benchmark spans Sinhala, Tamil, English, Singlish, and Tanglish over call and SMS styles, and is constructed to include same-landmark hard negatives, deliberately unequal evidence across reports, and noisy low-evidence reports. A chronological history constructor admits only reports arriving strictly before the query, yielding 4,085 fixed candidate-incident rows that both compared methods see identically — a common evaluation set that isolates the effect of the evidence unit from any difference in available information.

A jointly fine-tuned mDeBERTa-v3 incident-history matcher scores a query report against the concatenated accumulated history. It is compared against an independently fine-tuned pairwise control that receives the same historical reports but scores them individually with late aggregation.

### C. Results and operationalization

The joint model improves incident-level AUROC on every test split, with the largest and statistically supported gain on the geographically confusable split: 0.8442 to 0.9638, an increase of 0.1195 with a 95% confidence interval of 0.0540 to 0.1894 (Table IV). Ablations localise the advantage to multi-report histories and same-landmark negatives, confirming that the benefit arises precisely where fragmented evidence and misleading lexical similarity coincide, rather than uniformly across the benchmark. A calibration extension specified in advance was rejected on validation and is reported as such.

**TABLE IV. Incident-level AUROC by test split**

| Split | Pairwise control | Joint history |
|---|---|---|
| Geographically confusable | 0.8442 | **0.9638** |

The frozen matcher is then operationalized through a learned cost-sensitive policy that converts candidate evidence into a merge, separate, or defer action without hand-written semantic rules, and evaluated end to end against an incident store the system builds itself rather than a gold-constructed one. This exposes a safety–coverage trade-off invisible under gold histories, because earlier decisions change the candidate set available to later ones: the false-merge rate falls to 0.0224 and cluster purity remains between 0.967 and 0.990, at the cost of deferring 61.3% of queries to human adjudication. The deployed pipeline's decision path was verified to reproduce the evaluated system exactly across the full benchmark.

## VIII. Discussion and Limitations

**Recognition is the binding constraint.** The urgency component's external validation exposes a failure that the other components do not encounter, because they were evaluated on simulated audio or on text. On authentic telephony recordings the Sinhala recognition model produces in-script but semantically wrong transcripts, silently disabling 20 of 111 urgency features. The dialogue component's recognition stage employs a domain-fine-tuned model with multi-engine candidate scoring, and integrating it in place of the urgency component's own recognition is the highest-value remaining work — it would restore the lexical features and remove a redundant transcription pass simultaneously.

**External validation is uneven across components.** Only the urgency component has been evaluated on authentic emergency recordings. The other three are validated on purpose-built simulated corpora, so their reported figures are development estimates in the same sense that 0.621 is, and should not be read as generalization guarantees. The gap between simulated and authentic performance observed for the urgency component is a caution that applies to all four.

**Known residual weaknesses.** Location extraction is weak in both the SMS component (40% recall) and the dialogue component, in both cases traceable to sparse location annotation rather than modelling failure. The MPDS severity check retains one adversarial under-triage case. The correlation policy defers 61.3% of queries, trading coverage for a low false-merge rate. Diarization on 8 kHz telephony audio failed on four recordings and could not be recovered by resampling, since the information lost above 4 kHz carries much of the speaker-discriminative signal. Medium urgency remains the weakest class throughout, at F1 0.463, reflecting both its position between two adjacent classes and a definitional gap: DMC Medium calls include welfare and shelter requests absent from the simulated corpus.

**Deployment considerations.** Per-turn urgency inference on CPU is dominated by recognition cost, which is prohibitive if performed independently; consuming the dialogue component's existing transcript reduces it to acoustic feature extraction and makes synchronous operation feasible. Concurrent multi-victim SMS traffic is not yet supported.

## IX. Conclusion

We have presented an integrated multilingual emergency intake pipeline for Sri Lanka spanning text and voice channels, comprising hybrid LLM-guarded SMS triage with protocol-based priority assignment, a closed-loop dialogue agent with an auditable clarification policy, a multimodal urgency classifier validated externally on authentic Disaster Management Centre recordings, and a joint incident-history matcher for cross-channel correlation. Each component confines its use of generative or learned inference to points where deterministic methods are demonstrably insufficient, and each is evaluated against a baseline rather than reported in isolation.

The external validation of the urgency component on real recordings, and the recognition failure it exposed, are reported as the most informative result in this work: they identify the specific engineering dependency on which deployment performance rests. Immediate future work is the substitution of the domain-fine-tuned recognition stage across components, followed by re-validation on the authentic corpus.

## Acknowledgment

The authors thank the Sri Lanka Disaster Management Centre for providing authentic emergency call recordings for external validation, and the supervising faculty of the Faculty of Computing, SLIIT, for architectural review.

## References

[1] A. Conneau et al., "Unsupervised cross-lingual representation learning at scale," in *Proc. 58th Annu. Meeting Assoc. Comput. Linguistics*, 2020, pp. 8440–8451.

[2] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust speech recognition via large-scale weak supervision," in *Proc. 40th Int. Conf. Mach. Learn.*, 2023, pp. 28492–28518.

[3] T. Solorio et al., "Overview for the first shared task on language identification in code-switched data," in *Proc. 1st Workshop Comput. Approaches Code Switching*, 2014, pp. 62–72.

[4] B. Schuller, A. Batliner, S. Steidl, and D. Seppi, "Recognising realistic emotions and affect in speech: State of the art and lessons learnt from the first challenge," *Speech Commun.*, vol. 53, no. 9–10, pp. 1062–1087, 2011.

[5] J. Lafferty, A. McCallum, and F. C. N. Pereira, "Conditional random fields: Probabilistic models for segmenting and labeling sequence data," in *Proc. 18th Int. Conf. Mach. Learn.*, 2001, pp. 282–289.

[6] G. Lample, M. Ballesteros, S. Subramanian, K. Kawakami, and C. Dyer, "Neural architectures for named entity recognition," in *Proc. NAACL-HLT*, 2016, pp. 260–270.

[7] M. Henderson, B. Thomson, and J. D. Williams, "The second dialog state tracking challenge," in *Proc. 15th Annu. Meeting SIGDIAL*, 2014, pp. 263–272.

[8] M. Aliannejadi, H. Zamani, F. Crestani, and W. B. Croft, "Asking clarifying questions in open-domain information-seeking conversations," in *Proc. 42nd Int. ACM SIGIR Conf.*, 2019, pp. 475–484.

[9] International Academies of Emergency Dispatch, "Medical Priority Dispatch System (MPDS)," Salt Lake City, UT, USA.

[10] S. M. Bossaert and P. Vandenberghe, "Simple Triage and Rapid Treatment (START)," in *Disaster Medicine*, 2nd ed., 2011.

[11] MCECR: a multilingual cross-document event coreference dataset, in *Proc. LREC*, 2022.

[12] L. Xue et al., "mT5: A massively multilingual pre-trained text-to-text transformer," in *Proc. NAACL-HLT*, 2021, pp. 483–498.

[13] H. Bredin, "pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe," in *Proc. INTERSPEECH*, 2023.

[14] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *J. Artif. Intell. Res.*, vol. 16, pp. 321–357, 2002.

[15] A. Williams, N. Nangia, and S. Bowman, "A broad-coverage challenge corpus for sentence understanding through inference," in *Proc. NAACL-HLT*, 2018, pp. 1112–1122.
