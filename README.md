# Emergency Call Urgency Detection System
### R26_DS_011 — PP1 Prototype

Multimodal urgency classification of Sri Lankan emergency calls (Sinhala / Tamil / Singlish / Tanglish / English) into **High / Medium / Low** urgency using prosodic and conversational features.

---

## Folder Structure

```
R26_DS_011/
├── audio/
│   ├── *.ogg                  ← raw labelled call recordings
│   └── clean/                 ← preprocessed .wav files (auto-generated)
├── labels.csv                 ← 96 labelled samples
├── features.csv               ← extracted feature matrix (auto-generated)
├── models/                    ← saved model artefacts (auto-generated)
├── reports/                   ← evaluation plots and CSVs (auto-generated)
├── templates/index.html       ← web demo UI
├── app.py                     ← Flask web interface
├── requirements.txt
└── scripts/
    ├── 01_preprocess.py       ← noise reduction, resample, trim
    ├── 02_extract_features.py ← acoustic + Whisper textual features
    ├── 03_train.py            ← Random Forest + SMOTE training
    ├── 04_evaluate.py         ← confusion matrix, F1, feature importance
    ├── 05_predict.py          ← single-file command line prediction
    ├── 06_compare_models.py   ← RF vs SVM vs Logistic Regression
    ├── 07_shap_analysis.py    ← SHAP explainability plots
    ├── 08_language_analysis.py← per-language F1 breakdown
    └── 09_keyword_discovery.py← TF-IDF data-driven keyword analysis
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running the Full Pipeline

Run all scripts from the `scripts/` folder:

```bash
cd scripts

# Step 1 — Preprocess audio
python 01_preprocess.py

# Step 2 — Extract features (takes ~20 min for 96 files)
python 02_extract_features.py

# Step 3 — Train classifier
python 03_train.py

# Step 4 — Evaluate and generate plots
python 04_evaluate.py

# Step 5 — Predict a single call
python 05_predict.py ../audio/call9.ogg
```

## Advanced Analysis

```bash
python 06_compare_models.py    # RF vs SVM vs Logistic Regression
python 07_shap_analysis.py     # SHAP feature explainability
python 08_language_analysis.py # Per-language F1 breakdown
python 09_keyword_discovery.py # TF-IDF keyword discovery
```

## Web Demo

```bash
python app.py
# Open http://localhost:5000
```

Upload any audio file to get urgency classification with confidence scores and Whisper transcript.

---

## System Architecture

```
Raw .ogg call
      │
      ▼
  Preprocessing
  (noise reduction → 16kHz resample → silence trim)
      │
      ├─────────────────────┬─────────────────────┐
      │                     │                     │
  Acoustic Branch       Linguistic Branch         │
  88 features           4 features                │
  MFCCs (×40)           Whisper STT               │
  Pitch / F0            Keyword count             │
  RMS energy            Word count                │
  ZCR                   Repetition rate           │
  Spectral centroid     Sentiment polarity        │
  Speaking rate                                   │
  Jitter                                          │
      │                     │                     │
      └─────────────────────┘                     │
              92 features combined                │
                    │                             │
                    ▼                             │
      Random Forest (200 trees)                   │
      StandardScaler + SMOTE                      │
      5-fold stratified CV                        │
                    │                             │
                    ▼                             │
        High / Medium / Low ─────────────────────┘
```

---

## Results

| Metric | Value |
|---|---|
| Macro F1 (5-fold CV) | 0.34 ± 0.10 |
| High urgency F1 | 0.56 |
| Medium urgency F1 | 0.37 |
| Low urgency F1 | 0.09 |
| Best model | Random Forest (vs SVM, Logistic Regression) |

**Key finding:** Whisper base does not support Sinhala — outputs hallucinated script. Acoustic branch classifies Sinhala calls independently without transcription. Tamil and English benefit from full multimodal classification.

---

## PP2 Roadmap

| Limitation | Plan |
|---|---|
| Whisper fails on Sinhala | Facebook MMS — supports Sinhala natively |
| Small dataset (96 samples) | Auto-label DMC call data, target 500+ samples |
| Keyword counting is brittle | Multilingual sentence embeddings |
| No real-time input | WebSocket-based live stream classification |
