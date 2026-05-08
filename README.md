# Emergency Call Urgency Detection System

Classifies Sri Lankan emergency calls (Sinhala / Tamil / English) into
**High / Medium / Low** urgency using acoustic + textual features.

## Folder structure

```
emergency_urgency/
├── audio/
│   ├── *.ogg          ← raw audio files
│   └── clean/         ← preprocessed .wav (auto-generated)
├── labels.csv         ← 100 labelled samples
├── features.csv       ← extracted feature matrix (auto-generated)
├── models/            ← saved model artefacts (auto-generated)
├── reports/           ← plots + classification report (auto-generated)
└── scripts/
    ├── 01_preprocess.py
    ├── 02_extract_features.py
    ├── 03_train.py
    ├── 04_evaluate.py
    └── 05_predict.py
```

## Quick start

```bash
# 1. Install dependencies
pip install librosa soundfile noisereduce pandas numpy scikit-learn \
            imbalanced-learn openai-whisper matplotlib seaborn joblib

# 2. Place .ogg files in audio/ and labels.csv in root

# From the scripts/ folder:
cd scripts

# 3. Preprocess audio
python 01_preprocess.py

# 4. Extract features (runs Whisper STT — takes ~10-20 min for 100 files)
python 02_extract_features.py

# 5. Train classifier
python 03_train.py

# 6. Evaluate & generate plots
python 04_evaluate.py

# 7. Classify a single call (demo)
python 05_predict.py ../audio/sample_call.ogg
```

## Pipeline overview

```
Raw .ogg → Noise reduction → Resample 16 kHz → Trim silence
                ↓
     ┌──────────┴──────────┐
  Acoustic              Textual
  (MFCCs, pitch,        (Whisper STT →
   RMS, ZCR, …)         keywords, sentiment)
     └──────────┬──────────┘
            Early fusion
                ↓
       Random Forest (200 trees)
       + SMOTE for imbalance
                ↓
       High / Medium / Low
```
