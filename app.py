"""
app.py
Flask web interface — upload an audio file, get urgency classification.
Run: python app.py
Then open: http://localhost:5000
"""

import os
import tempfile
import numpy as np
import librosa
import noisereduce as nr
import joblib
import whisper
from collections import Counter
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Load models once at startup ───────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

print("Loading model...")
pipeline     = joblib.load(os.path.join(MODELS_DIR, "rf_urgency.pkl"))
le           = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))

print("Loading Whisper...")
whisper_model = whisper.load_model("base")
print("Ready.")

TARGET_SR = 16_000
N_MFCC    = 40

URGENCY_KEYWORDS = [
    # ── English — confirmed from actual Whisper transcripts ───────────────────
    "help", "fire", "inside", "trapped", "quickly", "emergency",
    "blood", "dying", "dead", "accident", "attack", "hurt", "injured",
    "crash", "ambulance", "police", "gas", "smoke", "burning", "flood",
    "workers", "grandmother", "children", "baby", "unconscious",
    "breathing", "cannot breathe", "stuck", "still inside",

    # ── Sinhala — Whisper hallucinates on Sinhala audio (outputs Arabic/Cyrillic)
    # Only romanised fragments that actually appeared in transcripts are kept
    "api",       # Whisper sometimes outputs this for Sinhala "api" (we)
    "na",        # negative particle — appears in low urgency Sinhala
    "rush",      # Singlish urgency marker confirmed in call7
    "inna",      # Sinhala "inna" (is/are) — appeared in call14

    # ── Tamil — confirmed from actual Tamil script Whisper output ─────────────
    "உதவி",              # help
    "உதவி செய்யுங்கள்", # please help us
    "உதவுங்கள்",         # help (imperative)
    "முக்க",             # can't breathe / face (suffocation) — confirmed call21
    "உள்ளே",             # inside / trapped inside — confirmed call23
    "நாங்கள்",           # we — group in distress — confirmed call23
    "வெள்ளம்",           # flood
    "வெள்ளனிர்",         # floodwater — confirmed call23
    "தீ",                # fire
    "இரத்தம்",           # blood
    "விபத்து",           # accident
    "சிறிக்கிறோம்",      # we are trapped — confirmed call23
    "தெரிய வில்லை",      # cannot see/don't know — distress marker call21
    "சரியாக",            # properly (as in "not working properly") — call21
]

NON_FEATURE_COLS = {"filename", "urgency_label", "language", "transcript"}

URGENCY_COLORS = {
    "High":   {"color": "#e74c3c", "emoji": "🔴", "message": "Immediate response required"},
    "Medium": {"color": "#f39c12", "emoji": "🟡", "message": "Monitor and prepare to respond"},
    "Low":    {"color": "#2ecc71", "emoji": "🟢", "message": "No immediate threat"},
}


# ── Audio processing functions ────────────────────────────────────────────────

def preprocess(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    y = nr.reduce_noise(y=y, sr=sr)
    y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    y, _ = librosa.effects.trim(y, top_db=20)
    return y


def extract_acoustic(y):
    feats = {}
    mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1}_std"]  = float(np.std(mfcc[i]))

    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=400, sr=TARGET_SR)
    voiced_f0 = f0[voiced] if voiced is not None and voiced.any() else np.array([0.0])
    feats["pitch_mean"] = float(np.nanmean(voiced_f0))
    feats["pitch_std"]  = float(np.nanstd(voiced_f0))

    rms = librosa.feature.rms(y=y)[0]
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"]  = float(np.std(rms))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats["zcr_mean"] = float(np.mean(zcr))

    sc = librosa.feature.spectral_centroid(y=y, sr=TARGET_SR)[0]
    feats["spec_centroid_mean"] = float(np.mean(sc))

    hop = 512
    voiced_frames = int(np.sum(voiced)) if voiced is not None else 0
    feats["speaking_rate"] = voiced_frames / max(len(y) / hop, 1)
    feats["jitter"] = float(np.mean(np.abs(np.diff(voiced_f0)))) if voiced_f0.size > 1 else 0.0
    return feats


def extract_textual(audio: np.ndarray):
    result = whisper_model.transcribe(audio.astype(np.float32), task="transcribe")
    text   = result.get("text", "").lower()
    words  = text.split()
    feats  = {}

    feats["keyword_count"] = sum(kw.lower() in text for kw in URGENCY_KEYWORDS)
    feats["word_count"]    = len(words)

    counts   = Counter(words)
    repeated = sum(1 for w, c in counts.items() if c > 1)
    feats["repetition_rate"] = repeated / max(len(counts), 1)

    positive = {"safe", "okay", "fine", "calm", "alright"}
    negative = {"help", "hurt", "dying", "trapped", "fire", "blood",
                "attack", "crash", "emergency", "pain", "dead"}
    pos = sum(1 for w in words if w in positive)
    neg = sum(1 for w in words if w in negative)
    feats["sentiment_polarity"] = (pos - neg) / max(len(words), 1)
    feats["transcript"] = result.get("text", "")
    return feats


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    suffix = os.path.splitext(file.filename)[1] or ".ogg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        file.save(tmp_in.name)
        tmp_in_path = tmp_in.name

    try:
        y = preprocess(tmp_in_path)

        acoustic = extract_acoustic(y)
        textual  = extract_textual(y)

        all_feats = {**acoustic, **textual}
        X = np.array([all_feats.get(col, 0.0) for col in feature_cols]).reshape(1, -1)

        pred_enc   = pipeline.predict(X)[0]
        pred_label = le.inverse_transform([pred_enc])[0]
        proba      = pipeline.predict_proba(X)[0]
        class_proba = {cls: round(float(p) * 100, 1)
                       for cls, p in zip(le.classes_, proba)}

        info = URGENCY_COLORS[pred_label]

        return jsonify({
            "urgency":     pred_label,
            "color":       info["color"],
            "emoji":       info["emoji"],
            "message":     info["message"],
            "transcript":  textual["transcript"],
            "keywords":    int(textual["keyword_count"]),
            "confidence":  class_proba,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        os.unlink(tmp_in_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
