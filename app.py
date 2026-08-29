"""
app.py
Standalone Flask web interface - upload an audio file, get urgency
classification. Run: python app.py, then open http://localhost:5000

Uses the same validated pipeline as everything else in this project
(integration/urgency_bridge.py: preprocess_like_training + frozen
02_extract_features.py acoustic extraction + context_features.py Sinhala/
Tamil scoring + the cached frozen v1 model) rather than a separate
reimplementation, so a standalone demo run here matches every other result
reported for this system. Whisper transcription happens here because this
app has no upstream component supplying a transcript already.
"""

import os
import sys
import tempfile

import numpy as np
import librosa
import whisper
from flask import Flask, request, jsonify, render_template

_HERE = os.path.dirname(os.path.abspath(__file__))
_INTEGRATION_DIR = os.path.join(_HERE, "integration")
if _INTEGRATION_DIR not in sys.path:
    sys.path.insert(0, _INTEGRATION_DIR)
from urgency_bridge import classify_urgency_late_fusion, warmup  # noqa: E402

app = Flask(__name__)

TARGET_SR = 16_000

print("Loading Whisper...")
whisper_model = whisper.load_model("base")
print("Warming up urgency model...")
warmup()
print("Ready.")

URGENCY_COLORS = {
    "High":   {"color": "#e74c3c", "emoji": "🔴", "message": "Immediate response required"},
    "Medium": {"color": "#f39c12", "emoji": "🟡", "message": "Monitor and prepare to respond"},
    "Low":    {"color": "#2ecc71", "emoji": "🟢", "message": "No immediate threat"},
}


@app.route("/favicon.ico")
def favicon():
    return "", 204


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
        y, _ = librosa.load(tmp_in_path, sr=TARGET_SR, mono=True)

        whisper_result = whisper_model.transcribe(y.astype(np.float32), task="transcribe")
        transcript = whisper_result.get("text", "")
        language = whisper_result.get("language", "en")

        result = classify_urgency_late_fusion(audio=y, transcript=transcript, language=language)

        if result.get("skipped"):
            return jsonify({"error": result.get("reason", "Could not classify this audio")}), 400

        pred_label = result["label"]
        info = URGENCY_COLORS[pred_label]
        confidence_pct = {cls: round(p * 100, 1) for cls, p in result["probabilities"].items()}

        return jsonify({
            "urgency":    pred_label,
            "color":      info["color"],
            "emoji":      info["emoji"],
            "message":    info["message"],
            "transcript": transcript,
            "keywords":   int(round(result.get("keyword_count", 0))),
            "confidence": confidence_pct,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        os.unlink(tmp_in_path)


if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=False)
