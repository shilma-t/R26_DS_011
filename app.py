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

LOCKED MODEL (2026-08-30): late fusion (0.5/0.5), Sinhala transcribed with
the multilingual ASR used by the live call-handling integration (not this
project's own Sinhala fine-tune), Tamil/English via base Whisper. Chosen
over the project's own-ASR configuration because it improves on ALL three
evaluation views, not just one -- simulated CV acc 66.4%/F1 0.639 (vs
63.9%/0.613), DMC n=11 scope-matched acc 80.0%/F1 0.753 (vs 67.3%/0.630),
DMC n=59 full acc 44.8%/F1 0.431 (vs 43.7%/0.428) -- see
scripts/analysis/04_domain_shift_and_scope/asr_source_comparison_*.py.
That's a materially different result from every other tuning attempt this
project tried (weight/threshold/HP tuning, confidence gating, a duration
safeguard, max-confidence-wins), all of which failed on simulated CV alone
and were rejected; this is the first candidate that actually clears that
bar on real data, not just DMC.
"""

import os
import subprocess
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

# Locked Sinhala ASR: the multilingual ASR used by the live call-handling
# integration (faster-whisper/CTranslate2 format) -- see module docstring for
# why this was chosen over this project's own Sinhala fine-tune.
SINHALA_MODEL_ID = "SPEAK-ASR/faster-whisper-medium-si-exp10-fp16"
SINHALA_ASR_BEAM_SIZE = 5
SINHALA_ASR_PROMPT = (
    "Sri Lankan emergency phone call. Sinhala, Tamil, or English. "
    "Possible topics: fire, flood, smoke, trapped people, injury, address."
)

print("Loading Whisper...")
whisper_model = whisper.load_model("base")
print("Warming up urgency model...")
warmup()
print("Ready.")

_sinhala_asr = None


def get_sinhala_asr():
    """Locked Sinhala ASR (see module docstring) -- lazy-loaded so a call that
    never needs it (English/Tamil) never pays the load cost."""
    global _sinhala_asr
    if _sinhala_asr is None:
        from faster_whisper import WhisperModel
        print("  Loading Sinhala ASR (" + SINHALA_MODEL_ID + ") ...")
        _sinhala_asr = WhisperModel(SINHALA_MODEL_ID, device="cpu", compute_type="int8")
    return _sinhala_asr


def normalize_audio_for_stt(input_path: str) -> str:
    """Same STT-only preprocessing the integrated pipeline applies before
    transcription (services/audio_utils.py normalize_audio_to_wav): mono 16kHz
    WAV, a light high-pass to cut rumble, and gentle loudness normalization.
    Confirmed live that skipping this step is enough on its own to make the
    Sinhala ASR produce a meaningfully different transcript for the same
    audio -- this is NOT applied to the audio used for urgency's acoustic
    features (that path -- classify_urgency_late_fusion below -- gets the
    plain, unfiltered audio, exactly matching the integrated pipeline's own
    separate, plainer conversion for that path, convert_to_wav_16k)."""
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    filter_chain = "highpass=f=80,loudnorm=I=-23:TP=-2:LRA=11"
    cmd_with_filters = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000",
                        "-af", filter_chain, out_path]
    cmd_plain = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", out_path]
    try:
        subprocess.run(cmd_with_filters, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        subprocess.run(cmd_plain, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

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

    stt_wav_path = None
    try:
        # Plain audio for urgency's acoustic features -- matches the
        # integrated pipeline's own separate, unfiltered conversion for
        # this path (convert_to_wav_16k). Do NOT apply STT normalization
        # here; classify_urgency_late_fusion does its own preprocessing
        # internally, matched to how the training data was prepared.
        y, _ = librosa.load(tmp_in_path, sr=TARGET_SR, mono=True)

        # Separately-normalized audio for STT only (see normalize_audio_for_stt).
        stt_wav_path = normalize_audio_for_stt(tmp_in_path)
        y_stt, _ = librosa.load(stt_wav_path, sr=TARGET_SR, mono=True)

        whisper_result = whisper_model.transcribe(y_stt.astype(np.float32), task="transcribe")
        transcript = whisper_result.get("text", "")
        language = whisper_result.get("language", "en")

        # Base Whisper detected Sinhala -- re-transcribe with the locked Sinhala
        # ASR (see module docstring). English/Tamil keep the base Whisper result.
        # Parameters mirror whisper_engine.py's transcribe_with_forced_language()
        # exactly, so this matches the integrated pipeline's own decode behavior,
        # not just its model choice.
        if language == "si":
            sinhala_model = get_sinhala_asr()
            segments, _info = sinhala_model.transcribe(
                y_stt.astype(np.float32),
                language="si",
                task="transcribe",
                beam_size=SINHALA_ASR_BEAM_SIZE,
                best_of=SINHALA_ASR_BEAM_SIZE,
                patience=1.0,
                vad_filter=True,
                condition_on_previous_text=False,
                temperature=0,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                hallucination_silence_threshold=1.0,
                initial_prompt=SINHALA_ASR_PROMPT,
            )
            sinhala_text = " ".join(seg.text.strip() for seg in segments).strip()
            transcript = sinhala_text or transcript

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
        if stt_wav_path and os.path.exists(stt_wav_path):
            os.unlink(stt_wav_path)


if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=False)
