"""
urgency_bridge.py
Entry point for Component 2 (adaptive clarification dialogue) to obtain an
urgency classification for a caller turn.

Contract
--------
    from urgency_bridge import classify_urgency, end_call

    urgency = classify_urgency(
        audio="audio/live_mic/ab12cd_t1.wav",   # path or numpy array
        transcript="ගෙදර ගිනිගන්නවා, ළමයා ඇතුළේ",
        language="si-LK",                        # Component 2's code
        session_id=session.session_id,           # optional, enables aggregation
    )
    # -> {"label": "High", "confidence": 0.78, "probabilities": {...}, ...}

    final = end_call(session.session_id)
    # -> aggregated call-level urgency, ready for the final emergency JSON

Design notes
------------
NO ASR happens here. Component 2 has already transcribed the turn with its
fine-tuned faster-whisper model; re-transcribing would cost ~85 s per turn on
CPU and make live operation impossible. The transcript is an input.

Acoustic features come from the caller's turn audio via the frozen
extract_acoustic(); text and context features from the frozen lexicons via
text_features_from_transcript(). The 111-feature vector, the model, the
scaler and the prediction settings are exactly those of the frozen pipeline.

Aggregation across turns uses the mean probability vector, fixed before any
result was seen. Peak severity is reported alongside because under-triage is
the dangerous failure direction in dispatch.
"""

from __future__ import annotations

import os
import sys
import time
import threading
import importlib.util

import numpy as np

# ── locate the urgency project ───────────────────────────────────────────────
URGENCY_ROOT = os.environ.get(
    "URGENCY_ROOT",
    r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011",
)
PIPELINE_DIR = os.path.join(URGENCY_ROOT, "scripts", "pipeline")
MODEL_PATH = os.path.join(URGENCY_ROOT, "models", "urgency_rf_v1.joblib")
FEATURES_CSV = os.path.join(URGENCY_ROOT, "data", "features.csv")

if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

from text_features_from_transcript import (  # noqa: E402
    fire_smoke_match,
    normalise_language,
    text_features_from_transcript,
)

TARGET_SR = 16_000
SEED = 42
RF_PARAMS = dict(n_estimators=200, min_samples_leaf=2,
                 class_weight="balanced", random_state=SEED, n_jobs=-1)

# Frozen 111-feature order. Must match candidate_pipeline_freeze.txt.
ACOUSTIC = (
    [f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")] +
    ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
     "spec_centroid_mean", "speaking_rate", "jitter",
     "rms_slope", "rms_gradient", "speaking_rate_change"]
)
TEXT = ["asr_quality", "keyword_count", "word_count",
        "repetition_rate", "sentiment_polarity"]
CONTEXT = [
    "sin_context_score", "sin_danger_count", "sin_help_request_count",
    "sin_high_risk_phrase_count", "sin_location_context_count",
    "sin_person_at_risk_count", "sin_safety_state_count",
    "tam_context_score", "tam_danger_count", "tam_help_request_count",
    "tam_high_risk_phrase_count", "tam_location_context_count",
    "tam_person_at_risk_count", "tam_safety_state_count",
]
FEATURE_COLS = ACOUSTIC + TEXT + CONTEXT + ["fire_smoke_match"]

SEVERITY = {"Low": 0, "Medium": 1, "High": 2}

# Late-fusion pair: text-only (Condition C - word_count dropped, found to be a
# near-noise, heavily domain-shifted feature causing a Medium-class collapse
# on real DMC calls) + acoustic-only, combined 0.5/0.5. Validated on the
# flood/fire-scoped DMC subset (n=11) at 67.3% accuracy / 65.7% High recall,
# vs the single fused FEATURE_COLS model's 47.3%/31.4% on the same subset.
# See reports/pp2_finalized_reports/SUMMARY.md.
TEXT_ONLY_COLS = ["asr_quality", "keyword_count", "repetition_rate",
                  "sentiment_polarity"] + CONTEXT + ["fire_smoke_match"]
ACOUSTIC_ONLY_COLS = ACOUSTIC
TEXT_MODEL_PATH = os.path.join(URGENCY_ROOT, "models", "urgency_text_only_v1.joblib")
ACOUSTIC_MODEL_PATH = os.path.join(URGENCY_ROOT, "models", "urgency_acoustic_only_v1.joblib")

_lock = threading.Lock()
_model = None
_labels = None
_text_model = None
_acoustic_model = None
_frozen = None
_sessions: dict[str, list[dict]] = {}
_pending: dict[str, list[threading.Thread]] = {}


def _frozen_module():
    """Import 02_extract_features.py for extract_acoustic()."""
    global _frozen
    if _frozen is None:
        path = os.path.join(PIPELINE_DIR, "02_extract_features.py")
        spec = importlib.util.spec_from_file_location("frozen_features", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _frozen = module
    return _frozen


def _build_model():
    """
    Train the frozen configuration on the 267-call simulated corpus.

    Trained once and cached to models/urgency_rf_v1.joblib. That filename is
    new: no v1 artifact is overwritten. fire_smoke_match is absent from
    features.csv and is recomputed from the stored transcripts, exactly as
    evaluate_dmc.py does, so training and inference agree.
    """
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    if os.path.exists(MODEL_PATH):
        payload = joblib.load(MODEL_PATH)
        return payload["model"], payload["classes"]

    if not os.path.exists(FEATURES_CSV):
        raise SystemExit(f"Training features not found: {FEATURES_CSV}")

    sim = pd.read_csv(FEATURES_CSV)
    if "fire_smoke_match" not in sim.columns:
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)

    missing = [c for c in FEATURE_COLS if c not in sim.columns]
    if missing:
        raise SystemExit(f"features.csv missing frozen columns: {missing}")

    sim = sim.dropna(subset=["urgency_label"])
    encoder = LabelEncoder().fit(["High", "Low", "Medium"])

    model = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=SEED)),
        ("clf", RandomForestClassifier(**RF_PARAMS)),
    ])
    model.fit(sim[FEATURE_COLS].fillna(0).values,
              encoder.transform(sim["urgency_label"].values))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model,
                 "classes": list(encoder.classes_),
                 "feature_cols": FEATURE_COLS,
                 "n_train": len(sim)}, MODEL_PATH)
    return model, list(encoder.classes_)


def _ensure_model():
    global _model, _labels
    with _lock:
        if _model is None:
            _model, _labels = _build_model()
    return _model, _labels


def _build_single_model(feature_cols, cache_path):
    """Same training recipe as _build_model(), for a smaller feature subset."""
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    if os.path.exists(cache_path):
        payload = joblib.load(cache_path)
        return payload["model"], payload["classes"]

    if not os.path.exists(FEATURES_CSV):
        raise SystemExit(f"Training features not found: {FEATURES_CSV}")

    sim = pd.read_csv(FEATURES_CSV)
    if "fire_smoke_match" not in sim.columns:
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)

    missing = [c for c in feature_cols if c not in sim.columns]
    if missing:
        raise SystemExit(f"features.csv missing columns: {missing}")

    sim = sim.dropna(subset=["urgency_label"])
    encoder = LabelEncoder().fit(["High", "Low", "Medium"])

    model = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=SEED)),
        ("clf", RandomForestClassifier(**RF_PARAMS)),
    ])
    model.fit(sim[feature_cols].fillna(0).values,
              encoder.transform(sim["urgency_label"].values))

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    joblib.dump({"model": model, "classes": list(encoder.classes_),
                 "feature_cols": feature_cols, "n_train": len(sim)}, cache_path)
    return model, list(encoder.classes_)


def _ensure_late_fusion_models():
    global _text_model, _acoustic_model, _labels
    with _lock:
        if _text_model is None:
            _text_model, labels = _build_single_model(TEXT_ONLY_COLS, TEXT_MODEL_PATH)
            _labels = _labels or labels
        if _acoustic_model is None:
            _acoustic_model, labels = _build_single_model(ACOUSTIC_ONLY_COLS, ACOUSTIC_MODEL_PATH)
            _labels = _labels or labels
    return _text_model, _acoustic_model, _labels


def warmup() -> None:
    """
    Load the model and feature code before the first call.

    Worth invoking at demo start: the first classify_urgency() otherwise pays
    model construction and librosa import cost mid-conversation.
    """
    _ensure_model()
    _ensure_late_fusion_models()
    _frozen_module()


def _load_audio(audio) -> np.ndarray:
    """Accept a path or an array; return mono float32 at TARGET_SR."""
    if isinstance(audio, (str, os.PathLike)):
        import librosa
        y, _ = librosa.load(str(audio), sr=TARGET_SR)
        return y

    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y


def preprocess_like_training(y, sr=TARGET_SR):
    """
    Denoise and silence-trim, exactly as 01_preprocess.py did to the training
    corpus before features were extracted from it.

    NOT OPTIONAL. The model learned its thresholds on audio that had been
    through these steps. Feeding raw microphone audio straight in inflates
    rms_mean roughly fourfold and depresses every MFCC standard deviation,
    which pushes predictions toward High and Medium and effectively removes
    Low from the model's range. This was measured on the DMC corpus, where
    skipping it produced a macro-F1 of 0.2524.
    """
    import librosa
    import noisereduce as nr

    y_nr = nr.reduce_noise(y=y, sr=sr)
    if sr != TARGET_SR:
        y_nr = librosa.resample(y_nr, orig_sr=sr, target_sr=TARGET_SR)
    y_trimmed, _ = librosa.effects.trim(y_nr, top_db=20)

    if y_trimmed.size < TARGET_SR // 2:
        return y_nr
    return y_trimmed


def classify_urgency(audio, transcript, language, session_id=None,
                     min_seconds=1.0) -> dict:
    """
    Classify one caller turn.

    audio       path to the turn WAV, or a numpy array at 16 kHz
    transcript  text already produced by Component 2's ASR
    language    Component 2's code ('si-LK', 'ta-IN', 'en-US', ...)
    session_id  when given, the turn joins that call's running aggregate

    Returns label, confidence, per-class probabilities, and diagnostics.
    A turn shorter than min_seconds is skipped: 91 of the 111 features are
    acoustic and their statistics are unreliable below about a second.
    """
    model, classes = _ensure_model()
    frozen = _frozen_module()

    y = _load_audio(audio)
    seconds = len(y) / TARGET_SR

    if seconds < min_seconds:
        return {
            "label": None,
            "confidence": None,
            "probabilities": {},
            "skipped": True,
            "reason": f"turn too short ({seconds:.2f}s < {min_seconds}s)",
            "turn_seconds": round(seconds, 2),
        }

    # Match the training corpus before extracting anything
    y = preprocess_like_training(y, TARGET_SR)
    if y.size < TARGET_SR // 2:
        return {
            "label": None,
            "confidence": None,
            "probabilities": {},
            "skipped": True,
            "reason": "nothing left after denoise and trim (silence?)",
            "turn_seconds": round(seconds, 2),
        }

    acoustic = frozen.extract_acoustic(y, TARGET_SR)
    textual = text_features_from_transcript(transcript, language)

    row = {**acoustic, **textual}
    vector = np.array([[float(row.get(c, 0.0) or 0.0) for c in FEATURE_COLS]])

    proba = model.predict_proba(vector)[0]
    index = int(proba.argmax())

    result = {
        "label": classes[index],
        "confidence": round(float(proba[index]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
        "skipped": False,
        "turn_seconds": round(seconds, 2),
        "language_normalised": textual["language_normalised"],
        "keyword_count": round(float(textual["keyword_count"]), 2),
        "asr_quality": round(float(textual["asr_quality"]), 3),
        "context_score": int(textual.get("sin_context_score", 0)
                             or textual.get("tam_context_score", 0) or 0),
    }

    print(
        "[urgency] turn "
        + str(len(_sessions.get(str(session_id), [])) + 1)
        + "  lang=" + str(textual["language_normalised"])
        + "  sec=" + str(round(seconds, 1))
        + "  kw=" + str(round(float(textual["keyword_count"]), 2))
        + "  ctx=" + str(result["context_score"])
        + "  asrq=" + str(round(float(textual["asr_quality"]), 2))
        + "  ->  " + str(result["label"])
        + " (H=" + str(result["probabilities"].get(classes[0], 0))
        + " L=" + str(result["probabilities"].get(classes[1], 0))
        + " M=" + str(result["probabilities"].get(classes[2], 0)) + ")"
        + "  transcript=" + repr(transcript[:60])
    )

    if session_id is not None:
        with _lock:
            _sessions.setdefault(str(session_id), []).append(result)
        result["turn_index"] = len(_sessions[str(session_id)])

    return result


def classify_urgency_late_fusion(audio, transcript, language, session_id=None,
                                 min_seconds=1.0) -> dict:
    """
    Same contract as classify_urgency(), but combines a text-only model
    (Condition C) and an acoustic-only model via unweighted 0.5/0.5 late
    fusion instead of one RF trained on all 111 concatenated features.

    Validated as the strongest candidate on the flood/fire-scoped DMC
    subset (67.3% accuracy / 65.7% High recall vs the single fused model's
    47.3%/31.4% on the same calls) - see
    reports/pp2_finalized_reports/SUMMARY.md section 4. Does NOT replace or
    modify urgency_rf_v1.joblib; caches two new model files alongside it.
    """
    text_model, acoustic_model, classes = _ensure_late_fusion_models()
    frozen = _frozen_module()

    y = _load_audio(audio)
    seconds = len(y) / TARGET_SR

    if seconds < min_seconds:
        return {
            "label": None, "confidence": None, "probabilities": {},
            "skipped": True,
            "reason": f"turn too short ({seconds:.2f}s < {min_seconds}s)",
            "turn_seconds": round(seconds, 2),
        }

    y = preprocess_like_training(y, TARGET_SR)
    if y.size < TARGET_SR // 2:
        return {
            "label": None, "confidence": None, "probabilities": {},
            "skipped": True,
            "reason": "nothing left after denoise and trim (silence?)",
            "turn_seconds": round(seconds, 2),
        }

    acoustic = frozen.extract_acoustic(y, TARGET_SR)
    textual = text_features_from_transcript(transcript, language)
    row = {**acoustic, **textual}

    text_vector = np.array([[float(row.get(c, 0.0) or 0.0) for c in TEXT_ONLY_COLS]])
    acoustic_vector = np.array([[float(row.get(c, 0.0) or 0.0) for c in ACOUSTIC_ONLY_COLS]])

    proba_text = text_model.predict_proba(text_vector)[0]
    proba_acoustic = acoustic_model.predict_proba(acoustic_vector)[0]
    proba = 0.5 * proba_text + 0.5 * proba_acoustic
    index = int(proba.argmax())

    result = {
        "label": classes[index],
        "confidence": round(float(proba[index]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
        "skipped": False,
        "turn_seconds": round(seconds, 2),
        "language_normalised": textual["language_normalised"],
        "keyword_count": round(float(textual["keyword_count"]), 2),
        "asr_quality": round(float(textual["asr_quality"]), 3),
        "context_score": int(textual.get("sin_context_score", 0)
                             or textual.get("tam_context_score", 0) or 0),
    }

    # Observational only -- classify_urgency() (the single-model path) already
    # logs this; the late-fusion path Component 2 actually calls had no
    # logging at all, so a live misclassification was undiagnosable after the
    # fact. Mirrors the existing log line exactly, plus the two branch
    # probabilities so a genuine text/acoustic disagreement is visible.
    print(
        "[urgency-latefusion] turn "
        + str(len(_sessions.get(str(session_id), [])) + 1)
        + "  lang=" + str(textual["language_normalised"])
        + "  sec=" + str(round(seconds, 1))
        + "  kw=" + str(round(float(textual["keyword_count"]), 2))
        + "  ctx=" + str(result["context_score"])
        + "  asrq=" + str(round(float(textual["asr_quality"]), 2))
        + "  fire_smoke_match=" + str(row.get("fire_smoke_match"))
        + "  text_proba=" + str({c: round(float(p), 3) for c, p in zip(classes, proba_text)})
        + "  acoustic_proba=" + str({c: round(float(p), 3) for c, p in zip(classes, proba_acoustic)})
        + "  ->  " + str(result["label"])
        + " (fused=" + str(result["probabilities"]) + ")"
        + "  transcript=" + repr(transcript[:80])
    )

    if session_id is not None:
        with _lock:
            _sessions.setdefault(str(session_id), []).append(result)
        result["turn_index"] = len(_sessions[str(session_id)])

    return result


def classify_urgency_late_fusion_async(audio, transcript, language, session_id,
                                       min_seconds=1.0):
    """classify_urgency_async(), but using classify_urgency_late_fusion()."""
    if isinstance(audio, (str, os.PathLike)):
        payload = str(audio)
    else:
        payload = np.array(audio, copy=True)

    def _work():
        try:
            classify_urgency_late_fusion(payload, transcript, language,
                                         session_id=session_id, min_seconds=min_seconds)
        except Exception as error:
            with _lock:
                _sessions.setdefault(str(session_id), []).append({
                    "label": None, "skipped": True,
                    "reason": f"error: {error}",
                })

    thread = threading.Thread(target=_work, daemon=True,
                              name=f"urgency-latefusion-{session_id}")
    with _lock:
        _pending.setdefault(str(session_id), []).append(thread)
    thread.start()
    return thread


def classify_urgency_async(audio, transcript, language, session_id,
                           min_seconds=1.0):
    """
    Same as classify_urgency() but returns immediately, doing the work on a
    background thread.

    Acoustic extraction is ~2 s for a 5 s turn, almost entirely pitch
    tracking, and scales with turn length. The urgency verdict is not needed
    until the call ends, so there is no reason to make the caller wait for it
    mid-conversation. end_call() joins any threads still running.

    Returns the Thread, which callers can normally ignore.
    """
    if isinstance(audio, (str, os.PathLike)):
        payload = str(audio)          # read on the worker thread
    else:
        payload = np.array(audio, copy=True)

    def _work():
        try:
            classify_urgency(payload, transcript, language,
                             session_id=session_id, min_seconds=min_seconds)
        except Exception as error:                       # never kill the call
            with _lock:
                _sessions.setdefault(str(session_id), []).append({
                    "label": None, "skipped": True,
                    "reason": f"error: {error}",
                })

    thread = threading.Thread(target=_work, daemon=True,
                              name=f"urgency-{session_id}")
    with _lock:
        _pending.setdefault(str(session_id), []).append(thread)
    thread.start()
    return thread


def wait_for_pending(session_id, timeout: float = 30.0) -> int:
    """Join outstanding async turns for a session. Returns how many completed."""
    with _lock:
        threads = list(_pending.get(str(session_id), []))

    deadline = time.time() + timeout
    done = 0
    for thread in threads:
        remaining = max(0.0, deadline - time.time())
        thread.join(timeout=remaining)
        if not thread.is_alive():
            done += 1

    with _lock:
        _pending[str(session_id)] = [t for t in threads if t.is_alive()]
    return done


def get_call_urgency(session_id) -> dict:
    """
    Aggregate every turn recorded for a session into one call-level verdict.

    Rule fixed in advance: mean probability vector across turns, argmax.
    Peak severity is reported alongside, not substituted, since a single
    confidently-High turn is operationally meaningful even when the call
    averages calmer.
    """
    _, classes = _ensure_model()
    turns = [t for t in _sessions.get(str(session_id), []) if not t.get("skipped")]

    if not turns:
        return {"label": None, "n_turns": 0,
                "reason": "no classifiable turns"}

    matrix = np.array([[t["probabilities"][c] for c in classes] for t in turns])
    mean = matrix.mean(axis=0)
    index = int(mean.argmax())

    peak = max((t["label"] for t in turns), key=lambda lab: SEVERITY[lab])

    return {
        "label": classes[index],
        "confidence": round(float(mean[index]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, mean)},
        "peak_label": peak,
        "n_turns": len(turns),
        "turn_labels": [t["label"] for t in turns],
        "aggregation": "mean_probability",
    }


def end_call(session_id, clear: bool = True, timeout: float = 30.0) -> dict:
    """
    Final urgency for a completed call.

    Waits for any async turns still in flight before aggregating, so a verdict
    is never computed from a partial set of turns.
    """
    wait_for_pending(session_id, timeout=timeout)
    summary = get_call_urgency(session_id)
    if clear:
        with _lock:
            _sessions.pop(str(session_id), None)
            _pending.pop(str(session_id), None)
    return summary


def reset_session(session_id) -> None:
    with _lock:
        _sessions.pop(str(session_id), None)
        _pending.pop(str(session_id), None)


if __name__ == "__main__":
    print(__doc__)
    print("Loading model ...")
    model, classes = _ensure_model()
    print(f"  classes      : {classes}")
    print(f"  features     : {len(FEATURE_COLS)}")
    print(f"  model cached : {MODEL_PATH}")
