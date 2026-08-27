"""
evaluate_dmc.py
Locked external evaluation of the frozen candidate pipeline on real DMC calls.

FAITHFULNESS
------------
This script does not reimplement feature extraction. It imports and calls the
same functions that produced data/features.csv for the simulated corpus:

  02_extract_features.extract_acoustic()    - 91 acoustic features
  02_extract_features.extract_textual()     - ASR + 5 text features
  context_features.extract_context_features() - 14 context features (via extract_textual)

An earlier version of this file contained simplified copies of those functions
whose outputs did not match (keyword_count used 14 fire words instead of the
~70-term URGENCY_KEYWORDS list, context features used truncated word lists with
no phrase matching, sentiment_polarity and speaking_rate_change were hardcoded
to zero). Those copies are gone. Do not reintroduce them - if the training
pipeline changes, this evaluation must change with it automatically.

fire_smoke_match is absent from features.csv, so it is recomputed here for the
training rows from their stored transcripts. Training and test therefore both
receive real values for it.

The frozen candidate is condition D (ungated). No ASR-quality gate is applied.

THIS SCRIPT MUST BE RUN ONCE AND THE RESULT REPORTED AS-IS.
Tuning anything after seeing the output turns DMC into development data and
destroys the external validation.

Usage:
  python scripts/pipeline/evaluate_dmc.py             # caller-only (primary)
  python scripts/pipeline/evaluate_dmc.py --raw       # whole-call baseline

Outputs:
  reports/pp2/dmc_evaluation[_raw].csv
  reports/pp2/dmc_results[_raw].txt
"""

import os
import sys
import argparse
import importlib.util
from datetime import datetime

import numpy as np
import pandas as pd
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

_ROOT      = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
PIPE_DIR   = os.path.join(_ROOT, "scripts", "pipeline")
CALLER_DIR = os.path.join(_ROOT, "audio", "dmc_caller")
RAW_DIR    = os.path.join(_ROOT, "audio", "dmc_raw")
METADATA   = os.path.join(_ROOT, "data", "dmc_metadata.csv")
FEATURES   = os.path.join(_ROOT, "data", "features.csv")
REPORTS    = os.path.join(_ROOT, "reports", "pp2")

SEED = 42
RF_PARAMS = dict(n_estimators=200, min_samples_leaf=2,
                 class_weight="balanced", random_state=SEED, n_jobs=-1)
TARGET_SR = 16_000

# Frozen feature set (111) - must match candidate_pipeline_freeze.txt exactly
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

FIRE_SMOKE_KEYWORDS = [
    "fire", "smoke", "burning", "flames", "gas",
    "ගිනිගන්නව", "ගිනිඅරගෙන", "ගින්නක්", "ගිනි", "දුමක්", "දුම",
    "தீ", "புகை", "gini",
]


def load_pipeline_module():
    """Import 02_extract_features.py (leading digit blocks normal import)."""
    if PIPE_DIR not in sys.path:
        sys.path.insert(0, PIPE_DIR)
    path = os.path.join(PIPE_DIR, "02_extract_features.py")
    spec = importlib.util.spec_from_file_location("extract_features_mod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preprocess_like_training(y, sr=TARGET_SR):
    """
    Apply the three steps 01_preprocess.py applies to the training corpus.

    CRITICAL. The 267 training calls were denoised, resampled and silence-
    trimmed into audio/clean/ before 02_extract_features.py ever saw them, so
    the model's thresholds were learned on preprocessed audio. An earlier
    version of this script fed DMC audio in raw, which inflated rms_mean by
    ~4x, depressed every MFCC standard deviation, and produced a macro-F1 of
    0.2524 that measured the mismatch rather than the model.

    Verified by scripts/analysis/verify_extraction_path.py: with this applied,
    DMC feature distributions sit on top of the training distributions
    (|Cohen's d| falls from 5.45 to 0.16 on rms_mean).
    """
    import noisereduce as nr

    y_nr = nr.reduce_noise(y=y, sr=sr)
    if sr != TARGET_SR:
        y_nr = librosa.resample(y_nr, orig_sr=sr, target_sr=TARGET_SR)
    y_trimmed, _ = librosa.effects.trim(y_nr, top_db=20)

    # Trimming can wipe a very quiet clip entirely; keep the denoised audio
    # rather than returning an empty array.
    if y_trimmed.size < TARGET_SR // 2:
        return y_nr
    return y_trimmed


def fire_smoke_match(text):
    if not isinstance(text, str):
        return 0
    lowered = text.lower()
    return int(any(kw.lower() in lowered for kw in FIRE_SMOKE_KEYWORDS))


def build_training_set(feat_mod):
    """267 simulated calls, using the stored feature matrix as-is."""
    sim = pd.read_csv(FEATURES)

    if "fire_smoke_match" not in sim.columns:
        if "transcript" not in sim.columns:
            raise SystemExit(
                "features.csv has neither fire_smoke_match nor transcript. "
                "Cannot reconstruct the frozen feature set - re-run 02_extract_features.py."
            )
        sim["fire_smoke_match"] = sim["transcript"].map(fire_smoke_match)
        print(f"  fire_smoke_match recomputed from transcripts: "
              f"{int(sim['fire_smoke_match'].sum())}/{len(sim)} calls positive")

    missing = [c for c in FEATURE_COLS if c not in sim.columns]
    if missing:
        raise SystemExit(f"features.csv is missing frozen feature columns: {missing}")

    if "urgency_label" not in sim.columns:
        raise SystemExit("features.csv has no urgency_label column.")

    sim = sim.dropna(subset=["urgency_label"])
    return sim


def extract_turn_features(feat_mod, meta):
    """
    One feature row per caller TURN.

    Turns are the unit the live system classifies (one caller utterance
    between agent questions), and their median length of 5.2s matches the
    6.9s median of the simulated training clips. Whole-call audio does not.
    """
    from caller_turns import build_all_turns

    turns = build_all_turns()
    if turns.empty:
        raise SystemExit("No caller turns produced - check worksheet labels.")

    label_by_id = {}
    lang_by_id = {}
    for row in meta.itertuples():
        rec_id = os.path.splitext(str(row.filename))[0]
        label_by_id[rec_id] = row.dmc_priority_label
        lang_by_id[rec_id] = str(row.language).strip()

    rows = []
    grouped = list(turns.groupby("recording_id", sort=False))

    for call_no, (rec_id, call_turns) in enumerate(grouped, start=1):
        rec_id = str(rec_id)
        if rec_id not in label_by_id:
            continue

        path = os.path.join(CALLER_DIR, f"{rec_id}_caller.wav")
        src = os.path.join(_ROOT, "audio", "dmc", str(call_turns["filename"].iloc[0]))
        # Turn boundaries index the ORIGINAL recording, not the concatenated
        # caller file, so slice from the source audio.
        if not os.path.exists(src):
            print(f"  [{call_no}] MISSING {os.path.basename(src)}")
            continue

        y_full, sr = librosa.load(src, sr=TARGET_SR)
        language = lang_by_id[rec_id]

        for turn in call_turns.itertuples():
            a = max(0, int(turn.start * TARGET_SR))
            b = min(len(y_full), int(turn.end * TARGET_SR))
            y = y_full[a:b]
            if y.size < TARGET_SR:          # under 1s of samples
                continue

            # Preprocess each turn individually, mirroring training where each
            # call was denoised and trimmed as its own file.
            y = preprocess_like_training(y, TARGET_SR)
            if y.size < TARGET_SR // 2:
                continue

            acoustic = feat_mod.extract_acoustic(y, TARGET_SR)
            textual = feat_mod.extract_textual(y, language)
            transcript = textual.get("transcript", "")

            rows.append({
                "recording_id": rec_id,
                "turn_index":   turn.turn_index,
                "turn_start":   turn.start,
                "turn_end":     turn.end,
                "turn_seconds": round(turn.duration, 2),
                "language":     language,
                "true_label":   label_by_id[rec_id],
                **acoustic,
                **{k: v for k, v in textual.items() if k != "transcript"},
                "fire_smoke_match": fire_smoke_match(transcript),
                "transcript":   transcript,
            })

        done = sum(1 for r in rows if r["recording_id"] == rec_id)
        print(f"  [{call_no}/{len(grouped)}] {rec_id[:34]:<34} {done} turn(s)")

    return pd.DataFrame(rows)


def aggregate_turns(turn_df, le, model, feature_cols):
    """
    Turn predictions -> one label per call.

    Aggregation rule fixed BEFORE any result was seen:
      primary  = argmax of the mean probability vector across turns
      reported = peak severity (most urgent single turn), since operationally
                 a confidently-High turn should escalate rather than be averaged
    """
    X = turn_df[feature_cols].fillna(0).values
    proba = model.predict_proba(X)

    for i, cls in enumerate(le.classes_):
        turn_df[f"prob_{cls}"] = proba[:, i]
    turn_df["turn_prediction"] = le.inverse_transform(proba.argmax(axis=1))

    severity = {"Low": 0, "Medium": 1, "High": 2}
    prob_cols = [f"prob_{c}" for c in le.classes_]

    calls = []
    for rec_id, group in turn_df.groupby("recording_id", sort=False):
        mean_probs = group[prob_cols].mean()
        mean_label = le.classes_[int(mean_probs.values.argmax())]
        peak_label = max(group["turn_prediction"], key=lambda p: severity[p])

        calls.append({
            "recording_id":    rec_id,
            "true_label":      group["true_label"].iloc[0],
            "n_turns":         len(group),
            "predicted_label": mean_label,
            "peak_label":      peak_label,
            **{c: round(float(mean_probs[c]), 4) for c in prob_cols},
        })

    out = pd.DataFrame(calls)
    out["correct"] = out["true_label"] == out["predicted_label"]
    out["peak_correct"] = out["true_label"] == out["peak_label"]
    return out


def extract_dmc_features(feat_mod, audio_dir, suffix, meta):
    """Run the REAL extraction functions over each DMC recording."""
    rows = []
    total = len(meta)

    for i, row in enumerate(meta.itertuples(), start=1):
        rec_id = os.path.splitext(str(row.filename))[0]
        path = os.path.join(audio_dir, f"{rec_id}{suffix}.wav")

        if not os.path.exists(path):
            print(f"  [{i}/{total}] MISSING {os.path.basename(path)}")
            continue

        y, sr = librosa.load(path, sr=TARGET_SR)
        if y.size == 0:
            print(f"  [{i}/{total}] EMPTY {rec_id}")
            continue

        language = str(row.language).strip()

        # Match the training corpus: denoise + trim before extraction.
        raw_seconds = len(y) / TARGET_SR
        y = preprocess_like_training(y, TARGET_SR)

        acoustic = feat_mod.extract_acoustic(y, TARGET_SR)
        textual = feat_mod.extract_textual(y, language)   # ASR + text + context

        transcript = textual.get("transcript", "")
        record = {
            "recording_id": rec_id,
            "filename": row.filename,
            "language": language,
            "true_label": row.dmc_priority_label,
            **acoustic,
            **{k: v for k, v in textual.items() if k != "transcript"},
            "fire_smoke_match": fire_smoke_match(transcript),
            "transcript": transcript,
        }
        rows.append(record)

        preview = transcript[:44].replace("\n", " ")
        print(f"  [{i}/{total}] {rec_id[:30]:<30} "
              f"{raw_seconds:5.1f}->{len(y)/TARGET_SR:5.1f}s "
              f"q={textual.get('asr_quality', 0):.2f} "
              f"kw={textual.get('keyword_count', 0):.1f}  {preview}")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--turns", action="store_true",
                      help="Turn-level caller audio, aggregated (PRIMARY)")
    mode.add_argument("--raw", action="store_true",
                      help="Whole-call mixed audio (contamination baseline)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N calls (smoke test; "
                             "results are NOT a valid evaluation)")
    args = parser.parse_args()

    if args.turns:
        audio_dir, suffix = CALLER_DIR, "_caller"
        tag = "TURN-LEVEL CALLER (primary)"
    elif args.raw:
        audio_dir, suffix = RAW_DIR, "_raw"
        tag = "RAW WHOLE-CALL (contamination baseline)"
    else:
        audio_dir, suffix = CALLER_DIR, "_caller"
        tag = "WHOLE-CALL CALLER-ONLY (secondary)"

    if not os.path.isdir(audio_dir):
        raise SystemExit(f"Audio directory not found: {audio_dir}\n"
                         "Run extract_caller_audio.py first.")

    print("=" * 68)
    print(f"DMC LOCKED EXTERNAL EVALUATION  -  {tag}")
    print("=" * 68)

    meta = pd.read_csv(METADATA, dtype={"recording_id": str})
    meta = meta.dropna(subset=["dmc_priority_label"])

    if args.limit:
        meta = meta.head(args.limit)
        print("\n*** SMOKE TEST: %d call(s) only - NOT a valid evaluation ***"
              % args.limit)

    print(f"\nDMC calls with labels: {len(meta)}")
    print(meta["dmc_priority_label"].value_counts().to_string())

    print("\nLoading real feature-extraction code...")
    feat_mod = load_pipeline_module()
    print(f"  URGENCY_KEYWORDS terms: {len(feat_mod.URGENCY_KEYWORDS)}")

    print("\nBuilding training set from simulated corpus...")
    sim = build_training_set(feat_mod)
    print(f"  {len(sim)} calls")
    print("  " + sim["urgency_label"].value_counts().to_string().replace("\n", "\n  "))

    if args.turns:
        print("\nExtracting per-turn features from caller turns ...")
        print("(ASR runs per turn - this takes a while)\n")
        units = extract_turn_features(feat_mod, meta)
    else:
        print(f"\nExtracting DMC features from {os.path.basename(audio_dir)}/ ...")
        print("(ASR runs per call - this takes a while)\n")
        units = extract_dmc_features(feat_mod, audio_dir, suffix, meta)

    if units.empty:
        raise SystemExit("No DMC features extracted.")

    for col in FEATURE_COLS:
        if col not in units.columns:
            units[col] = 0.0

    if args.turns:
        print(f"\nExtracted {len(units)} turns across "
              f"{units['recording_id'].nunique()} calls")
    else:
        print(f"\nExtracted features for {len(units)}/{len(meta)} calls")

    # ── Train on the full simulated corpus ────────────────────────────────────
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    X_train = sim[FEATURE_COLS].fillna(0).values
    y_train = le.transform(sim["urgency_label"].values)

    model = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=SEED)),
        ("clf", RandomForestClassifier(**RF_PARAMS)),
    ])
    model.fit(X_train, y_train)
    print("Model trained on 267 simulated calls.")

    # ── Predict ───────────────────────────────────────────────────────────────
    if args.turns:
        dmc = aggregate_turns(units, le, model, FEATURE_COLS)
        units.to_csv(os.path.join(REPORTS, "dmc_turn_predictions.csv"), index=False)
        print(f"Per-turn predictions -> reports/pp2/dmc_turn_predictions.csv")
        y_true = le.transform(dmc["true_label"].values)
        y_pred = le.transform(dmc["predicted_label"].values)
    else:
        dmc = units
        X_dmc = dmc[FEATURE_COLS].fillna(0).values
        y_true = le.transform(dmc["true_label"].values)
        y_pred = model.predict(X_dmc)
        y_proba = model.predict_proba(X_dmc)
        dmc["predicted_label"] = le.inverse_transform(y_pred)
        for i, cls in enumerate(le.classes_):
            dmc[f"prob_{cls}"] = y_proba[:, i]
        dmc["correct"] = dmc["true_label"] == dmc["predicted_label"]

    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=le.classes_,
                                   zero_division=0, digits=3)
    cm = confusion_matrix(y_true, y_pred, labels=le.transform(le.classes_))

    print("\n" + "=" * 68)
    print(f"RESULT  -  {tag}")
    print("=" * 68)
    print(f"Calls      : {len(dmc)}")
    print(f"Accuracy   : {dmc['correct'].mean():.3f}")
    print(f"Macro-F1   : {macro:.4f}")
    if args.turns:
        peak_macro = f1_score(y_true, le.transform(dmc["peak_label"].values),
                              average="macro", zero_division=0)
        print(f"Turns      : {len(units)} ({len(units) / len(dmc):.1f} per call)")
        print(f"Peak-severity rule: accuracy {dmc['peak_correct'].mean():.3f}, "
              f"macro-F1 {peak_macro:.4f}")
    print(f"(simulated development macro-F1 was 0.6208 - separate table)")
    print()
    print(report)
    print("Confusion matrix (rows=true, cols=predicted)")
    print(f"Classes: {list(le.classes_)}")
    print(cm)

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.turns:
        stem, txt_name = "dmc_evaluation_turns", "dmc_results_turns.txt"
    elif args.raw:
        stem, txt_name = "dmc_evaluation_raw", "dmc_results_raw.txt"
    else:
        stem, txt_name = "dmc_evaluation", "dmc_results.txt"

    # A partial run must never overwrite a real result file
    if args.limit:
        stem += "_SMOKETEST"
        txt_name = txt_name.replace(".txt", "_SMOKETEST.txt")

    dmc.to_csv(os.path.join(REPORTS, f"{stem}.csv"), index=False)
    txt = os.path.join(REPORTS, txt_name)
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write("DMC LOCKED EXTERNAL EVALUATION\n")
        fh.write("=" * 68 + "\n")
        fh.write(f"Condition    : {tag}\n")
        fh.write(f"Audio source : audio/{os.path.basename(audio_dir)}/\n")
        fh.write(f"Run at       : {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"Features     : {len(FEATURE_COLS)} (frozen candidate D, ungated)\n")
        fh.write(f"Training set : {len(sim)} simulated calls\n")
        fh.write(f"Calls        : {len(dmc)}\n\n")
        fh.write(f"Accuracy : {dmc['correct'].mean():.3f}\n")
        fh.write(f"Macro-F1 : {macro:.4f}\n\n")
        fh.write(report)
        fh.write("\nConfusion matrix (rows=true, cols=predicted)\n")
        fh.write(f"Classes: {list(le.classes_)}\n")
        fh.write(str(cm) + "\n\n")
        fh.write("Simulated development macro-F1 (5-fold CV): 0.6208\n")
        fh.write("Report simulated and DMC results in SEPARATE tables.\n")

    print(f"\nSaved -> reports/pp2/{stem}.csv")
    print(f"Saved -> reports/pp2/{os.path.basename(txt)}")


if __name__ == "__main__":
    main()
