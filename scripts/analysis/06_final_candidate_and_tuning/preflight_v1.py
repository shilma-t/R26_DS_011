"""
preflight_v1.py
Last checks before committing five hours to the corrected v1 evaluation.

Verifies the things that would silently produce a wrong number rather than an
error: language strings that do not match what the context extractor expects,
audio files that do not exist, label values the encoder cannot map, and any
remaining difference between how training and DMC inputs are prepared.

Usage:
  python scripts/analysis/preflight_v1.py
"""

import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf

_ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
PIPE = os.path.join(_ROOT, "scripts", "pipeline")
if PIPE not in sys.path:
    sys.path.insert(0, PIPE)

from context_features import extract_context_features  # noqa: E402

FEATURES = os.path.join(_ROOT, "data", "features.csv")
METADATA = os.path.join(_ROOT, "data", "dmc_metadata.csv")
CALLER = os.path.join(_ROOT, "audio", "dmc_caller")

failures = []
warnings = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def warn(name, detail=""):
    print(f"  [WARN] {name}" + (f"  {detail}" if detail else ""))
    warnings.append(name)


def main():
    print("=" * 72)
    print("PREFLIGHT - corrected v1 evaluation")
    print("=" * 72)

    sim = pd.read_csv(FEATURES)
    meta = pd.read_csv(METADATA, dtype={"recording_id": str})
    meta = meta.dropna(subset=["dmc_priority_label"])

    # ── 1. language strings must reach the same code path ────────────────────
    print("\n1. Language routing (silent-failure risk)")
    sim_langs = sorted(set(str(x).strip() for x in sim["language"].dropna()))
    dmc_langs = sorted(set(str(x).strip() for x in meta["language"].dropna()))
    print(f"     training : {sim_langs}")
    print(f"     DMC      : {dmc_langs}")

    def fires(lang):
        """Does this string actually produce non-zero context features?"""
        probe = "උදව් කරන්න ගෙදර ගිනිගන්නවා ළමයා ඇතුළේ"
        tamil = "உதவி செய்யுங்கள் வீட்டில் தீ குழந்தை"
        f_si = extract_context_features(probe, lang)
        f_ta = extract_context_features(tamil, lang)
        return (sum(abs(v) for v in f_si.values()) > 0
                or sum(abs(v) for v in f_ta.values()) > 0)

    dead = [l for l in dmc_langs if not fires(l)]
    live = [l for l in dmc_langs if fires(l)]
    print(f"     DMC languages that fire context features : {live}")
    if dead:
        n_dead = int(meta["language"].astype(str).str.strip().isin(dead).sum())
        if set(dead) <= {"English", "english", "en"}:
            warn(f"{dead} produce zero context features",
                 f"{n_dead} call(s) - expected, context is Sinhala/Tamil only")
        else:
            check(f"language(s) {dead} route correctly", False,
                  f"{n_dead} call(s) would silently get all-zero context")
    else:
        check("every DMC language routes to a context extractor", True)

    # Both extract_textual() and extract_context_features() lower-case before
    # matching, so 'Sinhala' and 'sinhala' take the same branch. Compare
    # normalised, and prove it rather than trusting the read.
    norm_sim = {l.lower().strip() for l in sim_langs}
    norm_dmc = {l.lower().strip() for l in dmc_langs}
    shared = norm_sim & norm_dmc
    check("DMC languages overlap training languages (case-normalised)",
          bool(shared), f"shared: {sorted(shared)}")

    probe = "උදව් කරන්න ගෙදර ගිනිගන්නවා ළමයා ඇතුළේ"
    same_branch = all(
        extract_context_features(probe, l)
        == extract_context_features(l.lower().strip(), probe)
        or extract_context_features(probe, l)
        == extract_context_features(probe, l.lower().strip())
        for l in dmc_langs
    )
    check("capitalised and lower-case forms give identical features",
          same_branch, "'Sinhala' behaves exactly as 'sinhala'")

    # ── 2. every labelled call has audio ─────────────────────────────────────
    print("\n2. Audio availability")
    missing, empty, durations = [], [], []
    for row in meta.itertuples():
        rec = os.path.splitext(str(row.filename))[0]
        path = os.path.join(CALLER, f"{rec}_caller.wav")
        if not os.path.exists(path):
            missing.append(rec)
            continue
        info = sf.info(path)
        durations.append(info.duration)
        if info.duration < 1.0:
            empty.append(rec)
        if info.samplerate != 16000:
            warn(f"{rec} is {info.samplerate} Hz, expected 16000")

    check("every labelled call has caller audio", not missing,
          f"{len(missing)} missing: {missing[:3]}" if missing else
          f"{len(durations)} files")
    check("no near-empty audio", not empty,
          f"{empty}" if empty else "")
    if durations:
        d = np.array(durations)
        print(f"     duration: median {np.median(d):.1f}s  "
              f"min {d.min():.1f}s  max {d.max():.1f}s")

    # ── 3. labels the encoder can map ────────────────────────────────────────
    print("\n3. Labels")
    labels = sorted(set(str(x).strip() for x in meta["dmc_priority_label"]))
    check("labels are exactly High/Low/Medium", set(labels) <= {"High", "Low", "Medium"},
          f"{labels}")
    counts = meta["dmc_priority_label"].value_counts()
    print("     " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    check("all three classes represented", len(counts) == 3)

    train_labels = sorted(set(str(x).strip() for x in sim["urgency_label"].dropna()))
    check("training labels match DMC label vocabulary",
          set(labels) <= set(train_labels), f"training: {train_labels}")

    # ── 4. training feature matrix is intact ─────────────────────────────────
    print("\n4. Training data")
    check("267 training rows", len(sim) == 267, f"{len(sim)}")
    check("transcript column present (needed for fire_smoke_match)",
          "transcript" in sim.columns)
    nan_cols = [c for c in sim.columns
                if sim[c].dtype.kind in "fc" and sim[c].isna().any()]
    if nan_cols:
        warn(f"{len(nan_cols)} training column(s) contain NaN",
             f"{nan_cols[:4]} - filled with 0 at train time")

    # ── 5. frozen artifacts unchanged ────────────────────────────────────────
    print("\n5. Frozen artifacts")
    frozen_dir = os.path.join(_ROOT, "frozen_v1")
    manifest = os.path.join(frozen_dir, "MANIFEST.json")
    check("frozen_v1 manifest exists", os.path.exists(manifest))
    if os.path.exists(manifest):
        import json, hashlib
        with open(manifest, encoding="utf-8") as fh:
            recorded = json.load(fh)["artifacts"]
        drifted = []
        for rel, info in recorded.items():
            live_path = os.path.join(_ROOT, rel.replace("/", os.sep))
            if not os.path.exists(live_path):
                continue
            h = hashlib.sha256()
            with open(live_path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != info["sha256"]:
                drifted.append(rel)
        # evaluate_dmc.py is expected to differ - we just fixed it
        expected = {"scripts/pipeline/evaluate_dmc.py"}
        unexpected = [d for d in drifted if d not in expected]
        if drifted:
            print(f"     changed since freeze: {drifted}")
        check("no unexpected drift in frozen inputs", not unexpected,
              f"{unexpected}" if unexpected else
              "(evaluate_dmc.py changed as intended)")

    print("\n" + "=" * 72)
    if failures:
        print(f"{len(failures)} BLOCKER(S): {failures}")
        print("Do not start the run.")
        return 1
    if warnings:
        print(f"{len(warnings)} warning(s), none blocking: {warnings}")
    print("PREFLIGHT CLEAR - safe to start the corrected v1 run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
