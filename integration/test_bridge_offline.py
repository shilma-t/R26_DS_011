"""
test_bridge_offline.py
Exercise urgency_bridge against real DMC audio before Component 2 is involved.

Checks the things that would otherwise fail during a live demo. Two paths are
covered: the original single-model fused-feature path (classify_urgency /
classify_urgency_async - kept for regression coverage, though nothing live
calls it anymore) and the late-fusion path (classify_urgency_late_fusion /
_async), which is what Component 2 and the standalone demo actually call in
production. The two are separately-implemented functions in urgency_bridge.py
(copy-built, not sharing a common code path), so passing on one gives no
guarantee about the other - both need their own checks.

  1-8.  classify_urgency / classify_urgency_async (legacy; unused live, kept
        so a regression here is still caught)
  9-16. classify_urgency_late_fusion / _async (what Component 2 and the demo
        actually run)

Usage:
  python integration/test_bridge_offline.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from urgency_bridge import (  # noqa: E402
    classify_urgency,
    classify_urgency_async,
    classify_urgency_late_fusion,
    classify_urgency_late_fusion_async,
    end_call,
    get_call_urgency,
    reset_session,
    warmup,
    _load_audio,
)

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
TURNS = os.path.join(ROOT, "reports", "pp2", "dmc_caller_turns.csv")
DMC = os.path.join(ROOT, "audio", "dmc")
SCRATCH = os.path.join(_HERE, "_test_turns")

# Representative Sinhala emergency text, standing in for Component 2's ASR
SAMPLE_TRANSCRIPTS = [
    "උදව් කරන්න, ගෙදර ගිනිගන්නවා, ළමයා ඇතුළේ ඉන්නවා",
    "හිරවෙලා ඉන්නවා, වතුර ඇතුළට ආවා, ඉක්මනට එන්න",
    "ආයුබෝවන්, මම දැනගන්න කැමතියි පාර විවෘතද කියලා",
]

failures = []


def check(name, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}"
          + (f"  {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def main():
    print("=" * 66)
    print("URGENCY BRIDGE - OFFLINE TEST")
    print("=" * 66)

    print("\nWarming up (loads model + feature code) ...")
    t0 = time.perf_counter()
    warmup()
    print(f"  warmup: {time.perf_counter() - t0:.1f}s")

    if not os.path.exists(TURNS):
        raise SystemExit(f"Turn table not found: {TURNS}")

    turns = pd.read_csv(TURNS, dtype={"recording_id": str, "filename": str})
    os.makedirs(SCRATCH, exist_ok=True)

    # Cut three real caller turns from one call, standing in for live mic turns
    import soundfile as sf
    rec_id = turns["recording_id"].iloc[0]
    call_turns = turns[turns["recording_id"] == rec_id].head(3)
    src = os.path.join(DMC, str(call_turns["filename"].iloc[0]))

    import librosa
    y_full, _ = librosa.load(src, sr=16000)

    # Component 2 records fixed 5 s turns (record_wav default), so cap each
    # turn at 5 s. Testing on raw 18 s DMC turns would measure a latency the
    # live system never incurs.
    LIVE_TURN_SEC = 5.0
    clips = []
    for row in call_turns.itertuples():
        a = int(row.start * 16000)
        b = min(int(row.end * 16000), a + int(LIVE_TURN_SEC * 16000))
        path = os.path.join(SCRATCH, f"turn{row.turn_index}.wav")
        sf.write(path, y_full[a:b], 16000, subtype="PCM_16")
        clips.append((path, round((b - a) / 16000, 1)))

    print(f"\nUsing {len(clips)} real caller turns from {rec_id}, "
          f"capped at {LIVE_TURN_SEC:.0f}s to match live recording length")
    for p, d in clips:
        print(f"  {os.path.basename(p)}  {d}s")

    # ── 1. basic prediction ──────────────────────────────────────────────────
    print("\n1. Prediction")
    t0 = time.perf_counter()
    result = classify_urgency(clips[0][0], SAMPLE_TRANSCRIPTS[0], "si-LK")
    elapsed = time.perf_counter() - t0
    check("returns a label", result["label"] in {"High", "Medium", "Low"},
          f"{result['label']} @ {result['confidence']}")
    # Probabilities are rounded to 4dp for the emitted JSON, so the sum can
    # drift by up to ~1.5e-4. Tolerance reflects that, not model error.
    check("probabilities sum to 1",
          abs(sum(result["probabilities"].values()) - 1.0) < 5e-4,
          f"sum={sum(result['probabilities'].values()):.6f}")
    print(f"         probabilities: {result['probabilities']}")

    # ── 2. path vs array ─────────────────────────────────────────────────────
    print("\n2. Path input vs array input")
    arr = _load_audio(clips[0][0])
    from_array = classify_urgency(arr, SAMPLE_TRANSCRIPTS[0], "si-LK")
    check("identical probabilities",
          from_array["probabilities"] == result["probabilities"])

    # ── 3. language mapping ──────────────────────────────────────────────────
    print("\n3. Language code mapping")
    si = classify_urgency(clips[0][0], SAMPLE_TRANSCRIPTS[0], "si-LK")
    check("si-LK normalises to sinhala",
          si["language_normalised"] == "sinhala")
    check("Sinhala context features fire", si["context_score"] > 0,
          f"context_score={si['context_score']}")

    # ── 4. the silent failure ────────────────────────────────────────────────
    print("\n4. Unmapped language code (the silent failure mode)")
    bad = classify_urgency(clips[0][0], SAMPLE_TRANSCRIPTS[0], "sinhala-LK")
    check("unmapped code zeroes context", bad["context_score"] == 0,
          "confirms why the mapping is load-bearing")
    check("but still returns a label", bad["label"] is not None,
          f"{bad['label']} - fails quietly, not loudly")

    # ── 5. short turn guard ──────────────────────────────────────────────────
    print("\n5. Short-turn guard")
    tiny = np.zeros(int(0.4 * 16000), dtype=np.float32)
    skipped = classify_urgency(tiny, "ඔව්", "si-LK")
    check("sub-second turn skipped", skipped["skipped"] is True,
          skipped.get("reason", ""))

    # ── 6. session aggregation ───────────────────────────────────────────────
    print("\n6. Session aggregation")
    session = "test_session"
    reset_session(session)
    for (path, _), text in zip(clips, SAMPLE_TRANSCRIPTS):
        r = classify_urgency(path, text, "si-LK", session_id=session)
        print(f"         turn {r.get('turn_index')}: {r['label']:<7} "
              f"conf={r['confidence']:.3f}  kw={r['keyword_count']:.1f}")

    running = get_call_urgency(session)
    check("aggregates all turns", running["n_turns"] == len(clips),
          f"n_turns={running['n_turns']}")
    check("produces a call label", running["label"] in {"High", "Medium", "Low"},
          f"{running['label']} (peak={running['peak_label']})")
    print(f"         turn labels: {running['turn_labels']}")
    print(f"         call mean  : {running['probabilities']}")

    final = end_call(session)
    check("end_call returns the same verdict", final["label"] == running["label"])
    check("session cleared after end_call",
          get_call_urgency(session)["n_turns"] == 0)

    # ── 7. latency, synchronous ──────────────────────────────────────────────
    print("\n7. Latency (warm, synchronous)")
    times = []
    for path, _ in clips:
        t0 = time.perf_counter()
        classify_urgency(path, SAMPLE_TRANSCRIPTS[0], "si-LK")
        times.append(time.perf_counter() - t0)
    mean_ms = 1000 * sum(times) / len(times)
    print(f"         mean {mean_ms:.0f} ms per turn "
          f"(min {1000*min(times):.0f}, max {1000*max(times):.0f})")
    check("under 5 s per turn", mean_ms < 5000,
          "acceptable inside the conversation loop")

    # ── 8. async path ────────────────────────────────────────────────────────
    # NOTE: classify_urgency_async is NOT what Component 2 or the demo call live
    # (see Section 16) -- kept here only as regression coverage for this
    # otherwise-unused function.
    print("\n8. Async path (legacy single-model function; see Section 16 for "
          "the path actually used live)")
    session = "async_session"
    reset_session(session)

    t0 = time.perf_counter()
    for (path, _), text in zip(clips, SAMPLE_TRANSCRIPTS):
        classify_urgency_async(path, text, "si-LK", session_id=session)
    dispatch_ms = 1000 * (time.perf_counter() - t0)
    print(f"         dispatch of {len(clips)} turns: {dispatch_ms:.0f} ms")
    check("async dispatch is effectively free", dispatch_ms < 200,
          "conversation loop is never blocked")

    t0 = time.perf_counter()
    final = end_call(session)
    drain_ms = 1000 * (time.perf_counter() - t0)
    print(f"         end_call drain: {drain_ms:.0f} ms")
    check("async produced all turns", final["n_turns"] == len(clips),
          f"n_turns={final['n_turns']}")
    check("async verdict is valid",
          final["label"] in {"High", "Medium", "Low"},
          f"{final['label']} (peak={final['peak_label']})")

    # ═══════════════════════════════════════════════════════════════════════
    # LATE FUSION - the path Component 2 and the standalone demo actually use
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 66)
    print("LATE FUSION (classify_urgency_late_fusion) - the live path")
    print("=" * 66)

    # ── 9. basic prediction ──────────────────────────────────────────────────
    print("\n9. Prediction")
    t0 = time.perf_counter()
    lf_result = classify_urgency_late_fusion(clips[0][0], SAMPLE_TRANSCRIPTS[0], "si-LK")
    elapsed = time.perf_counter() - t0
    check("returns a label", lf_result["label"] in {"High", "Medium", "Low"},
          f"{lf_result['label']} @ {lf_result['confidence']}")
    check("probabilities sum to 1",
          abs(sum(lf_result["probabilities"].values()) - 1.0) < 5e-4,
          f"sum={sum(lf_result['probabilities'].values()):.6f}")
    print(f"         probabilities: {lf_result['probabilities']}")

    # ── 10. path vs array ────────────────────────────────────────────────────
    print("\n10. Path input vs array input")
    arr = _load_audio(clips[0][0])
    lf_from_array = classify_urgency_late_fusion(arr, SAMPLE_TRANSCRIPTS[0], "si-LK")
    check("identical probabilities",
          lf_from_array["probabilities"] == lf_result["probabilities"])

    # ── 11. language mapping ─────────────────────────────────────────────────
    print("\n11. Language code mapping")
    lf_si = classify_urgency_late_fusion(clips[0][0], SAMPLE_TRANSCRIPTS[0], "si-LK")
    check("si-LK normalises to sinhala",
          lf_si["language_normalised"] == "sinhala")
    check("Sinhala context features fire", lf_si["context_score"] > 0,
          f"context_score={lf_si['context_score']}")

    # ── 12. the silent failure ───────────────────────────────────────────────
    print("\n12. Unmapped language code (the silent failure mode)")
    lf_bad = classify_urgency_late_fusion(clips[0][0], SAMPLE_TRANSCRIPTS[0], "sinhala-LK")
    check("unmapped code zeroes context", lf_bad["context_score"] == 0,
          "confirms why the mapping is load-bearing here too")
    check("but still returns a label", lf_bad["label"] is not None,
          f"{lf_bad['label']} - fails quietly, not loudly")

    # ── 13. short turn guard ─────────────────────────────────────────────────
    print("\n13. Short-turn guard")
    tiny = np.zeros(int(0.4 * 16000), dtype=np.float32)
    lf_skipped = classify_urgency_late_fusion(tiny, "ඔව්", "si-LK")
    check("sub-second turn skipped", lf_skipped["skipped"] is True,
          lf_skipped.get("reason", ""))

    # ── 14. session aggregation ──────────────────────────────────────────────
    print("\n14. Session aggregation")
    session = "test_session_latefusion"
    reset_session(session)
    for (path, _), text in zip(clips, SAMPLE_TRANSCRIPTS):
        r = classify_urgency_late_fusion(path, text, "si-LK", session_id=session)
        print(f"         turn {r.get('turn_index')}: {r['label']:<7} "
              f"conf={r['confidence']:.3f}  kw={r['keyword_count']:.1f}")

    lf_running = get_call_urgency(session)
    check("aggregates all turns", lf_running["n_turns"] == len(clips),
          f"n_turns={lf_running['n_turns']}")
    check("produces a call label", lf_running["label"] in {"High", "Medium", "Low"},
          f"{lf_running['label']} (peak={lf_running['peak_label']})")
    print(f"         turn labels: {lf_running['turn_labels']}")
    print(f"         call mean  : {lf_running['probabilities']}")

    lf_final = end_call(session)
    check("end_call returns the same verdict", lf_final["label"] == lf_running["label"])
    check("session cleared after end_call",
          get_call_urgency(session)["n_turns"] == 0)

    # ── 15. latency, synchronous ─────────────────────────────────────────────
    print("\n15. Latency (warm, synchronous)")
    lf_times = []
    for path, _ in clips:
        t0 = time.perf_counter()
        classify_urgency_late_fusion(path, SAMPLE_TRANSCRIPTS[0], "si-LK")
        lf_times.append(time.perf_counter() - t0)
    lf_mean_ms = 1000 * sum(lf_times) / len(lf_times)
    print(f"         mean {lf_mean_ms:.0f} ms per turn "
          f"(min {1000*min(lf_times):.0f}, max {1000*max(lf_times):.0f})")
    check("under 5 s per turn", lf_mean_ms < 5000,
          "acceptable inside the conversation loop")

    # ── 16. async path - what Component 2 and the demo actually call ────────
    print("\n16. Async path (classify_urgency_late_fusion_async - what Component 2 "
          "and the demo actually call)")
    session = "async_session_latefusion"
    reset_session(session)

    t0 = time.perf_counter()
    for (path, _), text in zip(clips, SAMPLE_TRANSCRIPTS):
        classify_urgency_late_fusion_async(path, text, "si-LK", session_id=session)
    lf_dispatch_ms = 1000 * (time.perf_counter() - t0)
    print(f"         dispatch of {len(clips)} turns: {lf_dispatch_ms:.0f} ms")
    check("async dispatch is effectively free", lf_dispatch_ms < 200,
          "conversation loop is never blocked")

    t0 = time.perf_counter()
    lf_async_final = end_call(session)
    lf_drain_ms = 1000 * (time.perf_counter() - t0)
    print(f"         end_call drain: {lf_drain_ms:.0f} ms")
    check("async produced all turns", lf_async_final["n_turns"] == len(clips),
          f"n_turns={lf_async_final['n_turns']}")
    check("async verdict is valid",
          lf_async_final["label"] in {"High", "Medium", "Low"},
          f"{lf_async_final['label']} (peak={lf_async_final['peak_label']})")

    print("\n" + "=" * 66)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: {failures}")
        return 1
    print("All checks passed, including the late-fusion path (classify_urgency_"
          "late_fusion / _async) that Component 2 and the demo actually call live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
