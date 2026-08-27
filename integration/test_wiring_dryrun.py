"""
test_wiring_dryrun.py
Exercise the live_mic_demo wiring without a microphone.

Simulates what the demo does per turn - hand a WAV path, a transcript and a
Component 2 language code to the bridge asynchronously - then finalises the
call and confirms the verdict is shaped correctly for the final emergency JSON.

Catches integration mistakes that a syntax check cannot: a wrong import path,
the wrong language code reaching the classifier, or a verdict that cannot be
serialised into the JSON the dispatch queue reads.

Usage:
  python integration/test_wiring_dryrun.py
"""

import os
import sys
import json
import time

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
INTEGRATION = os.path.join(ROOT, "integration")
BACKEND = r"C:\Users\shafr\Desktop\component 2\backend"

if INTEGRATION not in sys.path:
    sys.path.insert(0, INTEGRATION)

from urgency_bridge import (  # noqa: E402
    classify_urgency_async,
    end_call,
    warmup,
)

SCRATCH = os.path.join(INTEGRATION, "_test_turns")

TURNS = [
    ("turn1.wav", "උදව් කරන්න, ගෙදර ගිනිගන්නවා, ළමයා ඇතුළේ ඉන්නවා", "si-LK"),
    ("turn2.wav", "හිරවෙලා ඉන්නවා, වතුර ඇතුළට ආවා, ඉක්මනට එන්න", "si-LK"),
]

failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def main():
    print("=" * 66)
    print("LIVE DEMO WIRING - DRY RUN (no microphone)")
    print("=" * 66)

    # ── 1. the demo's import path resolves ───────────────────────────────────
    print("\n1. Import path used by live_mic_demo")
    check("integration dir exists", os.path.isdir(INTEGRATION), INTEGRATION)
    check("bridge importable from it",
          os.path.exists(os.path.join(INTEGRATION, "urgency_bridge.py")))

    demo = os.path.join(BACKEND, "live_mic_demo.py")
    check("live_mic_demo.py present", os.path.exists(demo))
    if os.path.exists(demo):
        src = open(demo, encoding="utf-8").read()
        check("demo points at the integration dir",
              "URGENCY_INTEGRATION" in src and "integration" in src)
        check("demo dispatches asynchronously",
              "classify_urgency_async" in src)
        check("demo finalises on call end", "finalise_urgency" in src)
        check("demo writes urgency into final JSON",
              "urgency_level" in src)

    # ── 2. warmup ────────────────────────────────────────────────────────────
    print("\n2. Warmup (demo does this before the first turn)")
    t0 = time.perf_counter()
    warmup()
    print(f"         {time.perf_counter() - t0:.1f}s")

    missing = [f for f, _, _ in TURNS
               if not os.path.exists(os.path.join(SCRATCH, f))]
    if missing:
        raise SystemExit(
            f"Test turn audio missing: {missing}\n"
            "Run integration/test_bridge_offline.py first to create it."
        )

    # ── 3. per-turn dispatch, exactly as the demo does it ────────────────────
    print("\n3. Per-turn async dispatch")
    session = "dryrun_session"
    dispatch_times = []
    for fname, transcript, language in TURNS:
        path = os.path.join(SCRATCH, fname)
        t0 = time.perf_counter()
        classify_urgency_async(
            audio=path,
            transcript=transcript,
            language=language,
            session_id=session,
        )
        ms = 1000 * (time.perf_counter() - t0)
        dispatch_times.append(ms)
        print(f"         {fname}: dispatched in {ms:.0f} ms")

    check("dispatch never blocks the call", max(dispatch_times) < 200,
          f"max {max(dispatch_times):.0f} ms")

    # ── 4. finalise ──────────────────────────────────────────────────────────
    print("\n4. Call end")
    t0 = time.perf_counter()
    verdict = end_call(session)
    print(f"         drain + aggregate: {time.perf_counter() - t0:.1f}s")

    check("verdict has a label", verdict.get("label") in {"High", "Medium", "Low"},
          f"{verdict.get('label')}")
    check("all turns classified", verdict.get("n_turns") == len(TURNS),
          f"n_turns={verdict.get('n_turns')}")
    check("peak severity reported", verdict.get("peak_label") is not None,
          f"peak={verdict.get('peak_label')}")

    # ── 5. JSON serialisability ──────────────────────────────────────────────
    print("\n5. Final emergency JSON payload")
    final = {
        "session_id": session,
        "status": "READY",
        "incident_type": "fire",
        "urgency_level": verdict["label"],
        "urgency_confidence": verdict["confidence"],
        "urgency_detail": verdict,
    }
    try:
        blob = json.dumps(final, ensure_ascii=False, indent=2)
        ok = True
    except TypeError as exc:
        ok = False
        blob = str(exc)
    check("payload is JSON-serialisable", ok,
          "" if ok else "numpy types must be cast before writing")

    if ok:
        print("\n         what the dispatch queue receives:")
        for line in blob.splitlines()[:14]:
            print("         " + line)

    print("\n" + "=" * 66)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED: {failures}")
        return 1
    print("Wiring verified. Ready for a live microphone test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
