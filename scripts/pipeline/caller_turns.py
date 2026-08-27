"""
caller_turns.py
Reconstruct caller TURNS from diarization segments.

Diarization splits on every pause, producing fragments. A turn is what the
caller says in one go before the operator responds - the same unit the live
system processes in her process_text_turn().

Consecutive caller segments separated by less than MERGE_GAP seconds are
merged. Turns shorter than MIN_TURN seconds are dropped: 91 of the 111
features are acoustic, and MFCC statistics over a second of audio are noise.

Used by evaluate_dmc.py for the turn-level condition, and by urgency_bridge.py
at inference time so evaluation and deployment segment identically.
"""

from __future__ import annotations

import os

import pandas as pd

_ROOT     = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
REPORTS   = os.path.join(_ROOT, "reports", "pp2")
MANIFEST  = os.path.join(REPORTS, "dmc_diarization_manifest.csv")
WORKSHEET = os.path.join(REPORTS, "dmc_verify_worksheet.xlsx")

MERGE_GAP = 1.0   # seconds; shorter gaps are within-turn pauses
MIN_TURN  = 2.0   # seconds; below this, acoustic features are unreliable

CALLER_VALUES = {"yes", "yes-mixed"}


def normalise(value) -> str:
    """Fold label spelling variants: 'Yes/Mixed' -> 'yes-mixed'."""
    return str(value or "").strip().lower().replace("/", "-").replace(" ", "")


def load_caller_map(worksheet: str = WORKSHEET) -> dict:
    """recording_id -> set of speaker labels the human verified as caller."""
    ws = pd.read_excel(worksheet, dtype=str)
    ws = ws[ws["speaker"].notna()
            & ws["speaker"].astype(str).str.startswith("SPEAKER")].copy()
    ws["norm"] = ws["is_caller"].map(normalise)

    mapping = {}
    for rec_id, group in ws.groupby("recording_id"):
        mapping[str(rec_id)] = {
            row.speaker for row in group.itertuples() if row.norm in CALLER_VALUES
        }
    return mapping


def merge_to_turns(intervals, merge_gap=MERGE_GAP, min_turn=MIN_TURN):
    """
    [(start, end), ...] -> [(start, end), ...] merged into turns.

    Overlapping intervals are unioned, which matters for the 10 calls where
    the caller was split across two diarization labels.
    """
    if not intervals:
        return []

    ordered = sorted(intervals, key=lambda s: s[0])
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(s, e) for s, e in merged if (e - s) >= min_turn]


def caller_turns_for_call(call_df, caller_speakers,
                          merge_gap=MERGE_GAP, min_turn=MIN_TURN):
    """Turn boundaries for one call, given its manifest rows and caller labels."""
    segs = call_df[call_df["speaker"].isin(caller_speakers)]
    return merge_to_turns(
        [(r.start, r.end) for r in segs.itertuples()],
        merge_gap=merge_gap,
        min_turn=min_turn,
    )


def load_manifest(path: str = MANIFEST) -> pd.DataFrame:
    man = pd.read_csv(path, dtype={"recording_id": str, "filename": str})
    man = man[man["speaker"].notna() & (man["speaker"] != "ERROR")]
    for col in ("start", "end", "duration"):
        man[col] = pd.to_numeric(man[col], errors="coerce")
    return man.dropna(subset=["start", "end", "duration"])


def build_all_turns(manifest_path=MANIFEST, worksheet_path=WORKSHEET,
                    merge_gap=MERGE_GAP, min_turn=MIN_TURN) -> pd.DataFrame:
    """One row per caller turn across every call."""
    man = load_manifest(manifest_path)
    caller_map = load_caller_map(worksheet_path)

    rows = []
    for rec_id, call in man.groupby("recording_id", sort=False):
        rec_id = str(rec_id)
        speakers = caller_map.get(rec_id, set())
        if not speakers:
            continue

        turns = caller_turns_for_call(call, speakers, merge_gap, min_turn)
        for index, (start, end) in enumerate(turns, start=1):
            rows.append({
                "recording_id": rec_id,
                "filename":     call["filename"].iloc[0],
                "turn_index":   index,
                "start":        round(start, 3),
                "end":          round(end, 3),
                "duration":     round(end - start, 3),
            })

    return pd.DataFrame(rows)


def main():
    turns = build_all_turns()

    if turns.empty:
        raise SystemExit("No turns produced - check the worksheet labels.")

    per_call = turns.groupby("recording_id").agg(
        turns=("turn_index", "max"),
        speech=("duration", "sum"),
    )

    print("=" * 60)
    print(f"CALLER TURNS  (merge gap {MERGE_GAP}s, min turn {MIN_TURN}s)")
    print("=" * 60)
    print(f"calls              : {turns['recording_id'].nunique()}")
    print(f"turns              : {len(turns)}")
    print(f"turns per call     : median {per_call['turns'].median():.0f}, "
          f"min {per_call['turns'].min()}, max {per_call['turns'].max()}")
    print()
    print("turn duration (s):")
    print(turns["duration"].describe().round(1).to_string())
    print()
    print(f"total turn speech  : {turns['duration'].sum() / 60:.1f} min")

    # Compare against training clip length - the reason turns exist at all
    labels = os.path.join(_ROOT, "data", "labels_frozen_20260817.csv")
    if os.path.exists(labels):
        sim = pd.to_numeric(
            pd.read_csv(labels)["duration_sec"], errors="coerce"
        ).dropna()
        print()
        print(f"simulated training clips: median {sim.median():.1f}s")
        print(f"caller turns            : median {turns['duration'].median():.1f}s")

    thin = per_call[per_call["turns"] < 2]
    if len(thin):
        print(f"\ncalls with fewer than 2 turns ({len(thin)}):")
        for rec_id, row in thin.iterrows():
            print(f"  {rec_id:<34} {int(row['turns'])} turn(s), {row['speech']:.1f}s")

    out = os.path.join(REPORTS, "dmc_caller_turns.csv")
    turns.to_csv(out, index=False)
    print(f"\nSaved -> reports/pp2/dmc_caller_turns.csv")


if __name__ == "__main__":
    main()
