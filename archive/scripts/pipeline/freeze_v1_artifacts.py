"""
freeze_v1_artifacts.py
Copy every artifact the v1 result depends on into frozen_v1/ and record a
SHA-256 for each, so the baseline stays reproducible no matter what v2 does.

Nothing in this project should ever write into frozen_v1/. Every v2 artifact
takes a NEW filename rather than overwriting a v1 one.

Re-running is safe: existing copies are verified against their recorded hash
rather than silently replaced, and any mismatch is reported loudly.

Usage:
  python scripts/pipeline/freeze_v1_artifacts.py
"""

import os
import json
import shutil
import hashlib
from datetime import datetime

_ROOT  = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
FROZEN = os.path.join(_ROOT, "frozen_v1")

# Everything the v1 number depends on. Paths are relative to project root.
ARTIFACTS = [
    "data/features.csv",
    "data/labels_frozen_20260817.csv",
    "data/dmc_metadata.csv",
    "reports/pp2/candidate_pipeline_freeze.txt",
    "reports/pp2/ablation_manifest.txt",
    "reports/pp2/dmc_diarization_manifest.csv",
    "reports/pp2/dmc_verify_worksheet.xlsx",
    "reports/pp2/dmc_caller_manifest.csv",
    "reports/pp2/dmc_caller_turns.csv",
    "scripts/pipeline/02_extract_features.py",
    "scripts/pipeline/context_features.py",
    "scripts/pipeline/evaluate_dmc.py",
    "scripts/pipeline/caller_turns.py",
    "scripts/pipeline/extract_caller_audio.py",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(FROZEN, exist_ok=True)
    record_path = os.path.join(FROZEN, "MANIFEST.json")

    previous = {}
    if os.path.exists(record_path):
        with open(record_path, encoding="utf-8") as fh:
            previous = json.load(fh).get("artifacts", {})

    record = {}
    copied = skipped = missing = changed = 0

    for rel in ARTIFACTS:
        src = os.path.join(_ROOT, rel.replace("/", os.sep))
        if not os.path.exists(src):
            print(f"  MISSING   {rel}")
            missing += 1
            continue

        digest = sha256(src)
        flat = rel.replace("/", "__")
        dst = os.path.join(FROZEN, flat)

        if os.path.exists(dst):
            # Already frozen - verify rather than overwrite
            old = previous.get(rel, {}).get("sha256")
            if old and old != digest:
                print(f"  CHANGED   {rel}")
                print(f"            frozen copy kept; live file now differs")
                changed += 1
            else:
                skipped += 1
        else:
            shutil.copy2(src, dst)
            print(f"  frozen    {rel}")
            copied += 1

        record[rel] = {
            "sha256": digest,
            "bytes": os.path.getsize(src),
            "frozen_as": flat,
        }

    with open(record_path, "w", encoding="utf-8") as fh:
        json.dump({
            "created": previous and None or datetime.now().isoformat(timespec="seconds"),
            "updated": datetime.now().isoformat(timespec="seconds"),
            "note": "v1 baseline artifacts. Do not modify. v2 uses new filenames.",
            "artifacts": record,
        }, fh, indent=2)

    print()
    print("=" * 58)
    print(f"frozen_v1/  {copied} copied, {skipped} already present, "
          f"{missing} missing, {changed} changed since freeze")
    print(f"manifest -> frozen_v1/MANIFEST.json")
    print("=" * 58)
    if changed:
        print("\nWARNING: live files differ from their frozen copies.")
        print("The frozen copies were NOT overwritten. Investigate before")
        print("treating any new result as comparable to the v1 baseline.")


if __name__ == "__main__":
    main()
