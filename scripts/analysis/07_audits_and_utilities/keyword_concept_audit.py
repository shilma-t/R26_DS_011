"""
keyword_concept_audit.py
Group keywords into canonical concepts and audit match quality.

For each concept:
  - How many calls matched (B=original, C=expanded)
  - Match rate per urgency class (High / Medium / Low)
  - Match rate per language
  - Whether matches appear on bad-ASR calls
  - Class separation: does High match more than Medium and Low?

Outputs:
  reports/pp2/keyword_concept_audit.csv      — per concept summary
  reports/pp2/keyword_concept_matches.csv    — per call match detail
"""

import os
import pandas as pd
import numpy as np

_ROOT    = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
FEATURES = os.path.join(_ROOT, "data", "features.csv")
REPORTS  = os.path.join(_ROOT, "reports", "pp2")

# ── Concept groups ────────────────────────────────────────────────────────────
# Each concept has two lists: original (B) keywords and expanded (C) additions

CONCEPTS = {
    "HELP": {
        "B": ["help", "udaw", "உதவி", "உதவுங்கள்", "please"],
        "C_extra": ["உதவி செய்யுங்கள்", "உதவுங்கள்", "உதவி",
                    "உதவி செய்யுங்கள்", "බේරගන්න", "උදව්", "උදව් කරන්න"],
    },
    "FIRE": {
        "B": ["fire", "gini", "தீ"],
        "C_extra": ["burning", "flames", "smoke", "gas",
                    "ගිනිගන්නව", "ගිනිඅරගෙන", "ගින්නක්", "ගිනි",
                    "දුමක්", "දුම", "புகை"],
    },
    "FLOOD_WATER": {
        "B": ["flood", "water"],
        "C_extra": ["வெள்ளம்", "தண்ணீர்", "வெள்ளம்",
                    "වතුර ඇතුළට", "ගෙදර ඇතුළට", "ගංවතුරට", "යටවෙලා",
                    "ගංවතුරට"],
    },
    "TRAPPED": {
        "B": ["trapped"],
        "C_extra": ["stuck", "still inside", "inside",
                    "හිරවෙලා", "සිරවෙලා", "යටවෙලා"],
    },
    "INJURY_MEDICAL": {
        "B": ["blood", "dying", "dead", "hurt", "injured",
              "ambulance", "இரத்தம்"],
        "C_extra": ["unconscious", "breathing", "cannot breathe",
                    "හුස්ම ගන්නත් අමාරු", "හුස්ම"],
    },
    "ACCIDENT_ATTACK": {
        "B": ["accident", "attack", "crash", "knife", "gun",
              "விபத்து"],
        "C_extra": [],
    },
    "CHILD_AT_RISK": {
        "B": [],
        "C_extra": ["children", "child", "baby", "grandmother",
                    "குழந்தை", "ළමයා", "ළමයිය", "ළමේ"],
    },
    "IMMEDIACY": {
        "B": ["quickly", "fast", "now"],
        "C_extra": ["ඉක්මනට", "ඉක්මනින්", "කරුණාකර", "දැන්"],
    },
    "EMERGENCY_SERVICES": {
        "B": ["police", "emergency"],
        "C_extra": [],
    },
    "WORKERS_OTHERS": {
        "B": [],
        "C_extra": ["workers", "save"],
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def match_keywords(text, keywords):
    if not isinstance(text, str):
        return []
    t = text.lower()
    return [kw for kw in keywords if kw.lower() in t]


def concept_matches(text, concept_dict, include_extra=True):
    kws = concept_dict["B"][:]
    if include_extra:
        kws += concept_dict["C_extra"]
    return match_keywords(text, kws)


def class_separation(df_concept, col="matched_B"):
    h = df_concept[df_concept["urgency_label"] == "High"][col].mean()
    m = df_concept[df_concept["urgency_label"] == "Medium"][col].mean()
    l = df_concept[df_concept["urgency_label"] == "Low"][col].mean()
    return h, m, l, round(max(h, m, l) - min(h, m, l), 3)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(FEATURES)
    n  = len(df)

    print("=" * 70)
    print("KEYWORD CONCEPT AUDIT")
    print(f"Dataset: {n} calls")
    print("=" * 70)

    # Per-call match table
    call_rows = []
    for _, row in df.iterrows():
        transcript = str(row.get("transcript", ""))
        asr_q      = row.get("asr_quality", 1.0)
        for concept, kws in CONCEPTS.items():
            b_matches = match_keywords(transcript, kws["B"])
            c_matches = match_keywords(transcript, kws["B"] + kws["C_extra"])
            call_rows.append({
                "filename":      row["filename"],
                "language":      row.get("language", ""),
                "urgency_label": row["urgency_label"],
                "asr_quality":   round(asr_q, 3),
                "concept":       concept,
                "matched_B":     int(len(b_matches) > 0),
                "matched_C":     int(len(c_matches) > 0),
                "C_only":        int(len(c_matches) > 0 and len(b_matches) == 0),
                "b_terms":       ", ".join(b_matches[:3]),
                "c_terms":       ", ".join(c_matches[:3]),
            })

    cdf = pd.DataFrame(call_rows)

    # Per-concept summary
    summary_rows = []
    print()
    for concept in CONCEPTS:
        sub = cdf[cdf["concept"] == concept]

        n_B     = sub["matched_B"].sum()
        n_C     = sub["matched_C"].sum()
        n_Conly = sub["C_only"].sum()

        h_B, m_B, l_B, sep_B = class_separation(sub, "matched_B")
        h_C, m_C, l_C, sep_C = class_separation(sub, "matched_C")

        # Bad ASR matches
        bad_B = sub[(sub["matched_B"] == 1) & (sub["asr_quality"] < 0.3)].shape[0]
        bad_C = sub[(sub["matched_C"] == 1) & (sub["asr_quality"] < 0.3)].shape[0]

        # Language breakdown
        lang_B = sub[sub["matched_B"] == 1]["language"].value_counts().to_dict()
        lang_C = sub[sub["C_only"]   == 1]["language"].value_counts().to_dict()

        # Does C add genuine High signal or add noise?
        if n_Conly > 0:
            c_only_sub = sub[sub["C_only"] == 1]
            c_high_pct = round(
                100 * (c_only_sub["urgency_label"] == "High").sum() / len(c_only_sub), 1
            )
        else:
            c_high_pct = float("nan")

        # Verdict
        if n_B == 0 and n_C == 0:
            v = "never fires"
        elif sep_B < 0.05 and sep_C < 0.05:
            v = "no class separation"
        elif h_B > m_B and h_B > l_B:
            v = "useful — High-associated"
        elif m_B > h_B or l_B > h_B:
            v = "check — Medium/Low fires more than High"
        else:
            v = "weak"

        summary_rows.append({
            "concept":        concept,
            "n_B_matches":    n_B,
            "n_C_matches":    n_C,
            "n_C_only_extra": n_Conly,
            "sep_B":          sep_B,
            "sep_C":          sep_C,
            "High_rate_B":    round(h_B, 3),
            "Medium_rate_B":  round(m_B, 3),
            "Low_rate_B":     round(l_B, 3),
            "High_rate_C":    round(h_C, 3),
            "Medium_rate_C":  round(m_C, 3),
            "Low_rate_C":     round(l_C, 3),
            "C_extra_high_pct": c_high_pct,
            "bad_asr_B":      bad_B,
            "bad_asr_C":      bad_C,
            "lang_B":         str(lang_B),
            "lang_C_extra":   str(lang_C),
            "verdict":        v,
        })

        print(f"  [{concept}]")
        print(f"    B matches: {n_B}   C matches: {n_C}   C-only extra: {n_Conly}")
        print(f"    Class rates B — High={h_B:.3f}  Medium={m_B:.3f}  Low={l_B:.3f}  sep={sep_B:.3f}")
        print(f"    Class rates C — High={h_C:.3f}  Medium={m_C:.3f}  Low={l_C:.3f}  sep={sep_C:.3f}")
        if n_Conly > 0:
            print(f"    C-only matches: {c_high_pct}% are High urgency")
        print(f"    Bad-ASR matches: B={bad_B}  C={bad_C}")
        print(f"    Lang B: {lang_B}   Lang C-extra: {lang_C}")
        print(f"    VERDICT: {v}")
        print()

    sdf = pd.DataFrame(summary_rows)

    print("=" * 70)
    print("SUMMARY SORTED BY CLASS SEPARATION (B)")
    print("=" * 70)
    print(sdf[["concept","n_B_matches","n_C_matches","n_C_only_extra",
               "sep_B","sep_C","C_extra_high_pct","verdict"]]
          .sort_values("sep_B", ascending=False)
          .to_string(index=False))

    # Save
    sdf.to_csv(os.path.join(REPORTS, "keyword_concept_audit.csv"),
               index=False, encoding="utf-8-sig")
    cdf.to_csv(os.path.join(REPORTS, "keyword_concept_matches.csv"),
               index=False, encoding="utf-8-sig")
    print(f"\nSaved → reports/pp2/keyword_concept_audit.csv")
    print(f"Saved → reports/pp2/keyword_concept_matches.csv")


if __name__ == "__main__":
    main()
