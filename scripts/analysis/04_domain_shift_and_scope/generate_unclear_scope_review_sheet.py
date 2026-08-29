"""
generate_unclear_scope_review_sheet.py
Produces a plain review sheet of every DMC call the keyword-based scope
categorizer (see dmc_scope_matched_evaluation.py / full_metrics_sim_and_dmc_
scopematched.py) currently marks "unclear" -- neither confidently in-scope
(fire/flood) nor confidently out-of-scope. About half of the 59 DMC calls
land here, which is why the scope-matched subset is only n=11.

This script makes NO judgment calls itself -- it only extracts the relevant
rows and both available transcripts (this project's own ASR, and the
teammate's ASR used for the earlier root-cause work) side by side, plus a
blank column for a native Sinhala/Tamil speaker to mark whether each call is
actually a fire/flood emergency or not. Nothing here changes any model,
feature, or evaluation script -- it only grows the labeled subset those
scripts already read from, once the review column is filled in.

Output: reports/pp2_finalized_reports/scope_review_unclear_calls.csv
"""
import pandas as pd

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"

TYPE_PATTERNS = {
    "fire": ["ගිනි", "ගින්", "දුම", "எரிச்சல்", "தீ", "புகை", "fire", "smoke", "burning"],
    "flood_water": ["වතුර", "ගංවතුර", "යටවෙලා", "හිරවෙලා", "සිරවෙලා", "කොටුවෙලා",
                    "வெள்ளம்", "தண்ணீர்", "flood", "water"],
    "tree_structural": ["ගහක්", "ගස්", "කඩන්නක්", "කඩන්නාව", "වැටිලා", "පුළුගහක්", "tree"],
    "tsunami_wind": ["සුනාමි", "සුළං", "සුළි", "tsunami"],
    "road_travel": ["පාර", "රූට්", "යන්න පුළුවන්", "route", "road", "kolamba", "කොළඹ"],
    "medical_rescue": ["හොස්පිට්ල්", "රෝහල", "දොක්තර", "வைத்தியர்", "hospital", "boat",
                       "බෝට්", "රැකවරණ", "රැකගන්න"],
    "status_no_impact": ["බලපෑමක් නැහැ", "අවදානමක් නැහ", "no impact"],
}
FLOOD_FIRE_CATEGORIES = {"fire", "flood_water", "flood_water+road_travel", "fire+flood_water"}


def categorize(text):
    if not isinstance(text, str) or not text.strip():
        return "empty"
    hits = [cat for cat, patterns in TYPE_PATTERNS.items() if any(p in text for p in patterns)]
    return "+".join(hits) if hits else "unclear"


def main():
    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    dmc_v2 = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation_v2.csv")
    dmc = dmc.reset_index(drop=True)
    dmc_v2 = dmc_v2.reset_index(drop=True)

    category = dmc_v2["transcript"].map(categorize)
    in_scope_now = category.isin(FLOOD_FIRE_CATEGORIES)

    sheet = pd.DataFrame({
        "recording_id": dmc["recording_id"],
        "language": dmc["language"],
        "true_urgency_label": dmc["true_label"],
        "current_category": category,
        "transcript": dmc["transcript"],
        "transcript_teammate_asr": dmc_v2["transcript"],
        # Blank for the reviewer. Expected values: "fire", "flood", "both",
        # "neither" (not actually fire/flood -- confirms current exclusion),
        # or "cannot tell" (ASR too garbled to judge either way).
        "reviewer_verdict": "",
        "reviewer_notes": "",
    })

    unclear = sheet[~in_scope_now].reset_index(drop=True)
    already_in_scope = sheet[in_scope_now].reset_index(drop=True)

    out_path = f"{ROOT}\\reports\\pp2_finalized_reports\\scope_review_unclear_calls.csv"
    unclear.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Total DMC calls: {len(sheet)}")
    print(f"Already in scope-matched set (n={len(already_in_scope)}): "
          f"{sorted(already_in_scope['recording_id'].tolist())}")
    print(f"Needs review (n={len(unclear)}): written to\n  {out_path}")
    print(f"\nCategory breakdown of the ones needing review:")
    print(unclear["current_category"].value_counts().to_string())
    print(f"\nTrue-label breakdown of the ones needing review "
          f"(shows what's at stake if some turn out in-scope):")
    print(unclear["true_urgency_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
