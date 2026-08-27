"""
keyword_discovery_v2.py
Rerun of the earlier PP1 TF-IDF keyword-discovery approach
(archive/scripts/09_keyword_discovery.py), but now using the CLEANER v2
transcripts (friend's fine-tuned ASR) instead of the original Whisper
transcripts - since v2 produces real, non-hallucinated Sinhala/Tamil/English
sentences even where v1 degenerated into repetition loops or empty output.

TF-IDF here is used ONLY as a discovery/ranking tool to surface candidate
words for the MANUAL keyword lexicon (URGENCY_KEYWORDS in
02_extract_features.py) - not as a live model feature. This avoids the
overfitting risk that killed word_count/compression_ratio as direct
features (spurious, domain-shifted, noisy on a 267/59-call corpus).

Uses BOTH the simulated corpus (data/features_v2.csv, labeled High/Medium/
Low) and DMC (reports/pp2/dmc_evaluation_v2.csv, also labeled) combined -
more real language coverage, especially for the kind of spontaneous distress
phrasing seen in DMC that a scripted simulated corpus alone might not
represent.

Candidates need native-speaker review before being added to the frozen
keyword list - a statistical score can't confirm real Sinhala/Tamil meaning.
"""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"

CURRENT_KEYWORDS = [
    "help", "fire", "inside", "trapped", "quickly", "emergency",
    "blood", "dying", "dead", "accident", "attack", "hurt", "injured",
    "crash", "ambulance", "police", "gas", "smoke", "burning", "flood",
    "workers", "grandmother", "children", "child", "baby", "unconscious",
    "breathing", "cannot breathe", "stuck", "still inside",
    "water", "flames", "fast", "save",
    "උදව්", "උදව් කරන්න", "බේරගන්න", "හදිස්සිය", "ගිනිගන්නව", "ගිනිඅරගෙන",
    "ගින්නක්", "ගිනි", "දුමක්", "දුම", "වතුර ඇතුළට", "ගෙදර ඇතුළට",
    "හිරවෙලා", "සිරවෙලා", "යටවෙලා", "ගංවතුරට", "ළමයා", "ළමයිය", "ළමේ",
    "හුස්ම ගන්නත් අමාරු", "හුස්ම", "ඉක්මනට", "ඉක්මනින්", "කරුණාකර", "දැන්",
    "உதவி", "உதவி செய்யுங்கள்", "உதவுங்கள்", "வெள்ளம்", "தீ", "இரத்தம்",
    "விபத்து", "தண்ணீர்", "குழந்தை", "புகை",
]

STOP_WORDS_EN = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "of", "and",
    "or", "but", "i", "my", "me", "we", "you", "he", "she", "they",
    "this", "that", "their", "there", "while", "with", "for", "from",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "than", "then", "so", "as", "if", "by", "up", "out", "about",
}


def main():
    sim_v2 = pd.read_csv(f"{ROOT}\\data\\features_v2.csv")
    dmc_v2 = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation_v2.csv")

    sim_v2 = sim_v2.rename(columns={"urgency_label": "label"})
    dmc_v2 = dmc_v2.rename(columns={"true_label": "label"})

    combined = pd.concat([
        sim_v2[["transcript", "label"]],
        dmc_v2[["transcript", "label"]],
    ], ignore_index=True)
    combined["transcript"] = combined["transcript"].fillna("").astype(str).str.lower()
    combined = combined[combined["transcript"].str.strip() != ""]

    print(f"Combined v2 corpus: {len(combined)} labeled transcripts "
          f"({len(sim_v2)} simulated + {len(dmc_v2)} DMC)")
    print(combined["label"].value_counts().to_dict())

    all_texts = combined["transcript"].tolist()
    all_labels = combined["label"].tolist()

    vectorizer = TfidfVectorizer(
        max_features=400,
        ngram_range=(1, 2),
        min_df=2,
        stop_words=list(STOP_WORDS_EN),
        token_pattern=r"(?u)\b[a-zA-Z஀-௿඀-෿]{2,}\b",
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    feature_names = vectorizer.get_feature_names_out()

    def class_mean(label):
        mask = [i for i, l in enumerate(all_labels) if l == label]
        if not mask:
            return np.zeros(len(feature_names))
        return np.asarray(tfidf_matrix[mask].mean(axis=0)).flatten()

    high_mean = class_mean("High")
    medium_mean = class_mean("Medium")
    low_mean = class_mean("Low")

    high_vs_rest = high_mean - (medium_mean + low_mean) / 2

    top_n = 40
    high_idx = np.argsort(high_vs_rest)[::-1][:top_n]
    high_keywords = [(feature_names[i], round(float(high_vs_rest[i]), 4)) for i in high_idx]

    print("\n" + "=" * 78)
    print("Top 40 HIGH-urgency discriminative words/bigrams (v2 corpus, sim+DMC)")
    print("=" * 78)
    new_candidates = []
    for word, score in high_keywords:
        already = word in CURRENT_KEYWORDS
        tag = "already in list" if already else "*** NEW CANDIDATE ***"
        print(f"  {word:<30} score={score:.4f}  {tag}")
        if not already and score > 0:
            new_candidates.append((word, score))

    print(f"\n{len(new_candidates)} new candidates found (score>0, not already in keyword list)")

    print("\n" + "=" * 78)
    print("NEW candidates only, for native-speaker review before adding to")
    print("URGENCY_KEYWORDS in scripts/pipeline/02_extract_features.py")
    print("=" * 78)
    for word, score in new_candidates:
        script_type = ("Sinhala" if re.search(r"[\u0D80-\u0DFF]", word)
                       else "Tamil" if re.search(r"[\u0B80-\u0BFF]", word)
                       else "English")
        print(f"  [{script_type:<8}] {word:<30} score={score:.4f}")

    out_path = f"{ROOT}\\reports\\pp2\\keyword_discovery_v2_candidates.csv"
    pd.DataFrame(new_candidates, columns=["word", "score"]).to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
