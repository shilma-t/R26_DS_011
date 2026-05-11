"""
09_keyword_discovery.py
Discovers urgency keywords from transcripts using TF-IDF.
Compares High vs Low urgency transcripts to find discriminative words.
Outputs: reports/discovered_keywords.csv, reports/keyword_tfidf.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

FEATURES_CSV = os.path.join("..", "features.csv")
REPORTS_DIR  = os.path.join("..", "reports")

# ── Current manual keyword list (for comparison) ─────────────────────────────
CURRENT_KEYWORDS = [
    "help", "emergency", "fire", "accident", "blood", "dying", "dead",
    "trapped", "attack", "knife", "gun", "hurt", "injured", "crash",
    "please", "ambulance", "police", "quickly", "fast", "now",
    "udaw", "udhaw", "hadisi", "gini", "ginnak", "wathura", "gangwathura",
    "maru", "wedi", "husma", "ikmanata", "beraganna", "awidaganna",
    "lamaya", "lamai", "athule", "yatawela",
    "thee", "puhai", "seekiram", "thanni", "wellam", "sikki",
    "udhawi", "meetpu", "thayawu",
]


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = pd.read_csv(FEATURES_CSV)

    if "transcript" not in df.columns:
        print("ERROR: No transcript column in features.csv. Run 02_extract_features.py first.")
        return

    df["transcript"] = df["transcript"].fillna("").str.lower()

    # Split by urgency
    high_texts   = df[df["urgency_label"] == "High"]["transcript"].tolist()
    medium_texts = df[df["urgency_label"] == "Medium"]["transcript"].tolist()
    low_texts    = df[df["urgency_label"] == "Low"]["transcript"].tolist()

    all_texts    = df["transcript"].tolist()
    all_labels   = df["urgency_label"].tolist()

    print(f"Transcripts: High={len(high_texts)}, Medium={len(medium_texts)}, Low={len(low_texts)}")
    print(f"Non-empty transcripts: {sum(1 for t in all_texts if t.strip())}")

    # ── TF-IDF on all transcripts ─────────────────────────────────────────────
    # Custom stop words — common words that appear everywhere, not urgency-specific
    # Keep emergency-relevant words like "no", "not", "now", "help" even if common
    STOP_WORDS = {
        "the", "a", "an", "is", "it", "in", "on", "at", "to", "of", "and",
        "or", "but", "i", "my", "me", "we", "you", "he", "she", "they",
        "this", "that", "their", "there", "while", "with", "for", "from",
        "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may",
        "than", "then", "so", "as", "if", "by", "up", "out", "about",
        # single characters and noise
        "m", "w", "s", "n", "r", "k", "p",
    }

    vectorizer = TfidfVectorizer(
        max_features=300,
        ngram_range=(1, 2),       # unigrams + bigrams
        min_df=2,                 # must appear in at least 2 docs
        stop_words=list(STOP_WORDS),
        token_pattern=r"(?u)\b[a-zA-Z஀-௿඀-෿]{2,}\b",
    )

    tfidf_matrix = vectorizer.fit_transform(all_texts)
    feature_names = vectorizer.get_feature_names_out()

    # ── Compute per-class average TF-IDF ─────────────────────────────────────
    def class_mean(label):
        mask = [i for i, l in enumerate(all_labels) if l == label]
        if not mask:
            return np.zeros(len(feature_names))
        return np.asarray(tfidf_matrix[mask].mean(axis=0)).flatten()

    high_mean   = class_mean("High")
    medium_mean = class_mean("Medium")
    low_mean    = class_mean("Low")

    # Discriminative score: High score - average of Medium+Low
    high_vs_rest  = high_mean - (medium_mean + low_mean) / 2
    low_vs_rest   = low_mean  - (high_mean  + medium_mean) / 2

    # ── Top keywords per class ────────────────────────────────────────────────
    top_n = 30
    high_idx = np.argsort(high_vs_rest)[::-1][:top_n]
    low_idx  = np.argsort(low_vs_rest)[::-1][:top_n]

    high_keywords = [(feature_names[i], round(float(high_vs_rest[i]), 4)) for i in high_idx]
    low_keywords  = [(feature_names[i], round(float(low_vs_rest[i]),  4)) for i in low_idx]

    print("\n── Top 30 HIGH urgency keywords (data-discovered) ──")
    for word, score in high_keywords:
        already = "✓ in list" if word in CURRENT_KEYWORDS else "★ NEW"
        print(f"  {word:<25} score={score:.4f}  {already}")

    print("\n── Top 30 LOW urgency keywords (data-discovered) ──")
    for word, score in low_keywords:
        print(f"  {word:<25} score={score:.4f}")

    # ── Check coverage of current keyword list ────────────────────────────────
    found_in_data = [kw for kw in CURRENT_KEYWORDS
                     if any(kw in t for t in all_texts if t.strip())]
    missing       = [kw for kw in CURRENT_KEYWORDS if kw not in found_in_data]

    print(f"\n── Current keyword list coverage ──")
    print(f"  Total manual keywords : {len(CURRENT_KEYWORDS)}")
    print(f"  Found in transcripts  : {len(found_in_data)}")
    print(f"  Never appeared        : {len(missing)}")
    if missing:
        print(f"  Missing: {missing}")

    # ── Save to CSV ───────────────────────────────────────────────────────────
    records = []
    for word, score in high_keywords:
        records.append({"word": word, "class": "High",   "score": score,
                        "in_current_list": word in CURRENT_KEYWORDS})
    for word, score in low_keywords:
        records.append({"word": word, "class": "Low",    "score": score,
                        "in_current_list": word in CURRENT_KEYWORDS})

    df_out = pd.DataFrame(records)
    df_out.to_csv(os.path.join(REPORTS_DIR, "discovered_keywords.csv"), index=False)
    print("\nSaved: reports/discovered_keywords.csv")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # High urgency keywords
    words_h  = [w for w, _ in high_keywords[:20]]
    scores_h = [s for _, s in high_keywords[:20]]
    colors_h = ["#c0392b" if w in CURRENT_KEYWORDS else "#e74c3c" for w in words_h]
    axes[0].barh(words_h[::-1], scores_h[::-1], color=colors_h[::-1])
    axes[0].set_title("Top 20 HIGH Urgency Keywords\n(dark = already in list, bright = newly discovered)",
                       fontsize=11, fontweight="bold")
    axes[0].set_xlabel("TF-IDF Discriminative Score")
    axes[0].axvline(0, color="white", linewidth=0.5)

    # Low urgency keywords
    words_l  = [w for w, _ in low_keywords[:20]]
    scores_l = [s for _, s in low_keywords[:20]]
    axes[1].barh(words_l[::-1], scores_l[::-1], color="#2ecc71")
    axes[1].set_title("Top 20 LOW Urgency Keywords\n(words that signal non-emergency)",
                       fontsize=11, fontweight="bold")
    axes[1].set_xlabel("TF-IDF Discriminative Score")

    fig.patch.set_facecolor("#1a1d2e")
    for ax in axes:
        ax.set_facecolor("#13151f")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2e3250")

    plt.suptitle("Data-Discovered Urgency Keywords via TF-IDF\n(vs manually curated list)",
                 fontsize=13, color="white", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "keyword_tfidf.png"),
                dpi=150, bbox_inches="tight", facecolor="#1a1d2e")
    plt.close()
    print("Saved: reports/keyword_tfidf.png")

    # ── Print updated keyword list suggestion ─────────────────────────────────
    new_words = [w for w, _ in high_keywords if w not in CURRENT_KEYWORDS]
    print(f"\n── {len(new_words)} newly discovered HIGH urgency keywords to consider adding ──")
    print(new_words)


if __name__ == "__main__":
    main()
