"""
acoustic_domain_shift_deep_dive_flood25.py
Diagnostic-only deep dive into why the acoustic model underperforms on real
DMC calls, specifically against the authoritative flood-only n=25 subset
(data/dmc_metadata.csv disaster_type == "Flood"). Reuses the exact Cohen's d
formula from domain_shift_diagnostic.py and the exact acoustic feature list /
RF training recipe used everywhere else in this project.

Does NOT retrain the deployed model, does NOT tune on DMC, does NOT change
the feature set or labels. Every RF fit here is throwaway, in-memory, for
diagnostic feature-importance readout only -- same recipe as the frozen
acoustic-only model, not a replacement for it.

Covers:
  2. Per-feature Cohen's d, sim vs DMC-flood-25, all 91 acoustic features, ranked
  3. RF feature importance (mean over 5 seeds) x domain shift -> suspicious features
  4. MFCC-specific breakdown of (2)
  5. Per-call error analysis on the acoustic-only model, DMC-flood-25
  6. PCA of the 91 acoustic features, sim vs DMC-flood-25 (visualization only)
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = r"C:\Users\shafr\Desktop\Uni\RESEARCH\R26_DS_011"
RF_BASE = dict(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1)
SEEDS = [1, 7, 42, 99, 123]

ACOUSTIC = ([f"mfcc_{i}_{s}" for i in range(1, 41) for s in ("mean", "std")] +
    ["pitch_mean", "pitch_std", "rms_mean", "rms_std", "zcr_mean",
     "spec_centroid_mean", "speaking_rate", "jitter",
     "rms_slope", "rms_gradient", "speaking_rate_change"])
assert len(ACOUSTIC) == 91


def cohens_d(a, b):
    """Identical formula to domain_shift_diagnostic.py. |d|>0.8 is a large shift."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                     / max(na + nb - 2, 1))
    if pooled == 0:
        return 0.0 if a.mean() == b.mean() else np.inf
    return (b.mean() - a.mean()) / pooled


def main():
    le = LabelEncoder().fit(["High", "Low", "Medium"])
    high_idx = list(le.classes_).index("High")

    sim = pd.read_csv(f"{ROOT}\\data\\features.csv")
    y_sim = le.transform(sim["urgency_label"].values)
    X_sim = sim[ACOUSTIC].fillna(0).values

    meta = pd.read_csv(f"{ROOT}\\data\\dmc_metadata.csv")
    dmc = pd.read_csv(f"{ROOT}\\reports\\pp2\\dmc_evaluation.csv")
    dmc = dmc.merge(meta[["recording_id", "disaster_type"]], on="recording_id", how="inner")
    dmc_flood = dmc[dmc["disaster_type"] == "Flood"].reset_index(drop=True)
    y_dmc = le.transform(dmc_flood["true_label"].values)
    X_dmc = dmc_flood[ACOUSTIC].fillna(0).values
    print(f"Simulated n={len(sim)}, DMC flood-25 n={len(dmc_flood)}\n")

    # ================= ANALYSIS 2: per-feature Cohen's d =================
    print("=" * 100)
    print("ANALYSIS 2: acoustic feature domain shift, sim vs DMC-flood-25 (Cohen's d)")
    print("=" * 100)
    rows = []
    for feat in ACOUSTIC:
        a = sim[feat].values.astype(float)
        b = dmc_flood[feat].values.astype(float)
        d = cohens_d(a, b)
        rows.append({
            "feature": feat, "sim_mean": np.nanmean(a), "sim_std": np.nanstd(a, ddof=1),
            "dmc_mean": np.nanmean(b), "dmc_std": np.nanstd(b, ddof=1),
            "sim_median": np.nanmedian(a), "dmc_median": np.nanmedian(b),
            "cohens_d": d, "abs_d": abs(d),
        })
    shift = pd.DataFrame(rows).sort_values("abs_d", ascending=False).reset_index(drop=True)
    shift["rank"] = np.arange(1, len(shift) + 1)

    pd.set_option("display.width", 160)
    print("\nTop 20 most shifted acoustic features:")
    print(shift[["rank", "feature", "sim_mean", "dmc_mean", "sim_std", "dmc_std", "cohens_d"]]
          .head(20).round(4).to_string(index=False))

    n_large = (shift["abs_d"] >= 0.8).sum()
    n_medium = ((shift["abs_d"] >= 0.5) & (shift["abs_d"] < 0.8)).sum()
    n_small = (shift["abs_d"] < 0.5).sum()
    print(f"\nShift magnitude summary (91 features):")
    print(f"  |d| >= 0.8 (large)      : {n_large}")
    print(f"  0.5 <= |d| < 0.8 (med)  : {n_medium}")
    print(f"  |d| < 0.5 (small)       : {n_small}")
    print(f"  median |d| across all 91: {shift['abs_d'].median():.3f}")
    print(f"  mean |d| across all 91  : {shift['abs_d'].mean():.3f}")

    out_path = f"{ROOT}\\reports\\pp2_finalized_reports\\acoustic_domain_shift_flood25_full.csv"
    shift.to_csv(out_path, index=False)
    print(f"\nFull 91-feature table written to {out_path}")

    # ================= ANALYSIS 3: RF feature importance x shift =================
    print("\n" + "=" * 100)
    print("ANALYSIS 3: RF feature importance (5-seed mean, diagnostic fit only) x domain shift")
    print("=" * 100)
    importances = []
    for seed in SEEDS:
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=seed)),
            ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
        ])
        pipe.fit(X_sim, y_sim)
        importances.append(pipe.named_steps["clf"].feature_importances_)
    mean_importance = np.mean(importances, axis=0)
    imp_df = pd.DataFrame({"feature": ACOUSTIC, "rf_importance": mean_importance})
    combined = shift.merge(imp_df, on="feature")
    combined["importance_rank"] = combined["rf_importance"].rank(ascending=False).astype(int)
    combined["suspicious_score"] = combined["rf_importance"] * combined["abs_d"]
    combined = combined.sort_values("suspicious_score", ascending=False).reset_index(drop=True)

    print("\nTop 20 by (importance x shift) -- most suspicious features:")
    print(combined[["feature", "rf_importance", "importance_rank", "cohens_d", "rank",
                    "suspicious_score"]].head(20).round(4).to_string(index=False))

    print("\nTop 10 by RF importance alone:")
    top_imp = combined.sort_values("rf_importance", ascending=False).head(10)
    print(top_imp[["feature", "rf_importance", "importance_rank", "cohens_d", "rank"]]
          .round(4).to_string(index=False))

    both_top20 = set(combined.nlargest(20, "rf_importance")["feature"]) & \
                 set(combined.nlargest(20, "abs_d")["feature"])
    print(f"\nFeatures in BOTH top-20-importance AND top-20-shift: {len(both_top20)}")
    print(f"  {sorted(both_top20)}")

    combined_out = f"{ROOT}\\reports\\pp2_finalized_reports\\acoustic_importance_x_shift_flood25.csv"
    combined.to_csv(combined_out, index=False)
    print(f"\nFull combined table written to {combined_out}")

    # ================= ANALYSIS 4: MFCC-specific =================
    print("\n" + "=" * 100)
    print("ANALYSIS 4: MFCC-specific shift (80 of 91 acoustic features)")
    print("=" * 100)
    mfcc_rows = shift[shift["feature"].str.startswith("mfcc_")].copy()
    mfcc_mean_rows = mfcc_rows[mfcc_rows["feature"].str.endswith("_mean")]
    mfcc_std_rows = mfcc_rows[mfcc_rows["feature"].str.endswith("_std")]
    print(f"\nMFCC feature count: {len(mfcc_rows)} (40 mean + 40 std)")
    print(f"  MFCC mean |d| (means only)  : {mfcc_mean_rows['abs_d'].mean():.3f}")
    print(f"  MFCC mean |d| (stds only)   : {mfcc_std_rows['abs_d'].mean():.3f}")
    print(f"  Non-MFCC acoustic mean |d|  : {shift[~shift['feature'].str.startswith('mfcc_')]['abs_d'].mean():.3f}")
    print(f"\nTop 10 most-shifted MFCC features specifically:")
    print(mfcc_rows.sort_values("abs_d", ascending=False).head(10)
          [["rank", "feature", "sim_mean", "dmc_mean", "cohens_d"]].round(4).to_string(index=False))
    n_mfcc_in_top20 = shift.head(20)["feature"].str.startswith("mfcc_").sum()
    print(f"\nOf the overall top 20 most-shifted features, {n_mfcc_in_top20} are MFCCs.")

    # ================= ANALYSIS 5: per-call error analysis =================
    print("\n" + "=" * 100)
    print("ANALYSIS 5: DMC flood-25 per-call acoustic-only predictions (5-seed majority)")
    print("=" * 100)
    all_preds = []
    for seed in SEEDS:
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=seed)),
            ("clf", RandomForestClassifier(random_state=seed, **RF_BASE)),
        ])
        pipe.fit(X_sim, y_sim)
        all_preds.append(pipe.predict(X_dmc))
    all_preds = np.array(all_preds)  # (5, n)

    error_rows = []
    for i in range(len(dmc_flood)):
        preds_i = le.inverse_transform(all_preds[:, i])
        vals, counts = np.unique(preds_i, return_counts=True)
        majority = vals[np.argmax(counts)]
        n_correct_seeds = int((preds_i == dmc_flood.iloc[i]["true_label"]).sum())
        error_rows.append({
            "recording_id": dmc_flood.iloc[i]["recording_id"],
            "true_label": dmc_flood.iloc[i]["true_label"],
            "majority_prediction": majority,
            "correct_seeds_of_5": n_correct_seeds,
            "consistently_correct": n_correct_seeds == 5,
            "consistently_wrong": n_correct_seeds == 0,
        })
    err_df = pd.DataFrame(error_rows)
    print(err_df.to_string(index=False))
    print(f"\nConsistently correct (5/5 seeds): {err_df['consistently_correct'].sum()}/{len(err_df)}")
    print(f"Consistently wrong (0/5 seeds)   : {err_df['consistently_wrong'].sum()}/{len(err_df)}")
    print("\nError pattern by true label:")
    print(err_df.groupby("true_label")["consistently_wrong"].agg(["sum", "count"]).to_string())
    print("\nWhat consistently-wrong calls get predicted as instead:")
    wrong = err_df[err_df["consistently_wrong"]]
    if len(wrong):
        print(wrong.groupby(["true_label", "majority_prediction"]).size().to_string())
    else:
        print("  (none)")

    err_out = f"{ROOT}\\reports\\pp2_finalized_reports\\acoustic_error_analysis_flood25.csv"
    err_df.to_csv(err_out, index=False)
    print(f"\nWritten to {err_out}")

    # ================= ANALYSIS 6: PCA visualization =================
    print("\n" + "=" * 100)
    print("ANALYSIS 6: PCA of 91 acoustic features -- VISUALIZATION ONLY, not a model")
    print("=" * 100)
    scaler = StandardScaler().fit(X_sim)
    X_sim_scaled = scaler.transform(X_sim)
    X_dmc_scaled = scaler.transform(X_dmc)  # same scaler fit on sim only -- diagnostic, not leakage

    pca = PCA(n_components=2, random_state=42).fit(X_sim_scaled)
    sim_pc = pca.transform(X_sim_scaled)
    dmc_pc = pca.transform(X_dmc_scaled)

    print(f"Explained variance ratio: PC1={pca.explained_variance_ratio_[0]:.3f}  "
          f"PC2={pca.explained_variance_ratio_[1]:.3f}  "
          f"(total {pca.explained_variance_ratio_[:2].sum():.3f})")

    sim_centroid = sim_pc.mean(axis=0)
    dmc_centroid = dmc_pc.mean(axis=0)
    centroid_dist = np.linalg.norm(sim_centroid - dmc_centroid)
    sim_spread = sim_pc.std(axis=0).mean()
    dmc_spread = dmc_pc.std(axis=0).mean()
    print(f"Sim centroid (PC1,PC2)  : ({sim_centroid[0]:.2f}, {sim_centroid[1]:.2f})")
    print(f"DMC centroid (PC1,PC2)  : ({dmc_centroid[0]:.2f}, {dmc_centroid[1]:.2f})")
    print(f"Centroid distance       : {centroid_dist:.2f}")
    print(f"Sim spread (mean std)   : {sim_spread:.2f}")
    print(f"DMC spread (mean std)   : {dmc_spread:.2f}")
    print(f"Centroid distance / sim spread : {centroid_dist / sim_spread:.2f}x "
          f"(how many 'sim clouds wide' apart the two groups sit)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(sim_pc[:, 0], sim_pc[:, 1], s=18, alpha=0.5, label=f"Simulated (n={len(sim)})", color="#4C72B0")
        ax.scatter(dmc_pc[:, 0], dmc_pc[:, 1], s=60, alpha=0.9, label=f"DMC flood (n={len(dmc_flood)})",
                  color="#C44E52", marker="^", edgecolors="black", linewidths=0.5)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax.set_title("PCA of 91 acoustic features: simulated vs DMC flood (n=25)\n"
                     "(visualization only -- not a predictive model)")
        ax.legend()
        fig.tight_layout()
        plot_path = f"{ROOT}\\reports\\pp2_finalized_reports\\acoustic_pca_sim_vs_dmc_flood25.png"
        fig.savefig(plot_path, dpi=150)
        print(f"\nPCA plot saved to {plot_path}")
    except Exception as e:
        print(f"\nCould not save PCA plot: {e}")


if __name__ == "__main__":
    main()
