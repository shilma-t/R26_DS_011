"""
step2_fold_differences.py
Fold-level paired differences between ablation conditions.
Reads reports/ablation_context_folds.csv (already generated).
"""

import os
import pandas as pd
import numpy as np

_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FOLDS   = os.path.join(_ROOT, "reports", "pp2", "ablation_context_folds.csv")
REPORTS = os.path.join(_ROOT, "reports", "pp2")

df = pd.read_csv(FOLDS)

# One macro_f1 row per (condition, fold) — take first class to avoid triple-count
mf1 = (df.groupby(["condition", "fold"])["macro_f1"]
         .first()
         .unstack("condition"))

# Medium recall per (condition, fold)
med_recall = (df[df["class"] == "Medium"]
              .groupby(["condition", "fold"])["recall"]
              .first()
              .unstack("condition"))

print("=" * 60)
print("MACRO F1 PER FOLD")
print("=" * 60)
print(mf1.round(4).to_string())

print("\n" + "=" * 60)
print("PAIRED DIFFERENCES PER FOLD")
print("=" * 60)

conds = ["A_acoustic", "B_old_keyword", "C_base", "D_context", "E_context_gate"]

diffs = pd.DataFrame(index=range(1, 6))
diffs.index.name = "fold"
diffs["B-A"] = mf1["B_old_keyword"] - mf1["A_acoustic"]
diffs["C-B"] = mf1["C_base"]        - mf1["B_old_keyword"]
diffs["D-C"] = mf1["D_context"]     - mf1["C_base"]
diffs["E-D"] = mf1["E_context_gate"]- mf1["D_context"]
diffs["E_med_recall - D_med_recall"] = (med_recall["E_context_gate"]
                                        - med_recall["D_context"])

print(diffs.round(4).to_string())

print("\n" + "=" * 60)
print("SUMMARY (mean ± std of paired differences)")
print("=" * 60)
for col in diffs.columns:
    vals = diffs[col]
    pos  = (vals > 0).sum()
    print(f"  {col:<35} mean={vals.mean():+.4f}  std={vals.std():.4f}  "
          f"positive in {pos}/5 folds")

print("\n" + "=" * 60)
print("E MEDIUM RECALL — FOLD BY FOLD")
print("=" * 60)
for fold in range(1, 6):
    d_val = med_recall.loc[fold, "D_context"]
    e_val = med_recall.loc[fold, "E_context_gate"]
    delta = e_val - d_val
    print(f"  Fold {fold}:  D={d_val:.4f}  E={e_val:.4f}  Δ={delta:+.4f}")

# Save
diffs.to_csv(os.path.join(REPORTS, "ablation_fold_differences.csv"))
print(f"\nSaved → reports/ablation_fold_differences.csv")
