# src/stats_and_validation.py
"""
Usage:
  python -m src.stats_and_validation --features data/processed/knee_angles_test.csv [--group_test]
"""

import argparse
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

def main(args):
    print(args)
    df = pd.read_csv(args.features)
    
def paired_test(df, subject_col='subject_id', cond_col='condition', value_col='ROM'):
    # Ensure columns exist
    for col in [subject_col, cond_col, value_col]:
        if col not in df.columns:
            print(f"Paired test skipped: column '{col}' not found")
            return np.array([]), np.array([]), *([np.nan]*6)

    subs = sorted(df[subject_col].dropna().unique())
    normals, restricts = [], []
    for s in subs:
        srows = df[df[subject_col] == s]
        nom = srows[srows[cond_col].str.lower()=='normal'][value_col].mean() if not srows.empty else np.nan
        res = srows[srows[cond_col].str.lower()=='restricted'][value_col].mean() if not srows.empty else np.nan
        if np.isfinite(nom) and np.isfinite(res):
            normals.append(nom)
            restricts.append(res)

    normals = np.array(normals)
    restricts = np.array(restricts)
    if len(normals) == 0:
        return normals, restricts, *([np.nan]*6)

    tstat, pval = stats.ttest_rel(normals, restricts)
    cohen_d = (normals - restricts).mean() / (normals - restricts).std(ddof=1)
    mean_diff = (normals - restricts).mean()
    return normals, restricts, tstat, pval, cohen_d, np.mean(normals), np.mean(restricts), mean_diff

def independent_test(df, group_col, value_col):
    # Ensure columns exist
    if group_col not in df.columns or value_col not in df.columns:
        print(f"Independent test skipped: missing columns '{group_col}' or '{value_col}'")
        return np.array([]), np.array([]), *([np.nan]*6)

    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        print(f"Independent test skipped: '{group_col}' must have exactly 2 unique values")
        return np.array([]), np.array([]), *([np.nan]*6)

    g1 = pd.to_numeric(df[df[group_col]==groups[0]][value_col], errors='coerce').dropna().values
    g2 = pd.to_numeric(df[df[group_col]==groups[1]][value_col], errors='coerce').dropna().values
    if len(g1)==0 or len(g2)==0:
        print(f"Independent test skipped: one or both groups have no numeric values in '{value_col}'")
        return g1, g2, *([np.nan]*6)

    tstat, pval = stats.ttest_ind(g1, g2, equal_var=False)
    cohen_d = (np.mean(g1) - np.mean(g2)) / np.sqrt((np.std(g1, ddof=1)**2 + np.std(g2, ddof=1)**2)/2)
    mean_diff = np.mean(g1) - np.mean(g2)
    return g1, g2, tstat, pval, cohen_d, np.mean(g1), np.mean(g2), mean_diff

def bland_altman(a, b, title="Bland-Altman Plot"):
    if len(a)==0 or len(b)==0:
        print(f"Bland-Altman skipped: insufficient data")
        return
    mean_vals = (a+b)/2
    diff = a-b
    md = np.mean(diff)
    sd = np.std(diff, ddof=1)
    loa_upper = md + 1.96*sd
    loa_lower = md - 1.96*sd

    plt.figure(figsize=(6,4))
    plt.scatter(mean_vals, diff, alpha=0.7)
    plt.axhline(md, color='red', linestyle='--', label=f"Mean diff = {md:.2f}")
    plt.axhline(loa_upper, color='gray', linestyle='--', label=f"Upper LoA = {loa_upper:.2f}")
    plt.axhline(loa_lower, color='gray', linestyle='--', label=f"Lower LoA = {loa_lower:.2f}")
    plt.xlabel("Mean of paired measurements")
    plt.ylabel("Difference")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main(args):
    df = pd.read_csv(args.features)
    summary_rows = []

    # Paired test: Normal vs Restricted
    normals, restricts, tstat, pval, cohen_d, mean1, mean2, mean_diff = paired_test(df)
    if len(normals)>0:
        bland_altman(normals, restricts, title="Normal vs Restricted ROM")
    summary_rows.append({
        "Comparison":"Normal vs Restricted",
        "Test_Type":"Paired",
        "t_stat":tstat,
        "p_value":pval,
        "Cohen_d":cohen_d,
        "Mean_1":mean1,
        "Mean_2":mean2,
        "Mean_diff":mean_diff
    })

    # Optional independent group tests
    if args.group_test:
        # Gender
        g1, g2, tstat, pval, cohen_d, mean1, mean2, mean_diff = independent_test(df,'Gender','ROM')
        summary_rows.append({
            "Comparison":"Male vs Female",
            "Test_Type":"Independent",
            "t_stat":tstat,
            "p_value":pval,
            "Cohen_d":cohen_d,
            "Mean_1":mean1,
            "Mean_2":mean2,
            "Mean_diff":mean_diff
        })
        # ActivityLevel
        a1, a2, tstat, pval, cohen_d, mean1, mean2, mean_diff = independent_test(df,'ActivityLevel','ROM')
        summary_rows.append({
            "Comparison":"Athletic vs Normal",
            "Test_Type":"Independent",
            "t_stat":tstat,
            "p_value":pval,
            "Cohen_d":cohen_d,
            "Mean_1":mean1,
            "Mean_2":mean2,
            "Mean_diff":mean_diff
        })

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    outdir = Path("results")
    outdir.mkdir(exist_ok=True)
    summary_df.to_csv(outdir/"stats_summary.csv", index=False)
    print("Saved summary to results/stats_summary.csv")
    print(summary_df)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Paired and independent t-tests with Cohen's d + optional group test")
    parser.add_argument("--features", required=True, help="CSV file with features (e.g., data/processed/knee_angles_test.csv)")
    parser.add_argument(
    "--group_test",
    action="store_true",   # Makes it a boolean flag
    help="Run independent group t-tests (Gender, ActivityLevel)"
    )
    args = parser.parse_args()
    main(args)
    