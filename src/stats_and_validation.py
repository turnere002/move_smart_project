# src/stats_and_validation.py
"""
Usage (from project root):
  python src\stats_and_validation.py --features data/processed/knee_angles_test.csv [--group_test]

Performs:
 - Paired t-test (normal vs restricted)
 - Optional independent tests (Gender, ActivityLevel)
 - Bland-Altman for paired measurements
 - Rule-based classification using thresholds loaded from results/best_thresholds.json
 - Saves stats_summary.csv and rule_metrics.csv in results/
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ------------------------------
# Load thresholds (if present)
# ------------------------------
_cfg_path = Path("results/best_thresholds.json")
if _cfg_path.exists():
    try:
        _cfg = json.loads(_cfg_path.read_text())
        ROM_THRESHOLD = _cfg.get("rom_thresh", 130)
        PEAK_VEL_THRESHOLD = _cfg.get("pv_thresh", 0.75)
        SMOOTHNESS_THRESHOLD = _cfg.get("sm_thresh", 0.89)
        RULE_LOGIC = _cfg.get("logic", "OR")
    except Exception as e:
        print("Warning: failed to load results/best_thresholds.json:", e)
        ROM_THRESHOLD = 130
        PEAK_VEL_THRESHOLD = 0.75
        SMOOTHNESS_THRESHOLD = 0.89
        RULE_LOGIC = "OR"
else:
    ROM_THRESHOLD = 130
    PEAK_VEL_THRESHOLD = 0.75
    SMOOTHNESS_THRESHOLD = 0.89
    RULE_LOGIC = "OR"

# ------------------------------
# Helper: safe CSV read
# ------------------------------
def read_csv_safe(path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"ERROR reading CSV {path}: {e}")
        sys.exit(1)
    if df.shape[0] == 0:
        print(f"ERROR: CSV contains no rows: {path}")
        sys.exit(1)
    return df

# ------------------------------
# Rule-based classifier
# ------------------------------
def rule_predict_row(row, rom_thresh=ROM_THRESHOLD, pv_thresh=PEAK_VEL_THRESHOLD,
                     sm_thresh=SMOOTHNESS_THRESHOLD, logic=RULE_LOGIC):
    def to_float(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    rom = to_float(row.get('ROM', np.nan))
    pv = to_float(row.get('peak_vel', np.nan))
    sm = to_float(row.get('smoothness', np.nan))

    conds = []
    if not np.isnan(rom):
        conds.append(rom <= rom_thresh)
    if not np.isnan(pv):
        conds.append(pv <= pv_thresh)
    if not np.isnan(sm):
        conds.append(sm <= sm_thresh)

    if len(conds) == 0:
        return 'normal'

    if str(logic).upper() == "AND":
        return 'restricted' if all(conds) else 'normal'
    else:
        return 'restricted' if any(conds) else 'normal'

# ------------------------------
# Compute rule metrics
# ------------------------------
def compute_rule_metrics(df, pred_col='rule_pred', true_col='condition'):
    true_bin = df[true_col].astype(str).str.lower().map(lambda s: 1 if 'restricted' in s else 0).values
    pred_bin = df[pred_col].astype(str).str.lower().map(lambda s: 1 if 'restricted' in s else 0).values

    tp = int(((true_bin == 1) & (pred_bin == 1)).sum())
    tn = int(((true_bin == 0) & (pred_bin == 0)).sum())
    fp = int(((true_bin == 0) & (pred_bin == 1)).sum())
    fn = int(((true_bin == 1) & (pred_bin == 0)).sum())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'total': total
    }

# ------------------------------
# Paired and independent tests
# ------------------------------
def paired_test(df, subject_col='subject_id', cond_col='condition', value_col='ROM'):
    subs = sorted(df[subject_col].dropna().unique())
    normals, restricts = [], []

    for s in subs:
        srows = df[df[subject_col] == s]
        nom = srows[srows[cond_col].str.lower()=='normal'][value_col].mean()
        res = srows[srows[cond_col].str.lower()=='restricted'][value_col].mean()
        if np.isfinite(nom) and np.isfinite(res):
            normals.append(nom)
            restricts.append(res)

    normals = np.array(normals)
    restricts = np.array(restricts)
    if len(normals) == 0:
        return normals, restricts, *([np.nan]*6)

    diff = normals - restricts
    mean_diff = diff.mean() if len(diff) > 0 else np.nan
    if np.all(diff == 0):
        tstat, pval, cohen_d = np.nan, np.nan, np.inf
    else:
        tstat, pval = stats.ttest_rel(normals, restricts)
        std_diff = np.std(diff, ddof=1)
        cohen_d = np.inf if std_diff == 0 else diff.mean() / std_diff

    return normals, restricts, tstat, pval, cohen_d, np.mean(normals), np.mean(restricts), mean_diff

def independent_test(df, group_col, value_col):
    if group_col not in df.columns or value_col not in df.columns:
        return np.array([]), np.array([]), *([np.nan]*6)
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        return np.array([]), np.array([]), *([np.nan]*6)

    g1 = pd.to_numeric(df[df[group_col]==groups[0]][value_col], errors='coerce').dropna().values
    g2 = pd.to_numeric(df[df[group_col]==groups[1]][value_col], errors='coerce').dropna().values
    if len(g1)==0 or len(g2)==0:
        return g1, g2, *([np.nan]*6)

    tstat, pval = stats.ttest_ind(g1, g2, equal_var=False)
    cohen_d = (np.mean(g1) - np.mean(g2)) / np.sqrt((np.std(g1, ddof=1)**2 + np.std(g2, ddof=1)**2)/2)
    mean_diff = np.mean(g1) - np.mean(g2)
    return g1, g2, tstat, pval, cohen_d, np.mean(g1), np.mean(g2), mean_diff

# ------------------------------
# Bland-Altman plot
# ------------------------------
def bland_altman(a, b, title="Bland-Altman Plot"):
    if len(a)==0 or len(b)==0:
        return
    mean_vals = (a+b)/2
    diff = a-b
    md = np.mean(diff)
    sd = np.std(diff, ddof=1)
    loa_upper = md + 1.96*sd if sd > 0 else md + 0.001
    loa_lower = md - 1.96*sd if sd > 0 else md - 0.001

    plt.figure(figsize=(6,4))
    plt.scatter(mean_vals, diff, alpha=0.7)
    plt.axhline(md, color='red', linestyle='--', label=f"Mean diff = {md:.2f}")
    plt.axhline(loa_upper, color='gray', linestyle='--', label=f"Upper LoA = {loa_upper:.2f}")
    plt.axhline(loa_lower, color='gray', linestyle='--', label=f"Lower LoA = {loa_lower:.2f}")
    plt.xlabel("Mean of paired measurements")
    plt.ylabel("Difference (normal - restricted)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------
# Main
# ------------------------------
def main(args):
    df = read_csv_safe(args.features)
    outdir = Path("results")
    outdir.mkdir(exist_ok=True)

    # -------------------------
    # Paired test
    # -------------------------
    summary_rows = []
    normals, restricts, tstat, pval, cohen_d, mean1, mean2, mean_diff = paired_test(df)
    if len(normals) > 0:
        bland_altman(normals, restricts, title="Normal vs Restricted ROM")
    summary_rows.append({
        "Comparison": "Normal vs Restricted",
        "Test_Type": "Paired",
        "t_stat": tstat,
        "p_value": pval,
        "Cohen_d": cohen_d,
        "Mean_1": mean1,
        "Mean_2": mean2,
        "Mean_diff": mean_diff
    })

    # -------------------------
    # Independent tests
    # -------------------------
    if args.group_test:
        for group_col, comp_name in [('Gender', 'Male vs Female'), ('ActivityLevel', 'Athletic vs Normal')]:
            g1, g2, tstat, pval, cohen_d, mean1, mean2, mean_diff = independent_test(df, group_col, 'ROM')
            summary_rows.append({
                "Comparison": comp_name,
                "Test_Type": "Independent",
                "t_stat": tstat,
                "p_value": pval,
                "Cohen_d": cohen_d,
                "Mean_1": mean1,
                "Mean_2": mean2,
                "Mean_diff": mean_diff
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "stats_summary.csv", index=False)
    pd.set_option('display.float_format', '{:.6g}'.format)
    print("Saved summary to results/stats_summary.csv")
    print(summary_df)

    # -------------------------
    # Rule-based metrics
    # -------------------------
    df['rule_pred'] = df.apply(rule_predict_row, axis=1)
    metrics = compute_rule_metrics(df, pred_col='rule_pred', true_col='condition')
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(outdir / "rule_metrics.csv", index=False)

    print("\nRule-based classifier metrics (saved to results/rule_metrics.csv):")
    print(metrics_df.to_string(index=False))

    # Print confusion counts
    print(f"\nConfusion counts: TP={metrics['TP']}, TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}, Sensitivity (recall restricted): {metrics['sensitivity']:.3f}, Specificity: {metrics['specificity']:.3f}")

# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paired and independent t-tests with Cohen's d + optional group test")
    parser.add_argument("--features", required=True, help="CSV file with features (e.g., data/processed/knee_angles_test.csv)")
    parser.add_argument("--group_test", action="store_true", help="Run independent group t-tests (Gender, ActivityLevel)")
    args = parser.parse_args()
    main(args)
