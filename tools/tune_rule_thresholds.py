# tools/tune_rule_thresholds.py
"""
Grid-search tuner for rule-based thresholds.

Run from project root:
  python tools/tune_rule_thresholds.py --features data/processed/knee_angles_test.csv --out results/tune_results.csv

It will evaluate trial-level and subject-level metrics for combinations of thresholds
and save the full grid with metrics to the specified out CSV.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

def rule_predict_row_with_thresholds(row, rom_thresh, pv_thresh, sm_thresh):
    rom = row.get('ROM', np.nan)
    try:
        rom = float(rom)
    except Exception:
        rom = np.nan
    if not np.isnan(rom):
        return 'restricted' if rom <= rom_thresh else 'normal'
    pv = row.get('peak_vel', np.nan)
    try:
        pv = float(pv)
    except Exception:
        pv = np.nan
    if not np.isnan(pv):
        return 'restricted' if pv <= pv_thresh else 'normal'
    sm = row.get('smoothness', np.nan)
    try:
        sm = float(sm)
    except Exception:
        sm = np.nan
    if not np.isnan(sm):
        return 'restricted' if sm <= sm_thresh else 'normal'
    return 'normal'

def evaluate_trial_level(df, pred_col='pred'):
    # binary arrays: restricted=1, normal=0
    true = df['condition'].astype(str).str.lower().map(lambda s: 1 if 'restricted' in s else 0).values
    pred = df[pred_col].astype(str).str.lower().map(lambda s: 1 if 'restricted' in s else 0).values
    
    # force confusion matrix to be 2x2 even if a class is missing
    from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
    cm = confusion_matrix(true, pred, labels=[0,1])
    # cm is [[tn, fp],[fn, tp]]
    tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])

    # compute metrics safely (avoid divide-by-zero)
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total if total > 0 else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # recall for restricted
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = (2*precision*sensitivity/(precision+sensitivity)) if (precision and sensitivity and (precision+sensitivity)>0) else np.nan

    return {'accuracy': accuracy, 'sensitivity': sensitivity, 'precision': precision, 'f1': f1, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}

def evaluate_subject_level(df, pred_col='pred', subject_col='subject_id'):
    # aggregate per-subject: majority vote and any-restricted rules
    rows = []
    for subj, g in df.groupby(subject_col):
        true_label = 1 if any(g['condition'].astype(str).str.lower().str.contains('restricted')) else 0
        pred_vals = g[pred_col].astype(str).str.lower().map(lambda s: 1 if 'restricted' in s else 0).values
        maj = 1 if pred_vals.sum() >= (len(pred_vals)/2) else 0
        any_restricted = 1 if pred_vals.sum() >= 1 else 0
        rows.append({'subject': subj, 'true': true_label, 'pred_majority': maj, 'pred_any': any_restricted})
    subj_df = pd.DataFrame(rows)

    # evaluate both aggregation strategies, using labels=[0,1] to ensure 2x2 confusion matrix
    results = {}
    from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
    for col in ['pred_majority','pred_any']:
        true = subj_df['true'].values
        pred = subj_df[col].values
        cm = confusion_matrix(true, pred, labels=[0,1])
        tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
        total = tn + fp + fn + tp
        acc = (tn + tp) / total if total > 0 else np.nan
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        f1 = (2*prec*sens/(prec+sens)) if (prec and sens and (prec+sens)>0) else np.nan
        results[col] = {'accuracy': acc, 'sensitivity': sens, 'precision': prec, 'f1': f1, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}

    return results, subj_df

def grid_search(df, rom_range, pv_range, sm_range):
    records = []
    total = len(rom_range)*len(pv_range)*len(sm_range)
    i = 0
    for rom in rom_range:
        for pv in pv_range:
            for sm in sm_range:
                i += 1
                # predict
                df['pred'] = df.apply(lambda r: rule_predict_row_with_thresholds(r, rom, pv, sm), axis=1)
                trial_metrics = evaluate_trial_level(df, pred_col='pred')
                subj_results, subj_df = evaluate_subject_level(df, pred_col='pred', subject_col='subject_id')
                rec = {
                    'rom_thresh': rom,
                    'pv_thresh': pv,
                    'sm_thresh': sm,
                    # trial-level
                    'trial_acc': trial_metrics['accuracy'],
                    'trial_sens': trial_metrics['sensitivity'],
                    'trial_prec': trial_metrics['precision'],
                    'trial_f1': trial_metrics['f1'],
                    # subject-level (majority)
                    'subj_maj_acc': subj_results['pred_majority']['accuracy'],
                    'subj_maj_sens': subj_results['pred_majority']['sensitivity'],
                    'subj_maj_prec': subj_results['pred_majority']['precision'],
                    'subj_maj_f1': subj_results['pred_majority']['f1'],
                    # subject-level (any)
                    'subj_any_acc': subj_results['pred_any']['accuracy'],
                    'subj_any_sens': subj_results['pred_any']['sensitivity'],
                    'subj_any_prec': subj_results['pred_any']['precision'],
                    'subj_any_f1': subj_results['pred_any']['f1'],
                    # counts (trial-level)
                    'TP': trial_metrics['TP'], 'TN': trial_metrics['TN'], 'FP': trial_metrics['FP'], 'FN': trial_metrics['FN']
                }
                records.append(rec)
    return pd.DataFrame.from_records(records)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", default="results/tune_results.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    # ensure condition, subject_id exist
    if 'condition' not in df.columns or 'subject_id' not in df.columns:
        raise ValueError("CSV must contain 'condition' and 'subject_id' columns")

    # define ranges to search (tweak as needed)
    rom_range = np.arange(120, 141, 2)          # 120,122,...,140
    pv_range = np.round(np.arange(0.60, 0.91, 0.05), 3)  # 0.60..0.90 step 0.05
    sm_range = np.round(np.arange(0.80, 0.97, 0.02), 3)  # 0.80..0.96 step 0.02

    print(f"Grid search size: {len(rom_range)*len(pv_range)*len(sm_range)} combinations")
    results_df = grid_search(df, rom_range, pv_range, sm_range)

    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(outpath, index=False)
    print("Saved tuning results to", outpath)

    # print top 10 combos by subject-majority F1 (common clinically-relevant metric)
    top = results_df.sort_values('subj_maj_f1', ascending=False).head(10)
    print("\nTop 10 by subject-majority F1:\n", top[['rom_thresh','pv_thresh','sm_thresh','subj_maj_f1','subj_maj_sens','subj_maj_prec']])

if __name__ == "__main__":
    main()