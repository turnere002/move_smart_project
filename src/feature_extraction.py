# src/feature_extraction.py
"""
Usage:
    python feature_extraction.py --input_dir ../data/processed --output ../data/processed/features_table.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import math

def smooth_angles(arr, window=11, poly=2):
    # arr: 1D numpy array, may contain NaNs
    s = pd.Series(arr).interpolate(limit_direction='both')
    # choose window odd and <= len
    w = min(window, len(s))
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return s.values
    filt = savgol_filter(s.values, w, poly)
    return filt

def extract_features_from_file(csv_path, fps=30):
    df = pd.read_csv(csv_path)
    arr = df['angle_deg'].values.astype(float)
    clean = smooth_angles(arr, window=11, poly=2)
    max_angle = np.nanmax(clean)
    min_angle = np.nanmin(clean)
    rom = float(max_angle - min_angle)
    mean_angle = float(np.nanmean(clean))
    std_angle = float(np.nanstd(clean))
    # velocity deg/s
    vel = np.abs(np.diff(clean)) * fps
    peak_vel = float(np.nanmax(vel)) if len(vel)>0 else 0.0
    mean_vel = float(np.nanmean(vel)) if len(vel)>0 else 0.0
    smoothness = 1.0 / (1.0 + float(np.nanstd(vel)))
    peak_idx = int(np.nanargmax(clean))
    time_to_peak = peak_idx / fps
    return {
        'file': str(csv_path),
        'max_angle': max_angle,
        'min_angle': min_angle,
        'ROM': rom,
        'mean_angle': mean_angle,
        'std_angle': std_angle,
        'peak_vel': peak_vel,
        'mean_vel': mean_vel,
        'smoothness': smoothness,
        'time_to_peak': time_to_peak
    }

def main(input_dir, output_csv):
    input_dir = Path(input_dir)
    files = sorted(list(input_dir.glob("*_angles.csv")))
    rows = []
    for f in files:
        feat = extract_features_from_file(f)
        # parse subject and condition from filename: P01_normal_angles.csv
        stem = Path(f).stem
        parts = stem.split('_')
        subj = parts[0] if len(parts)>0 else 'unknown'
        cond = parts[1] if len(parts)>1 else 'unknown'
        feat['subject_id'] = subj
        feat['condition'] = cond
        rows.append(feat)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print("Saved feature table to", output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input_dir, args.output)