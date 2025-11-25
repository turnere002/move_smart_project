# src/visualize.py
"""
Example usage:
  python visualize.py --features ../data/processed/features_table.csv --output_dir ../figures
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_rom_boxplot(features_df, outpath):
    plt.figure(figsize=(6,4))
    sns.boxplot(data=features_df, x='condition', y='ROM')
    sns.stripplot(data=features_df, x='condition', y='ROM', color='black', alpha=0.5)
    plt.title("Range of Motion by Condition")
    plt.ylabel("ROM (degrees)")
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_example_trace(angle_csv, outpath):
    df = pd.read_csv(angle_csv)
    plt.figure(figsize=(8,3))
    plt.plot(df['timestamp_s'], df['angle_deg'])
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title(Path(angle_csv).stem)
    plt.savefig(outpath, dpi=150)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    features = pd.read_csv(args.features)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_rom_boxplot(features, outdir/"rom_boxplot.png")
    # example trace: choose first file
    if len(features)>0:
        plot_example_trace(features.loc[0,'file'], outdir/"example_trace.png")
    print("Figures saved to", outdir)