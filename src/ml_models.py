# src/ml_models.py
"""
Usage:
  python ml_models.py --features ../data/processed/features_table.csv --outdir ../results
"""
import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, GroupKFold, cross_validate
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
    import joblib
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install required packages with: pip install -r requirements.txt")
    sys.exit(1)

FEATURE_COLUMNS = ['ROM','Peak_Knee_Angle','smoothness']


def prepare_data(features_csv):
    df = pd.read_csv(features_csv)
    df = df.dropna(subset=FEATURE_COLUMNS + ['condition','subject_id'])
    df['label'] = df['condition'].map({'normal':0,'restricted':1})
    X = df[FEATURE_COLUMNS]
    y = df['label']
    groups = df['subject_id']
    return df, X, y, groups

def train_models(X_train, y_train):
    models = {
        'logreg': make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear')),
        'rf': RandomForestClassifier(n_estimators=200, random_state=42)
    }
    for name, m in models.items():
        m.fit(X_train, y_train)
    return models

def evaluate_model(m, X_test, y_test):
    y_pred = m.predict(X_test)
    try:
        y_proba = m.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test))>1 else np.nan
    except:
        auc = np.nan
    acc = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return {'accuracy':acc, 'roc_auc':auc, 'report':cr, 'confusion':cm}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    print("Features path:", args.features)
    print("Exists:", Path(args.features).exists())
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df, X, y, groups = prepare_data(args.features)
    # subject-wise split to avoid leakage
    subj_unique = df['subject_id'].unique()
    train_subj, test_subj = train_test_split(subj_unique, test_size=0.3, random_state=42)
    train_mask = df['subject_id'].isin(train_subj)
    X_train = df.loc[train_mask, FEATURE_COLUMNS]
    y_train = df.loc[train_mask, 'label']
    X_test = df.loc[~train_mask, FEATURE_COLUMNS]
    y_test = df.loc[~train_mask, 'label']

    models = train_models(X_train, y_train)
    results = {}
    for name,m in models.items():
        res = evaluate_model(m, X_test, y_test)
        results[name] = res
        joblib.dump(m, outdir/f"{name}.joblib")
        print(f"Model {name}: accuracy={res['accuracy']:.3f}, roc_auc={res['roc_auc']}")
        print(res['report'])
        print("Confusion matrix:\n", res['confusion'])
    # save a summary CSV
    summary = pd.DataFrame([
        {'model':name, 'accuracy':res['accuracy'], 'roc_auc':res['roc_auc']} for name,res in results.items()
    ])
    summary.to_csv(outdir/"model_summary.csv", index=False)
    print("Saved model summary to", outdir/"model_summary.csv")