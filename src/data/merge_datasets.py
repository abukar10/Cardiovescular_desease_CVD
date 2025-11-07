"""
merge_datasets.py
------------------
Merge the four UCI Heart Disease datasets (Cleveland, Hungarian, Switzerland, VA)
into a single unified dataset for cross-regional model training.
"""

import os
import pandas as pd
import numpy as np
import joblib

# ---------- Config ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ---------- Column name normalization ----------
COLUMN_MAP = {
    "age": "age",
    "sex": "sex",
    "cp": "cp",
    "trestbps": "trestbps",
    "chol": "chol",
    "fbs": "fbs",
    "restecg": "restecg",
    "thalach": "thalach",
    "exang": "exang",
    "oldpeak": "oldpeak",
    "slope": "slope",
    "ca": "ca",
    "thal": "thal",
    "num": "target",
    "target": "target"
}

# ---------- Helper ----------
def load_and_clean_dataset(path, source_name):
    """Load dataset, fix missing headers, standardize columns and target."""
    # Try reading with or without header
    try:
        df = pd.read_csv(path)
        # If most column names are numeric or very short, assume no header
        if all(str(c).replace('.', '', 1).isdigit() for c in df.columns):
            print(f"⚠️  {source_name}: numeric column names detected, reloading without header.")
            df = pd.read_csv(path, header=None)
    except Exception:
        df = pd.read_csv(path, header=None)

    # Replace ? with NaN
    df.replace("?", np.nan, inplace=True)

    # Assign standard UCI Heart Disease columns if count matches (~14)
    uci_columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    if len(df.columns) >= len(uci_columns):
        df.columns = uci_columns + [f"extra_{i}" for i in range(len(df.columns) - len(uci_columns))]
    else:
        print(f"⚠️  {source_name}: Unexpected number of columns ({len(df.columns)}), padding missing ones.")
        df.columns = uci_columns[:len(df.columns)]

    # Convert to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binarize target (UCI encodes 0–4)
    df["target"] = df["target"].apply(lambda x: 1 if pd.notna(x) and float(x) > 0 else 0)

    # Add source tag
    df["source"] = source_name

    return df


# ---------- Main ----------
def merge_all_datasets():
    merged_df = pd.DataFrame()
    datasets = ["cleveland", "hungarian", "switzerland", "va"]

    for name in datasets:
        path = os.path.join(RAW_DIR, f"{name}.csv")
        if os.path.exists(path):
            print(f"📂 Loading {name} dataset...")
            df = load_and_clean_dataset(path, name)
            merged_df = pd.concat([merged_df, df], axis=0, ignore_index=True)
        else:
            print(f"⚠️ Missing file: {path}")

    # Align all columns across datasets
    merged_df = merged_df.reindex(sorted(merged_df.columns), axis=1)

    # Drop rows missing critical features (age, sex, cp, target)
    merged_df.dropna(subset=["age", "sex", "cp", "target"], inplace=True)

    print(f"\n✅ Combined dataset shape: {merged_df.shape}")

    # Save CSV
    out_csv = os.path.join(PROCESSED_DIR, "combined_uci_heart.csv")
    merged_df.to_csv(out_csv, index=False)
    print(f"💾 Saved merged dataset to {out_csv}")

    # Split X/y and save joblib
    X = merged_df.drop(columns=["target"])
    y = merged_df["target"]

    joblib.dump(X, os.path.join(PROCESSED_DIR, "combined_uci_heart_X.joblib"))
    joblib.dump(y, os.path.join(PROCESSED_DIR, "combined_uci_heart_y.joblib"))
    print(f"💾 Saved X/y joblib files to {PROCESSED_DIR}")


if __name__ == "__main__":
    merge_all_datasets()
