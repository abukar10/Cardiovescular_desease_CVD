"""
download_datasets.py
--------------------
Handles download and loading of cardiovascular datasets (UCI and others).
"""

import os
import pandas as pd
from urllib.request import urlretrieve

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
os.makedirs(RAW_DIR, exist_ok=True)

# -------------------------------------------------------
# URLs for the four UCI heart datasets
# -------------------------------------------------------
UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"

UCI_SOURCES = {
    "cleveland": f"{UCI_BASE}/processed.cleveland.data",
    "hungarian": f"{UCI_BASE}/processed.hungarian.data",
    "switzerland": f"{UCI_BASE}/processed.switzerland.data",
    "va": f"{UCI_BASE}/processed.va.data",
}

UCI_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]


def download_uci_file(name, url):
    """Download one of the UCI heart datasets."""
    dest_path = os.path.join(RAW_DIR, f"{name}.csv")
    if not os.path.exists(dest_path):
        print(f"⬇️  Downloading {name} dataset...")
        try:
            urlretrieve(url, dest_path)
        except Exception as e:
            print(f"❌ Failed to download {name}: {e}")
    return dest_path


def load_uci_dataset(name):
    """Load a single UCI dataset into a DataFrame."""
    path = download_uci_file(name, UCI_SOURCES[name])
    df = pd.read_csv(path, header=None, names=UCI_COLUMNS)
    # Replace "?" with NaN and drop missing
    df = df.replace("?", pd.NA).dropna()
    # Convert to numeric
    df = df.apply(pd.to_numeric)
    # Recode target > 0 → 1 (disease)
    df["target"] = (df["target"] > 0).astype(int)
    return df


def load_dataset(dataset_name="cleveland"):
    """Main entry for pipeline."""
    if dataset_name == "combined_uci_heart":
        print("📦 Loading combined UCI Heart Disease dataset...")
        dfs = []
        for name in UCI_SOURCES.keys():
            df_part = load_uci_dataset(name)
            dfs.append(df_part)
            print(f"✅ {name} loaded: {df_part.shape}")
        df = pd.concat(dfs, axis=0).reset_index(drop=True)
        print(f"✅ Combined dataset shape: {df.shape}")
        return df

    elif dataset_name in UCI_SOURCES:
        print(f"📦 Loading UCI dataset: {dataset_name}")
        return load_uci_dataset(dataset_name)

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
