"""
preprocess_tabular.py
---------------------
Cleans, encodes, and scales cardiovascular disease datasets.
Outputs processed files to: data/processed/
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from joblib import dump

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../../data/processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

def preprocess_dataframe(df: pd.DataFrame, target_col: str = "target") -> tuple:
    """Return (X_processed, y, pipeline)"""
    df = df.copy()

    # Handle possible target name variations
    if target_col not in df.columns:
        possible = [c for c in df.columns if c.lower() in ("target", "output", "cardio", "heart_disease")]
        if possible:
            target_col = possible[0]
        else:
            raise ValueError("No target column found!")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Identify categorical and numeric columns
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or X[c].nunique() < 10]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Pipelines
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor


def preprocess_and_save(df: pd.DataFrame, name: str, target_col: str = "target"):
    """Preprocess dataframe and save to processed/ folder."""
    X, y, pipeline = preprocess_dataframe(df, target_col)
    X_path = os.path.join(PROCESSED_DIR, f"{name}_X.joblib")
    y_path = os.path.join(PROCESSED_DIR, f"{name}_y.joblib")
    pipe_path = os.path.join(PROCESSED_DIR, f"{name}_pipeline.joblib")

    dump(X, X_path)
    dump(y, y_path)
    dump(pipeline, pipe_path)

    print(f"✅ Saved processed data for {name}: {X.shape} -> {X_path}")
    return X, y, pipeline


# ===============================================================
# Entry Point — Preprocessing for any dataset
# ===============================================================
if __name__ == "__main__":
    import sys
    from src.data.download_datasets import load_dataset
    import joblib
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    # -----------------------------------------------------------
    # 1. Read dataset name from command-line argument
    # -----------------------------------------------------------
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "cleveland"
    print(f"🚀 Preprocessing dataset: {dataset_name}")

    # -----------------------------------------------------------
    # 2. Load dataset
    # -----------------------------------------------------------
    df = load_dataset(dataset_name)
    print(f"✅ Loaded dataset '{dataset_name}' with shape {df.shape}")

    # -----------------------------------------------------------
    # 3. Basic checks and split features/target
    # -----------------------------------------------------------
    if "target" not in df.columns:
        raise ValueError("❌ No 'target' column found in dataset!")

    X = df.drop("target", axis=1)
    y = df["target"]

    # Identify numerical and categorical features
    num_features = X.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X.select_dtypes(exclude=["int64", "float64"]).columns

    # -----------------------------------------------------------
    # 4. Build preprocessing pipeline
    # -----------------------------------------------------------
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )

    print("⚙️  Transforming data...")
    X_processed = preprocessor.fit_transform(X)

    # -----------------------------------------------------------
    # 5. Save processed outputs
    # -----------------------------------------------------------
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    X_path = os.path.join(PROCESSED_DIR, f"{dataset_name}_X.joblib")
    y_path = os.path.join(PROCESSED_DIR, f"{dataset_name}_y.joblib")
    joblib.dump(X_processed, X_path)
    joblib.dump(y, y_path)

    print(f"✅ Saved processed data for {dataset_name}: X={X_processed.shape}, y={y.shape}")
    print(f"📁 Files written to: {PROCESSED_DIR}")

