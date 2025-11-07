from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "combined_uci_heart.csv"
APP_DIR = BASE_DIR / "cardio_risk_app"
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"
RESULTS_DIR = BASE_DIR / "experiments" / "results"

FEATURES = [
    "age",
    "ca",
    "chol",
    "cp",
    "exang",
    "fbs",
    "oldpeak",
    "restecg",
    "sex",
    "slope",
    "thal",
    "thalach",
    "trestbps",
]


def prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    target = df["target"].astype(int)

    X = df[FEATURES].copy()

    # Impute missing numeric values with column medians
    for col in FEATURES:
        if X[col].isnull().any():
            if col == "thal":
                # use most common category for thalassemia
                mode_value = X[col].mode(dropna=True)
                fill_value = int(mode_value.iloc[0]) if not mode_value.empty else 3
                X[col].fillna(fill_value, inplace=True)
            else:
                X[col].fillna(X[col].median(), inplace=True)

    # Ensure integer-like columns are ints (helps with reproducibility)
    int_like = ["ca", "cp", "exang", "fbs", "restecg", "sex", "slope", "thal"]
    for col in int_like:
        X[col] = X[col].astype(int)

    return X, target


def train_models(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000, solver="lbfgs")
    log_reg.fit(X_train_scaled, y_train)

    svm = SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=42)
    svm.fit(X_train_scaled, y_train)

    metrics = {
        "logistic_regression": {
            "roc_auc": roc_auc_score(y_test, log_reg.predict_proba(X_test_scaled)[:, 1]),
            "report": classification_report(
                y_test, log_reg.predict(X_test_scaled), output_dict=True
            ),
        },
        "svm_rbf": {
            "roc_auc": roc_auc_score(y_test, svm.predict_proba(X_test_scaled)[:, 1]),
            "report": classification_report(
                y_test, svm.predict(X_test_scaled), output_dict=True
            ),
        },
    }

    return scaler, log_reg, svm, metrics


def save_artifacts(scaler, log_reg, svm, metrics):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scaler_path = DATA_DIR / "standard_scaler_combined_uci_heart.joblib"
    log_reg_path = MODELS_DIR / "logistic_regression_combined_uci_heart_model.joblib"
    svm_path = MODELS_DIR / "svm_combined_uci_heart_model.joblib"
    metrics_path = RESULTS_DIR / "app_model_metrics.json"

    joblib.dump(scaler, scaler_path)
    joblib.dump(log_reg, log_reg_path)
    joblib.dump(svm, svm_path)

    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    return scaler_path, log_reg_path, svm_path, metrics_path


def main():
    X, y = prepare_data()
    scaler, log_reg, svm, metrics = train_models(X, y)
    scaler_path, log_reg_path, svm_path, metrics_path = save_artifacts(
        scaler, log_reg, svm, metrics
    )

    print("Rebuilt artifacts:")
    print(f"  - Scaler: {scaler_path}")
    print(f"  - Logistic Regression: {log_reg_path}")
    print(f"  - SVM (RBF): {svm_path}")
    print(f"  - Metrics summary: {metrics_path}")
    print("\nROC AUC:")
    for name, info in metrics.items():
        print(f"  {name}: {info['roc_auc']:.3f}")


if __name__ == "__main__":
    main()

