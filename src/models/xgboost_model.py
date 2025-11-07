"""
xgboost_model.py
----------------
Train and evaluate an XGBoost classifier on a chosen processed dataset.

Usage:
    python -m src.models.xgboost_model             # uses 'cleveland'
    python -m src.models.xgboost_model combined_uci_heart
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==== Read dataset name from CLI ====
dataset_name = sys.argv[1] if len(sys.argv) > 1 else "cleveland"
print(f"🚀 Training XGBoost on dataset: {dataset_name}")

# ==== Paths ====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed")
RESULTS_DIR = os.path.join(BASE_DIR, "experiments/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==== Load processed data ====
X_path = os.path.join(PROCESSED_DIR, f"{dataset_name}_X.joblib")
y_path = os.path.join(PROCESSED_DIR, f"{dataset_name}_y.joblib")

print("📂 Loading processed data...")
X = joblib.load(X_path)
y = joblib.load(y_path)
print(f"✅ Data loaded: X={X.shape}, y={y.shape}")

# ==== Split train/test ====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")

# ==== Define model ====
print("⚙️  Training XGBoost model...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False
)

model.fit(X_train, y_train)

# ==== Predictions ====
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ==== Metrics ====
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# ==== Display metrics ====
print("\n📈 Evaluation Metrics:")
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print("Confusion Matrix:")
print(cm)

# ==== Save outputs ====
model_path = os.path.join(RESULTS_DIR, f"xgboost_{dataset_name}_model.joblib")
metrics_path = os.path.join(RESULTS_DIR, f"xgboost_{dataset_name}_metrics.csv")

joblib.dump(model, model_path)
report_df.to_csv(metrics_path, index=True)

print(f"\n✅ Model saved to: {model_path}")
print(f"✅ Metrics saved to: {metrics_path}")
