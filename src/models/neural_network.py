"""
neural_network.py
-----------------
Train and evaluate a feed-forward neural network (PyTorch)
on a chosen processed dataset (default: 'cleveland').

Usage:
    python -m src.models.neural_network
    python -m src.models.neural_network combined_uci_heart
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==== Read dataset name from CLI ====
dataset_name = sys.argv[1] if len(sys.argv) > 1 else "cleveland"
print(f"🚀 Training Neural Network on dataset: {dataset_name}")

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

# ==== Train/test split ====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")

# ==== Convert to tensors ====
def to_dense(arr):
    return arr.toarray() if hasattr(arr, "toarray") else arr

X_train_dense = to_dense(X_train)
X_test_dense = to_dense(X_test)

X_train_tensor = torch.tensor(X_train_dense, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_dense, dtype=torch.float32)

y_train_tensor = torch.tensor(
    y_train.values if hasattr(y_train, "values") else y_train,
    dtype=torch.float32,
).view(-1, 1)
y_test_tensor = torch.tensor(
    y_test.values if hasattr(y_test, "values") else y_test,
    dtype=torch.float32,
).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# ==== Define the neural network ====
class HeartNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


input_dim = X_train_tensor.shape[1]
model = HeartNN(input_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"🖥️  Using device: {device}")

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==== Training loop ====
epochs = 30
print("⚙️  Training Neural Network...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss/len(train_loader):.4f}")

# ==== Evaluation ====
model.eval()
y_pred_proba = []
y_pred_label = []

with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        probs = model(X_batch).cpu().numpy()
        y_pred_proba.extend(probs)
        y_pred_label.extend((probs > 0.5).astype(int))

y_pred_proba = np.array(y_pred_proba).flatten()
y_pred_label = np.array(y_pred_label).flatten()
y_true = y_test_tensor.numpy().flatten()

acc = accuracy_score(y_true, y_pred_label)
auc = roc_auc_score(y_true, y_pred_proba)
cm = confusion_matrix(y_true, y_pred_label)
report = classification_report(y_true, y_pred_label, output_dict=True)
report_df = pd.DataFrame(report).transpose()

print("\n📈 Evaluation Metrics (Neural Net):")
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print("Confusion Matrix:")
print(cm)

# ==== Save model + metrics ====
model_path = os.path.join(RESULTS_DIR, f"neural_network_{dataset_name}_model.pt")
metrics_path = os.path.join(RESULTS_DIR, f"neural_network_{dataset_name}_metrics.csv")

torch.save(model.state_dict(), model_path)
report_df.to_csv(metrics_path, index=True)

print(f"\n✅ Model saved to: {model_path}")
print(f"✅ Metrics saved to: {metrics_path}")
