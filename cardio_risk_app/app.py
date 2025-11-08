from pathlib import Path
import warnings
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

APP_DIR = Path(__file__).resolve().parent
FEATURE_COLUMNS = [
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

warnings.filterwarnings(
    "ignore",
    message=".*serialized model.*",
    category=UserWarning,
)


def build_heart_nn(input_dim: int) -> nn.Module:
    class HeartNN(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    return HeartNN(input_dim)


class TabTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.input_proj(x)
        tokens = proj.unsqueeze(1)
        encoded = self.encoder(tokens)
        pooled = encoded.squeeze(1)
        return self.fc(pooled)


# -------------------------------
# Load Models and Scaler
# -------------------------------
@st.cache_resource
def load_models() -> Tuple[Dict[str, object], object]:
    model_dir = APP_DIR / "models"
    scaler_path = APP_DIR / "data" / "standard_scaler_combined_uci_heart.joblib"

    scaler = None
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
        except Exception as exc:
            st.error(f"⚠️ Error loading scaler: {exc}")
    else:
        st.error("⚠️ Scaler file not found. Please add standard_scaler_combined_uci_heart.joblib to data/.")

    feature_names = getattr(scaler, "feature_names_in_", None) if scaler is not None else None
    input_dim = len(feature_names) if feature_names is not None else len(FEATURE_COLUMNS)

    model_specs = [
        {"name": "Logistic Regression", "path": model_dir / "logistic_regression_combined_uci_heart_model.joblib", "type": "joblib"},
        {"name": "SVM (RBF)", "path": model_dir / "svm_combined_uci_heart_model.joblib", "type": "joblib"},
        {"name": "XGBoost", "path": model_dir / "xgboost_combined_uci_heart_model.joblib", "type": "joblib"},
        {"name": "Neural Net", "path": model_dir / "neural_network_combined_uci_heart_model.pt", "type": "heart_nn"},
        {"name": "Transformer", "path": model_dir / "transformer_combined_uci_heart_model.pt", "type": "transformer"},
    ]

    models: Dict[str, object] = {}
    for spec in model_specs:
        path = spec["path"]
        if not path.exists():
            st.warning(f"⚠️ Missing artifact for {spec['name']}: {path.name}")
            continue

        try:
            if spec["type"] == "joblib":
                model = joblib.load(path)
            elif spec["type"] == "heart_nn":
                model = build_heart_nn(input_dim)
                state_dict = torch.load(path, map_location="cpu")
                model.load_state_dict(state_dict)
                model.eval()
            else:
                model = TabTransformer(num_features=input_dim)
                state_dict = torch.load(path, map_location="cpu")
                model.load_state_dict(state_dict)
                model.eval()
            models[spec["name"]] = model
        except Exception as exc:
            st.warning(f"⚠️ Could not load {spec['name']}: {exc}")

    if models:
        st.success(f"✅ Loaded {len(models)} model(s).")
    else:
        st.error("❌ No models could be loaded.")

    return models, scaler


models, scaler = load_models()

# -------------------------------
# Page Layout
# -------------------------------
st.title("🩺 Cardiovascular Disease Risk Predictor")

left_col, middle_col, right_col = st.columns([1.6, 2.8, 2.2])

with left_col:
    st.markdown("### About this tool")
    st.markdown(
        """
        Ensemble predictions from five curated models trained on the combined UCI
        heart datasets highlight cardiovascular disease risk while keeping
        inference consistent with the original preprocessing pipeline.
        """
    )
    st.markdown("### Quick guide")
    st.markdown(
        """
        1. Enter patient data in the middle panel.  
        2. Click **Predict Risk** on the right.  
        3. Review model probabilities, consensus label, and visuals below.
        """
    )
    st.markdown("### Factors included")
    st.markdown(
        "- Age, Sex, Chest Pain Type, Thalassemia\n"
        "- Resting BP, Serum Cholesterol\n"
        "- Fasting Blood Sugar, Resting ECG, Max Heart Rate\n"
        "- ST Depression, ST Slope, Exercise-induced Angina\n"
        "- Major Vessels Colored"
    )

with middle_col:
    st.markdown("### Patient Inputs")
    col1, col2 = st.columns(2)

    sex = col2.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    fbs = col1.selectbox("Fasting Blood Sugar > 120", [1, 0], format_func=lambda x: "True" if x == 1 else "False")
    exang = col2.selectbox("Exercise-induced Angina", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    thal_display = {
        3: "Normal (3)",
        6: "Fixed Defect (6)",
        7: "Reversible Defect (7)",
    }

    raw_features = {
        "age": col1.number_input("Age (years)", min_value=20, max_value=100, value=50),
        "ca": col2.slider("Major Vessels Colored (0–3)", min_value=0, max_value=3, value=0),
        "chol": col1.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200),
        "cp": col1.slider("Chest Pain Type (1–4)", min_value=1, max_value=4, value=3),
        "exang": exang,
        "fbs": fbs,
        "oldpeak": col2.number_input("ST Depression (0.0–6.0)", min_value=0.0, max_value=6.0, value=1.0, step=0.1),
        "restecg": col1.slider("Resting ECG (0–2)", min_value=0, max_value=2, value=1),
        "sex": sex,
        "slope": col2.slider("ST Slope (0–2)", min_value=0, max_value=2, value=1),
        "thal": col2.selectbox("Thalassemia", [3, 6, 7], format_func=lambda x: thal_display[x]),
        "thalach": col2.number_input("Max Heart Rate", min_value=60, max_value=220, value=150),
        "trestbps": col1.number_input("Resting BP (mm Hg)", min_value=80, max_value=200, value=120),
    }

with right_col:
    st.markdown("### Prediction")
    predict_button = st.button("🔍 Predict Risk", use_container_width=True)
    prediction_placeholder = st.container()

# Convert to array
input_df = pd.DataFrame([{key: raw_features[key] for key in FEATURE_COLUMNS}])

if scaler is None:
    st.stop()

expected_columns = getattr(scaler, "feature_names_in_", None)

if expected_columns is not None:
    missing = [col for col in expected_columns if col not in input_df.columns]
    extra = [col for col in input_df.columns if col not in expected_columns]

    if missing:
        st.error(
            "⚠️ Missing required feature(s) for the scaler: "
            + ", ".join(missing)
        )
        st.stop()

    if extra:
        st.warning(
            "ℹ️ Ignoring unexpected input feature(s): " + ", ".join(extra)
        )

    input_df = input_df.loc[:, expected_columns]

scaled = scaler.transform(input_df)

# -------------------------------
# Prediction Logic
# -------------------------------
with prediction_placeholder:
    if predict_button:
        st.subheader("Model Predictions")

        results = []
        for name, model in models.items():
            try:
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(scaled)[0, 1]
                elif hasattr(model, "__call__"):  # neural net / transformer
                    with torch.no_grad():
                        tensor_input = torch.tensor(scaled, dtype=torch.float32)
                        if isinstance(model, nn.Module):
                            params = list(model.parameters())
                            device = params[0].device if params else torch.device("cpu")
                            tensor_input = tensor_input.to(device)
                            logits = model(tensor_input).detach().cpu().squeeze().item()
                            prob = float(torch.sigmoid(torch.tensor(logits)).item())
                        else:
                            output = model(tensor_input).detach().cpu().item()
                            prob = float(np.clip(output, 0.0, 1.0))
                else:
                    continue
                pred = int(prob >= 0.5)
                results.append((name, prob, pred))
            except Exception as e:
                st.warning(f"⚠️ Skipping {name}: {e}")

        if results:
            df = pd.DataFrame(results, columns=["Model", "Prob", "Pred"])
            df["Risk"] = df["Pred"].map({1: "High Risk", 0: "Low Risk"})
            st.table(df)

            consensus = "High Risk" if df["Pred"].mean() >= 0.5 else "Low Risk"
            st.markdown(f"### 🩸 Consensus: **{consensus}**")
        else:
            st.error("No valid predictions could be made.")
    else:
        st.info("Adjust the inputs, then click **Predict Risk** to generate ensemble predictions.")

# -------------------------------
# Visualizations Section
# -------------------------------
st.markdown("---")
st.subheader("📈 Model Performance Visualizations")

visuals_dir = APP_DIR / "visuals"


@st.cache_data
def load_performance_summary() -> Optional[pd.DataFrame]:
    path = APP_DIR / "data" / "crossval_summary_final.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    renamed = df.rename(
        columns={
            "CV_Accuracy_mean": "CV Accuracy",
            "CV_AUC_mean": "CV AUC",
            "Accuracy": "Holdout Accuracy",
            "Macro F1": "Holdout Macro F1",
        }
    )
    display_cols = ["Model", "Holdout Accuracy", "Holdout Macro F1", "CV Accuracy", "CV AUC"]
    return renamed[[col for col in display_cols if col in renamed.columns]]


def display_visual(image_name: str, caption: str, container=st, width: int | None = 420):
    image_path = visuals_dir / image_name
    if image_path.exists():
        if width is None:
            container.image(str(image_path), caption=caption, use_column_width=True)
        else:
            container.image(str(image_path), caption=caption, width=width)
    else:
        container.info(f"ℹ️ {caption} visual not available.")


performance_df = load_performance_summary()

tab1, tab2, tab3 = st.tabs(["ROC Curves", "Confusion Matrix", "Cross-Validation Summary"])

with tab1:
    col_a, col_b = st.columns(2)
    display_visual("roc_curves.png", "ROC Curves — Primary Models", container=col_a, width=360)
    display_visual("roc_curves_all.png", "Expanded ROC Comparison", container=col_b, width=360)

with tab2:
    display_visual("confusion_matrix.png", "Confusion Matrices", width=420)

with tab3:
    col_img, col_table = st.columns([1, 1])
    display_visual("model_performance.png", "Cross-Validation Results Snapshot", container=col_img, width=360)
    if performance_df is not None:
        col_table.dataframe(performance_df.set_index("Model"), use_container_width=True)
    else:
        col_table.info("ℹ️ Cross-validation summary not available.")

st.markdown("### 🧠 Training Insights")
insight_row1 = st.columns(2)
display_visual("nn_auc_curve.png", "Neural Network ROC Curve", container=insight_row1[0], width=360)
display_visual("transformer_auc_curve.png", "Transformer ROC Curve", container=insight_row1[1], width=360)

insight_row2 = st.columns(2)
display_visual("nn_loss_curve.png", "Neural Network Training Loss", container=insight_row2[0], width=360)
display_visual("transformer_loss_curve.png", "Transformer Training Loss", container=insight_row2[1], width=360)
