from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

APP_DIR = Path(__file__).resolve().parent

warnings.filterwarnings(
    "ignore",
    message=".*serialized model.*",
    category=UserWarning,
)
# -------------------------------
# Load Models and Scaler
# -------------------------------
@st.cache_resource
def load_models():
    model_dir = APP_DIR / "models"
    scaler_path = APP_DIR / "data" / "standard_scaler_combined_uci_heart.joblib"

    model_files = {
        "Logistic Regression": model_dir / "logistic_regression_combined_uci_heart_model.joblib",
        "SVM (RBF)": model_dir / "svm_combined_uci_heart_model.joblib",
        "XGBoost": model_dir / "xgboost_combined_uci_heart_model.joblib",
        "Neural Net": model_dir / "neural_network_combined_uci_heart_model.pt",
    }

    models = {}
    for name, path in model_files.items():
        if not path.exists():
            st.warning(f"⚠️ Missing artifact for {name}: {path.name}")
            continue

        try:
            if path.suffix == ".joblib":
                models[name] = joblib.load(path)
            else:
                models[name] = torch.load(path, map_location="cpu")
        except Exception as exc:
            st.warning(f"⚠️ Could not load {name}: {exc}")

    if models:
        st.success(f"✅ Loaded {len(models)} model(s).")
    else:
        st.error("❌ No models could be loaded.")

    scaler = None
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
        except Exception as exc:
            st.error(f"⚠️ Error loading scaler: {exc}")
    else:
        st.error("⚠️ Scaler file not found. Please add standard_scaler_combined_uci_heart.joblib to data/.")

    return models, scaler

models, scaler = load_models()

# -------------------------------
# Page Layout
# -------------------------------
st.title("🩺 Cardiovascular Disease Risk Predictor")
st.markdown("Enter patient parameters below to estimate risk of heart disease.")

# Feature Inputs
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

feature_order = [
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

# Convert to array
input_df = pd.DataFrame([{key: raw_features[key] for key in feature_order}])

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
if st.button("🔍 Predict Risk"):
    st.subheader("Model Predictions")

    results = []
    for name, model in models.items():
        try:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(scaled)[0, 1]
            elif hasattr(model, "__call__"):  # neural net
                with torch.no_grad():
                    prob = torch.sigmoid(model(torch.tensor(scaled, dtype=torch.float32))).item()
            else:
                continue
            pred = int(prob >= 0.5)
            results.append((name, prob, pred))
        except Exception as e:
            st.warning(f"⚠️ Skipping {name}: {e}")

    if results:
        df = pd.DataFrame(results, columns=["Model", "Prob", "Pred"])
        df["Risk"] = df["Pred"].map({1: "High Risk", 0: "Low Risk"})
        st.dataframe(df, use_container_width=True)

        consensus = "High Risk" if df["Pred"].mean() >= 0.5 else "Low Risk"
        st.markdown(f"### 🩸 Consensus: **{consensus}**")
    else:
        st.error("No valid predictions could be made.")

# -------------------------------
# Visualizations Section
# -------------------------------
st.markdown("---")
st.subheader("📈 Model Performance Visualizations")

visuals_dir = APP_DIR / "visuals"


def display_visual(image_name: str, caption: str):
    image_path = visuals_dir / image_name
    if image_path.exists():
        st.image(str(image_path), caption=caption)
    else:
        st.info(f"ℹ️ {caption} visual not available.")

tab1, tab2, tab3 = st.tabs(["ROC Curves", "Confusion Matrix", "Cross-Validation Summary"])

with tab1:
    display_visual("roc_curves.png", "ROC Curves — Model Performance")

with tab2:
    display_visual("confusion_matrix.png", "Confusion Matrices")

with tab3:
    display_visual("model_performance.png", "Cross-Validation Results")
