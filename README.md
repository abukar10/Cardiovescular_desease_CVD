# Cardiovascular Disease Risk Predictor

Interactive Streamlit application for estimating cardiovascular disease risk using multiple trained models and presenting supporting performance visuals.

## Features
- Collects standard clinical inputs and outputs probabilistic risk predictions from logistic regression, SVM, XGBoost, and neural network models.
- Applies the same preprocessing scaler used during training to keep inference consistent.
- Displays model comparison visuals (ROC curves and cross-validation summaries) to aid interpretation.
- Handles missing artifacts gracefully with in-app notifications.

## Repository Layout
- `cardio_risk_app/app.py`: Streamlit entry point.
- `cardio_risk_app/models/`: Serialized model artifacts (`*.joblib`, `*.pt`).
- `cardio_risk_app/data/standard_scaler_combined_uci_heart.joblib`: Fitted scaler for inference.
- `cardio_risk_app/visuals/`: Static PNG assets surfaced in the UI.
- `data/`, `experiments/`, `src/`, `notebooks/`: Offline training, evaluation, and experimentation assets.
- `requirements.txt`: Python dependencies for runtime and experimentation.

## Environment Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Run Locally
```bash
streamlit run cardio_risk_app/app.py
```

## Deployment Checklist
1. Ensure the following are committed: `cardio_risk_app/app.py`, `cardio_risk_app/models/`, `cardio_risk_app/data/standard_scaler_combined_uci_heart.joblib`, `cardio_risk_app/visuals/` assets, and `requirements.txt`.
2. For Streamlit Community Cloud: set main file path to `cardio_risk_app/app.py`. No Procfile is required.
3. For container targets (Render, Azure, AWS, etc.): optionally create a `Dockerfile` that installs `requirements.txt` and runs `streamlit run cardio_risk_app/app.py --server.address 0.0.0.0 --server.port $PORT`.
4. Confirm model files remain under size limits imposed by the hosting provider (Streamlit Cloud recommends < 100 MB total).

## Notes
- `cardio_risk_app/app.py` reports missing visuals or artifacts with Streamlit warnings to simplify debugging.
- Add `cardio_risk_app/visuals/confusion_matrix.png` (or adjust the caption) if you want the confusion-matrix tab populated.
- Training notebooks and scripts remain available under `notebooks/` and `src/` for further experimentation.

