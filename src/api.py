from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import xgboost as xgb
import traceback


from .utils import FEATURES

# -------- Paths --------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

# Load all models
log_reg = joblib.load(MODEL_DIR / "logistic_regression_model.pkl")
rf_model = joblib.load(MODEL_DIR / "random_forest_model.pkl")
xgb_model = xgb.Booster()
xgb_model.load_model(str(MODEL_DIR / "xgboost_model.json"))
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

# -------- Input Schema --------
class DiabetesInput(BaseModel):
    Pregnancies: int = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    BloodPressure: float = Field(..., ge=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: int = Field(..., ge=0)
    model_choice: str = Field("logistic", description="Choose 'logistic', 'randomforest', or 'xgboost'")

# -------- FastAPI App --------
app = FastAPI(
    title="Diabetes Prediction API (Multi-Model)",
    description="Supports Logistic Regression, Random Forest, and XGBoost",
    version="2.0.0",
)

# -------- Routes --------
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Use POST /predict with patient data and model_choice"
    }

@app.post("/predict")
def predict(payload: DiabetesInput):
    # Prepare input
    row = np.array([[getattr(payload, f) for f in FEATURES]])
    row_scaled = scaler.transform(row)

       # Select model
    choice = payload.model_choice.lower()

        # Select model
    choice = payload.model_choice.lower()

    if choice == "logistic":
        pred = int(log_reg.predict(row_scaled)[0])
        prob = float(log_reg.predict_proba(row_scaled)[0][1])

    elif choice == "randomforest":
        pred = int(rf_model.predict(row_scaled)[0])
        prob = float(rf_model.predict_proba(row_scaled)[0][1])

    elif choice == "xgboost":
        dmatrix = xgb.DMatrix(row_scaled, feature_names=FEATURES)
        prob = float(xgb_model.predict(dmatrix)[0])
        pred = int(prob >= 0.5)

    else:
        return {"error": "Invalid model_choice. Use 'logistic', 'randomforest', or 'xgboost'."}

    label = "Diabetic" if pred == 1 else "Non-Diabetic"
    return {
        "model_used": choice,
        "prediction": pred,
        "label": label,
        "probability": round(prob, 4)
    }
