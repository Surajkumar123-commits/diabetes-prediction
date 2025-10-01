from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

from .utils import FEATURES

# -------- Paths --------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "diabetes_model.pkl"

# -------- Pydantic schema --------
class DiabetesInput(BaseModel):
    Pregnancies: int = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    BloodPressure: float = Field(..., ge=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: int = Field(..., ge=0)

# -------- FastAPI App --------
app = FastAPI(
    title="Diabetes Prediction API",
    description="Logistic Regression model served via FastAPI",
    version="1.0.0",
)

# -------- Load Model --------
model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"status": "ok", "message": "Use POST /predict with patient data"}

@app.post("/predict")
def predict(payload: DiabetesInput):
    # Keep input order consistent with training
    row = [[getattr(payload, f) for f in FEATURES]]
    pred = int(model.predict(row)[0])                  # 0 or 1
    prob = float(model.predict_proba(row)[0][1])       # probability of diabetes
    label = "Diabetic" if pred == 1 else "Non-Diabetic"
    return {"prediction": pred, "label": label, "probability": prob}
