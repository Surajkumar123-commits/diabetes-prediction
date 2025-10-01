from typing import List
import pandas as pd

# Keep one place for the feature order your model expects.
FEATURES: List[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET: str = "Outcome"

def load_diabetes_csv(path: str) -> pd.DataFrame:
    """Load the CSV as a DataFrame."""
    return pd.read_csv(path)
