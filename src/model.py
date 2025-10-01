from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .utils import FEATURES, TARGET, load_diabetes_csv

# -------- Paths --------
BASE_DIR = Path(__file__).resolve().parents[1]   # project root
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
MODEL_PATH = BASE_DIR / "models" / "diabetes_model.pkl"

# -------- Pipeline --------
def build_pipeline() -> Pipeline:
    """Create a simple, solid baseline pipeline."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, solver="lbfgs"))
    ])

# -------- Train + Save --------
def train_and_save(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> None:
    """Train the model from CSV and save to models/diabetes_model.pkl."""
    df: pd.DataFrame = load_diabetes_csv(str(data_path))
    X = df[FEATURES]
    y = df[TARGET]

    pipe = build_pipeline()
    pipe.fit(X, y)

    # ensure "models" directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    print(f"✅ Model trained and saved to {model_path}")

if __name__ == "__main__":
    train_and_save()
