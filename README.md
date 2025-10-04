Predicting Diabetes Using Logistic Regression
📌 Goal

Build a machine-learning model that predicts whether a patient is diabetic based on health measurements such as glucose, BMI, age, blood pressure, insulin, etc.

⚙️ 1. Environment Setup

Python 3.9+ (tested on 3.9.10, also compatible with 3.10/3.12)


Install required libraries:

pip install -r requirements.txt


Recommended tools: VS Code / Jupyter Notebook

🔍 2. Data Exploration (EDA)

Dataset: Pima Indians Diabetes Dataset

Key steps:

df.info(), df.describe() to inspect data

Handle missing values (e.g., blood pressure = 0)

Histograms, boxplots for distributions/outliers

Correlation heatmap with seaborn.heatmap

🧹 3. Data Cleaning

Replace invalid/zero values with median or imputed values

Normalize features using StandardScaler

Split into train/test (80/20 split)

🤖 4. Model Building

Baseline: Logistic Regression (sklearn)

Evaluation metrics: accuracy, precision, recall, F1, ROC–AUC

Improvements:

Hyperparameter tuning with GridSearchCV

Compared against Random Forest, XGBoost

### 📊 Performance Metrics

| Metric                  | Score |
|-------------------------|------:|
| **Accuracy**            | **71%** |
| **Precision**           | 61% |
| **Recall (Sensitivity)**| 52% |
| **F1-score**            | 56% |
| **ROC–AUC**             | 81% |

### 📌 Interpretation
- **Accuracy** → Overall fraction of correct predictions  
- **Precision** → Of all predicted diabetics, ~61% were actually diabetic  
- **Recall (Sensitivity)** → Model identified ~52% of the true diabetic cases  
- **F1-score** → Harmonic mean of precision and recall  
- **ROC–AUC** → Probability that the model ranks a random positive higher than a random negative case  

---

### 📈 Visualizations
Below are the final evaluation plots generated from the test set:

| Confusion Matrix | ROC Curve |
|------------------|-----------|
| ![Confusion Matrix](documents/confusion_matrix.png) | ![ROC Curve](documents/roc_curve.png) |


	
🚀 6. API Integration (FastAPI)

This project now includes an API to make predictions via HTTP requests.

Run the API server:
uvicorn src.api:app --reload


Server will start at: http://127.0.0.1:8000

Docs (Swagger UI): http://127.0.0.1:8000/docs

Example Request (JSON):
{
  "Pregnancies": 2,
  "Glucose": 150,
  "BloodPressure": 85,
  "SkinThickness": 25,
  "Insulin": 100,
  "BMI": 30.5,
  "DiabetesPedigreeFunction": 0.45,
  "Age": 42
}

🧪 7. Test Client Script

You can test predictions without Swagger UI using test_client.py:

python test_client.py


Output:

Status code: 200
Response: {"prediction": "Diabetic"}

📂 Project Structure
diabetes-prediction/
│── data/                     # Dataset
│── models/                   # Trained models (.pkl files)
│── notebooks/                # Jupyter notebooks (EDA, training, etc.)
│── src/                      # API + utilities
│   ├── api.py                # FastAPI app
│   ├── model.py              # Model loading & prediction
│   └── utils.py              # Helper functions
│── test_client.py            # Python script to test API
│── requirements.txt          # Dependencies
│── README.md                 # Documentation
│── .gitignore                # Ignore unnecessary files

📌 How to Reproduce

Clone this repo:

```bash
git clone https://github.com/Surajkumar123-commits/diabetes-prediction.git
cd diabetes-prediction

Install dependencies

pip install -r requirements.txt

Run Jupyter Notebooks for EDA/model building

Start API server with FastAPI

Test with test_client.py

## 🧠 Model Comparison and Results

Three models — **Logistic Regression**, **Random Forest**, and **XGBoost** — were trained and evaluated on the Diabetes dataset.  
Performance was measured using key classification metrics: **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC AUC**.

| Metric | Logistic Regression | Random Forest | XGBoost |
|:-------|:--------------------:|:--------------:|:--------:|
| **Accuracy** | 0.7078 | 0.7662 | **0.7727** |
| **Precision** | 0.6000 | **0.6957** | 0.6939 |
| **Recall (Sensitivity)** | 0.5000 | 0.5926 | **0.6296** |
| **F1-Score** | 0.5455 | 0.6400 | **0.6602** |
| **ROC AUC** | 0.8074 | **0.8239** | 0.8035 |

---

### 📊 Interpretation of Results

- **Logistic Regression:**  
  Baseline interpretable model. Simpler but with lower accuracy and recall.

- **Random Forest:**  
  Improved performance, best **ROC AUC (0.8239)**, and robust generalization.

- **XGBoost:**  
  Achieved **highest accuracy (77.3%)**, best **recall**, and overall **strong F1-score**.  
  Ideal for identifying diabetic patients with fewer false negatives.

---

### 🏁 **Conclusion**

> Based on the comparison, **XGBoost** delivers the best overall performance and is recommended for deployment.  
> **Random Forest** is a strong secondary option with excellent AUC performance.  
> **Logistic Regression** remains valuable as a lightweight, interpretable baseline.

---

📎 *Next Step:*  
Model tuning can be performed using **GridSearchCV** or **Optuna** to further optimize hyperparameters for Random Forest and XGBoost.

---

🔑 Key points fixed:  
- ✅ Added **triple backticks with `bash`** for commands.  
- ✅ Removed extra blank lines.  
- ✅ Clear step-by-step flow.

---

👉 If you copy this into your `README.md`, it will render beautifully on GitHub with **highlighted command blocks**.  

Do you want me to generate the **entire README.md final version** (all sections merged with this polish) so you can directly replace your current file?
