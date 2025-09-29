# *Title: Predicting Diabetes Using Logistic Regression.*

## Goal: Build a machine-learning model that predicts whether a patient is diabetic based on health measurements (glucose, BMI, age, etc.).

## **Steps to be Carried Out**

### 1. Environment Setup
- Install Python 3.11+
- Install libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`
- Launch Jupyter Notebook

### 2. Data Exploration (EDA)
- Load CSV with pandas
- Inspect `df.info()`, `df.describe()`, check for missing values
- Plot histograms & boxplots for distributions/outliers
- Create correlation heatmap with `seaborn.heatmap`

### 3. Data Cleaning
- Handle missing or zero values (e.g., blood pressure = 0)
- Scale/normalize features if needed (`StandardScaler`)

### 4. Model Building
- Train/Test split (e.g., 80/20)
- Baseline model: `LogisticRegression` from `sklearn`
- Evaluate accuracy, precision, recall, F1-score, confusion matrix

### 5. Model Improvement
- Hyper-parameter tuning with `GridSearchCV`
- Try alternative models (Random Forest, XGBoost) for comparison

### 6. Visualization & Insights
- Plot ROC curve & calculate AUC
- Show feature importance or logistic regression coefficients

### 7. Documentation
- Explain project purpose
- Note dataset source
- Provide steps to reproduce
- Add key results and screenshots of plots
