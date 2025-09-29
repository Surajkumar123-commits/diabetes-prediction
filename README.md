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


## Results

The logistic regression model was trained and evaluated on the **Pima Indians Diabetes Dataset**.

### Performance Metrics
| Metric                  | Score |
|-------------------------|------:|
| **Accuracy**            | **0.71** |
| **Precision**           | 0.61 |
| **Recall (Sensitivity)**| 0.52 |
| **F1-score**            | 0.56 |
| **ROC–AUC**             | 0.81 |

* **Accuracy** – Overall fraction of correct predictions.  
* **Precision** – Of all predicted diabetics, ~61 % were actually diabetic.  
* **Recall (Sensitivity)** – Model identified ~52 % of the true diabetic cases.  
* **F1-score** – Harmonic mean of precision and recall.  
* **ROC–AUC** – Probability that the model ranks a random positive higher than a random negative case.

### Visualizations
Below are the final evaluation plots generated from the test set:

| Confusion Matrix | ROC Curve |
|------------------|-----------|
| ![Confusion Matrix](documents/confusion_matrix.png) | ![ROC Curve](documents/roc_curve.png) |

*(Save your screenshots of these plots inside the `documents/` folder and adjust the paths if needed.)*

### How to Reproduce
1. **Clone the repository**  
   ```bash
   git clone https://github.com/<your-username>/diabetes-prediction.git
   cd diabetes-prediction
