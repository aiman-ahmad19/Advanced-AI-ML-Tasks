# Telco Customer Churn Prediction: End-to-End ML Pipeline

This repository contains a production-ready, modular machine learning pipeline built using the **Scikit-learn Pipeline API** to predict customer churn using the popular **Telco Customer Churn dataset** (originally published by IBM).

---

## Objective
The primary objective of this task is to design and implement a reusable, robust, and clean machine learning pipeline that:
1. **Automates Preprocessing:** Clean numerical and categorical columns, handles missing/blank inputs without data leakage.
2. **Trains & Tunes Models:** Trains both **Logistic Regression** and **Random Forest** classifiers, and tunes their hyperparameters using **GridSearchCV** to optimize the F1-Score.
3. **Exports for Production:** Exports the full end-to-end model (preprocessing + classifier) into a single serialized file using **joblib**, ensuring that raw data can be scored directly in production.
4. **Ensures Robustness:** Employs unit tests to verify that data cleaning and transformer pipelines run as expected.

---

## Project Structure
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv (Downloaded automatically)
├── models/
│   └── churn_pipeline_model.joblib          (Exported serialized pipeline)
├── plots/                                   (Model evaluation plots)
│   ├── logistic_regression_confusion_matrix.png
│   ├── random_forest_confusion_matrix.png
│   ├── rf_feature_importances.png
│   └── roc_curve_comparison.png
├── src/
│   ├── __init__.py
│   ├── data_loader.py                       (Automates downloading/caching dataset)
│   ├── pipeline.py                          (Defines preprocessing and ML pipelines)
│   ├── train.py                             (GridSearch tuning, evaluation, plots, export)
│   ├── predict.py                           (Inference module for new raw customers)
│   └── generate_notebook.py                 (Generates the exploratory Jupyter Notebook)
├── notebooks/
│   └── churn_analysis.ipynb                 (Jupyter Notebook containing EDA and insights)
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py                     (Automated unit tests)
├── README.md                                (Project walkthrough & documentation)
└── requirements.txt                         (Project dependencies)
```

---

## Installation & Setup
1. Open your terminal/command prompt and navigate to the project directory:
   ```bash
   cd "C:\Users\Admin\Desktop\AI internship task\task2"
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Methodology & Approach
### 1. Advanced Preprocessing Pipeline (`pipeline.py`)
To prevent train-test leakage and streamline deployment, all data transformations are wrapped within a Scikit-learn `ColumnTransformer`:
- **Numerical Features** (`tenure`, `MonthlyCharges`): Imputed with `SimpleImputer(strategy='median')` and scaled using `StandardScaler()`.
- **Special Numeric Feature** (`TotalCharges`): Since some rows representing new customers (tenure = 0) contain empty spaces (`" "`) instead of numeric values, we built a custom `TotalChargesCleaner` class. It converts strings to float, coerces errors to `NaN`, replaces `NaN` with `0.0`, and applies `StandardScaler()`.
- **Categorical Features** (16 variables): Imputed using `SimpleImputer(strategy='most_frequent')` and encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.

### 2. Model Training & Hyperparameter Tuning (`train.py`)
We perform 5-fold cross-validation with **GridSearchCV** targeting **F1-Score** optimization (since predicting churn is an imbalanced classification problem, optimizing F1-Score balances precision and recall).
- **Logistic Regression Hyperparameters:**
  - `classifier__C`: `[0.01, 0.1, 1.0, 10.0]`
  - `classifier__solver`: `['lbfgs', 'liblinear']`
- **Random Forest Hyperparameters:**
  - `classifier__n_estimators`: `[50, 100, 150]`
  - `classifier__max_depth`: `[5, 10, None]`
  - `classifier__min_samples_split`: `[2, 5]`

### 3. Verification & Inference (`test_pipeline.py` & `predict.py`)
- We include automated unit tests using the Python `unittest` framework to verify preprocessing behaves correctly under blank or noisy input.
- `predict.py` takes raw dictionaries (representing new, unscored customers) and directly passes them to the serialized pipeline, producing predicted classes and risk probabilities.

---

## Key Results & Observations

*Note: The following metrics are evaluated on the held-out test set (20% split, stratified).*

### Model Comparison Summary

| Metric | Logistic Regression | Random Forest |
| :--- | :---: | :---: |
| **Accuracy** | **~80.4%** | ~79.6% |
| **Precision (Class 1)** | **~64.3%** | ~63.8% |
| **Recall (Class 1)** | **~55.6%** | ~50.8% |
| **F1-Score (Class 1)** | **~59.7%** | ~56.6% |
| **ROC-AUC** | **~0.849** | ~0.842 |

### Key Observations & Insights:
1. **Best Model:** **Logistic Regression** slightly outperformed Random Forest in both F1-score and ROC-AUC for this dataset. This suggests that the relationship between the features and customer churn is largely linear.
2. **Contract Type is the Strongest Predictor:** Customers on **Month-to-month** contracts are highly vulnerable to churn. Multi-year contracts act as a massive stabilizer for customer retention.
3. **Tenure Impact:** Customer churn is highly clustered during the first 6 months of the customer lifetime. Early-stage onboarding and targeted promotions are essential.
4. **Internet Services:** Customers using **Fiber Optic** internet service show high churn rates, indicating pricing complaints or service issues.
5. **Billing & Payments:** Paperless billing and Electronic checks are correlated with higher churn rates compared to automatic bank/card payments.

---

## How to Run the Project

### 1. Run the Training & Tuning Script
Trains models, tunes hyperparameters, plots charts, and exports the best pipeline:
```bash
python src/train.py
```
This will generate and save ROC curves, feature importances, and confusion matrices into the `plots/` directory.

### 2. Run Test Predictions
Run inference on sample mock customer profiles:
```bash
python src/predict.py
```

### 3. Run Automated Unit Tests
Verify data transformation functions:
```bash
python -m unittest discover -s tests
```

### 4. Generate the Jupyter Notebook
Generate the interactive exploratory notebook:
```bash
python src/generate_notebook.py
```
Once generated, you can open and run `notebooks/churn_analysis.ipynb` using Jupyter Notebook or VS Code to visualize the results interactively.

