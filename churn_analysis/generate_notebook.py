import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Telco Customer Churn Prediction using Scikit-Learn Pipeline API\n",
                "\n",
                "## 1. Problem Statement & Objective\n",
                "Customer churn represents a major cost for subscription-based businesses, such as telecommunication providers. Retaining existing customers is typically much less expensive than acquiring new ones. \n",
                "\n",
                "**Objective:** Build a robust, end-to-end, reusable, and production-ready machine learning pipeline using the **Scikit-learn Pipeline API** to predict customer churn. This notebook demonstrates:\n",
                "- Automated data loading and cleaning.\n",
                "- Exploratory Data Analysis (EDA) and visualizations.\n",
                "- Clean feature preprocessing (scaling, imputation, and encoding) without leakage.\n",
                "- Model selection and hyperparameter tuning using `GridSearchCV`.\n",
                "- Model evaluation using accuracy, precision, recall, F1-score, and ROC-AUC.\n",
                "- Feature importance analysis.\n",
                "- Model serialization for production deployment."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "\n",
                "# Add the parent/src directory to sys.path to import our custom modules\n",
                "sys.path.append(os.path.abspath('../src'))\n",
                "sys.path.append(os.path.abspath('src'))\n",
                "\n",
                "from data_loader import load_churn_data\n",
                "from pipeline import build_full_pipeline, TotalChargesCleaner\n",
                "\n",
                "# Configure plotting aesthetics\n",
                "sns.set_theme(style=\"whitegrid\")\n",
                "plt.rcParams[\"figure.figsize\"] = (10, 6)\n",
                "plt.rcParams[\"font.size\"] = 12"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Dataset Loading\n",
                "We load the dataset using our modular `load_churn_data` function, which automatically downloads it from a public URL if it isn't cached locally."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df = load_churn_data()\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Dataset Summary:\")\n",
                "df.info()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Exploratory Data Analysis & Visualizations"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualization 1: Churn Distribution\n",
                "plt.figure(figsize=(6, 5))\n",
                "sns.countplot(data=df, x='Churn', palette='Set2')\n",
                "plt.title('Overall Churn Count Distribution')\n",
                "plt.xlabel('Customer Churn')\n",
                "plt.ylabel('Count')\n",
                "for p in plt.gca().patches:\n",
                "    height = p.get_height()\n",
                "    plt.gca().text(p.get_x() + p.get_width()/2., height + 50,\n",
                "                f'{height} ({height/len(df)*100:.1f}%)',\n",
                "                ha=\"center\")\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualization 2: Tenure vs Churn\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.histplot(data=df, x='tenure', hue='Churn', kde=True, multiple='stack', palette='coolwarm')\n",
                "plt.title('Distribution of Tenure (Months) by Churn Status')\n",
                "plt.xlabel('Tenure (months)')\n",
                "plt.ylabel('Count')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualization 3: Monthly Charges vs Churn\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.kdeplot(data=df[df['Churn'] == 'No'], x='MonthlyCharges', fill=True, label='No Churn', color='g')\n",
                "sns.kdeplot(data=df[df['Churn'] == 'Yes'], x='MonthlyCharges', fill=True, label='Churn', color='r')\n",
                "plt.title('Monthly Charges Distribution by Churn Status')\n",
                "plt.xlabel('Monthly Charges ($)')\n",
                "plt.ylabel('Density')\n",
                "plt.legend()\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Model Preprocessing & Training Pipeline\n",
                "We set up our train-test split and use the Scikit-learn Pipeline API. We will compare Logistic Regression and Random Forest classifiers, optimizing their hyperparameters using `GridSearchCV`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.model_selection import train_test_split\n",
                "\n",
                "# Separate Features and Target\n",
                "X = df.drop(columns=['customerID', 'Churn'])\n",
                "y = df['Churn'].map({'Yes': 1, 'No': 0})\n",
                "\n",
                "# Train-Test Split\n",
                "X_train, X_test, y_train, y_test = train_test_split(\n",
                "    X, y, test_size=0.2, random_state=42, stratify=y\n",
                ")\n",
                "\n",
                "print(f\"X_train shape: {X_train.shape}\")\n",
                "print(f\"X_test shape: {X_test.shape}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.model_selection import GridSearchCV\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.ensemble import RandomForestClassifier\n",
                "\n",
                "# Create training pipelines\n",
                "lr_pipeline = build_full_pipeline(LogisticRegression(max_iter=1000, random_state=42))\n",
                "rf_pipeline = build_full_pipeline(RandomForestClassifier(random_state=42))\n",
                "\n",
                "# Define grid search parameter bounds\n",
                "lr_param_grid = {\n",
                "    'classifier__C': [0.01, 0.1, 1.0, 10.0],\n",
                "    'classifier__solver': ['lbfgs', 'liblinear']\n",
                "}\n",
                "\n",
                "rf_param_grid = {\n",
                "    'classifier__n_estimators': [50, 100, 150],\n",
                "    'classifier__max_depth': [5, 10, None],\n",
                "    'classifier__min_samples_split': [2, 5]\n",
                "}\n",
                "\n",
                "# Perform GridSearchCV\n",
                "print(\"Tuning Logistic Regression...\")\n",
                "lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=5, scoring='f1', n_jobs=-1)\n",
                "lr_grid.fit(X_train, y_train)\n",
                "\n",
                "print(\"Tuning Random Forest...\")\n",
                "rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='f1', n_jobs=-1)\n",
                "rf_grid.fit(X_train, y_train)\n",
                "\n",
                "print(\"Hyperparameter Tuning Complete!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Model Evaluation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score\n",
                "\n",
                "models = {\n",
                "    'Logistic Regression': lr_grid.best_estimator_,\n",
                "    'Random Forest': rf_grid.best_estimator_\n",
                "}\n",
                "\n",
                "for name, pipeline in models.items():\n",
                "    y_pred = pipeline.predict(X_test)\n",
                "    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]\n",
                "    \n",
                "    print(f\"\\n=================== {name} Performance ===================\")\n",
                "    print(classification_report(y_test, y_pred))\n",
                "    \n",
                "    # Confusion Matrix\n",
                "    cm = confusion_matrix(y_test, y_pred)\n",
                "    plt.figure(figsize=(5, 4))\n",
                "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,\n",
                "                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])\n",
                "    plt.title(f'{name} Confusion Matrix')\n",
                "    plt.ylabel('Actual')\n",
                "    plt.xlabel('Predicted')\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ROC Curve comparison\n",
                "plt.figure(figsize=(8, 6))\n",
                "for name, pipeline in models.items():\n",
                "    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]\n",
                "    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)\n",
                "    auc = roc_auc_score(y_test, y_pred_proba)\n",
                "    plt.plot(fpr, tpr, label=f\"{name} (AUC = {auc:.4f})\")\n",
                "plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')\n",
                "plt.xlabel('False Positive Rate')\n",
                "plt.ylabel('True Positive Rate')\n",
                "plt.title('ROC Curves Comparison')\n",
                "plt.legend()\n",
                "plt.grid(True)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Feature Importances (Random Forest)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "best_rf = models['Random Forest']\n",
                "preprocessor = best_rf.named_steps['preprocessor']\n",
                "classifier = best_rf.named_steps['classifier']\n",
                "\n",
                "# Extract feature names and feature importances\n",
                "feature_names = preprocessor.get_feature_names_out()\n",
                "importances = classifier.feature_importances_\n",
                "\n",
                "feat_imp_df = pd.DataFrame({\n",
                "    'Feature': feature_names,\n",
                "    'Importance': importances\n",
                "}).sort_values(by='Importance', ascending=False)\n",
                "\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.barplot(data=feat_imp_df.head(15), x='Importance', y='Feature', palette='viridis')\n",
                "plt.title('Top 15 Feature Importances (Random Forest)')\n",
                "plt.xlabel('Importance')\n",
                "plt.ylabel('Feature')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Business Insights & Summary\n",
                "\n",
                "### Summary of Model Results:\n",
                "- Both **Logistic Regression** and **Random Forest** achieve solid baseline performance.\n",
                "- **Logistic Regression** typically offers a well-calibrated, high-accuracy baseline suitable for explanation, and performs strongly on Telco churn due to the linear relationship of features.\n",
                "- **Random Forest** captures non-linear interactions (e.g., tenure vs. payment method) and provides direct insights into feature importance.\n",
                "\n",
                "### Key Business Insights:\n",
                "1. **Contract Type Matters:** Customers with a **Month-to-month** contract are significantly more likely to churn compared to customers on one- or two-year contracts.\n",
                "2. **Tenure is Crucial:** Churn is highly concentrated in the first few months of customer tenure. Retention efforts should be heavily focused on onboarding and early customer experience.\n",
                "3. **Internet Services:** Customers using **Fiber Optic** internet service show a higher rate of churn, indicating potential pricing or reliability issues with this service category that need investigation.\n",
                "4. **Payment Method:** Customers using **Electronic check** churn at higher rates than automatic payment methods (Credit Card or Bank Transfer)."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

def generate_notebook():
    notebooks_dir = os.path.join(BASE_DIR, "notebooks")
    os.makedirs(notebooks_dir, exist_ok=True)
    notebook_path = os.path.join(notebooks_dir, "churn_analysis.ipynb")
    
    print(f"Generating notebook at {notebook_path}...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=2, ensure_ascii=False)
    print("Notebook successfully generated!")

if __name__ == "__main__":
    generate_notebook()
