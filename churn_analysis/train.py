import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve
)

from data_loader import load_churn_data
from pipeline import build_full_pipeline

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def train_and_evaluate():
    # 1. Load Data
    df = load_churn_data()
    
    # 2. Separate Features and Target
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # 4. Define pipelines
    lr_pipeline = build_full_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    rf_pipeline = build_full_pipeline(RandomForestClassifier(random_state=42))
    
    # 5. Hyperparameter Grids
    lr_param_grid = {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__solver': ['lbfgs', 'liblinear']
    }
    
    rf_param_grid = {
        'classifier__n_estimators': [50, 100, 150],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5]
    }
    
    # 6. Grid Search for Logistic Regression
    print("\nTuning Logistic Regression model...")
    lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=5, scoring='f1', n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    print(f"Logistic Regression - Best Params: {lr_grid.best_params_}")
    print(f"Logistic Regression - Best Cross-Val F1-Score: {lr_grid.best_score_:.4f}")
    
    # 7. Grid Search for Random Forest
    print("\nTuning Random Forest model...")
    rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='f1', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    print(f"Random Forest - Best Params: {rf_grid.best_params_}")
    print(f"Random Forest - Best Cross-Val F1-Score: {rf_grid.best_score_:.4f}")
    
    # 8. Evaluate both on test set
    models = {
        'Logistic Regression': lr_grid.best_estimator_,
        'Random Forest': rf_grid.best_estimator_
    }
    
    results = {}
    
    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'Pipeline': pipeline
        }
        
        print(f"\n=================== {name} Evaluation ===================")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {auc:.4f}")
        
        # Save confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
        plt.title(f'{name} Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{name.lower().replace(' ', '_')}_confusion_matrix.png"))
        plt.close()
        
    # 9. Plot comparison curves (ROC Curve)
    plt.figure(figsize=(8, 6))
    for name, pipeline in models.items():
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['ROC-AUC']:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curve_comparison.png"))
    plt.close()
    
    # 10. Extract Feature Importance for Random Forest
    try:
        best_rf = models['Random Forest']
        classifier = best_rf.named_steps['classifier']
        preprocessor = best_rf.named_steps['preprocessor']
        
        # Get feature names out
        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_
        
        # Map back to names
        feat_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        print("\nTop 15 Feature Importances (Random Forest):")
        print(feat_imp_df.head(15))
        
        # Save feature importance plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_imp_df.head(15), x='Importance', y='Feature', palette='viridis')
        plt.title('Top 15 Feature Importances - Random Forest')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "rf_feature_importances.png"))
        plt.close()
    except Exception as e:
        print(f"\nCould not extract feature importances: {e}")
        
    # 11. Find best model (based on F1-Score) and export
    best_model_name = max(results, key=lambda k: results[k]['F1-Score'])
    best_pipeline = results[best_model_name]['Pipeline']
    
    print(f"\nSaving the best model ({best_model_name}) based on F1-Score...")
    model_save_path = os.path.join(MODELS_DIR, "churn_pipeline_model.joblib")
    joblib.dump(best_pipeline, model_save_path)
    print(f"Model successfully saved to {model_save_path}")
    
    # Write a quick JSON summary of findings for reports
    summary_df = pd.DataFrame(results).T.drop(columns=['Pipeline'])
    print("\nModel Performance Summary:")
    print(summary_df)

if __name__ == "__main__":
    train_and_evaluate()
