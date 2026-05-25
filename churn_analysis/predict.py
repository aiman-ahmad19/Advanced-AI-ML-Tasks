import os
import joblib
import pandas as pd

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline_model.joblib")

def predict_churn(customer_data):
    """
    Predicts churn and probabilities for a list of customer data records.
    
    Parameters:
    customer_data (list of dicts or pd.DataFrame): Customer records.
    
    Returns:
    pd.DataFrame: Original records appended with 'Churn_Prediction' and 'Churn_Probability'.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run train.py first.")
        
    print(f"Loading serialized model pipeline from {MODEL_PATH}...")
    pipeline = joblib.load(MODEL_PATH)
    
    # Convert input to DataFrame if it's a list of dicts
    if isinstance(customer_data, list):
        df_input = pd.DataFrame(customer_data)
    elif isinstance(customer_data, pd.DataFrame):
        df_input = customer_data.copy()
    else:
        raise ValueError("customer_data must be a list of dictionaries or a pandas DataFrame.")
        
    # Ensure all required features are present
    required_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
        'MonthlyCharges', 'TotalCharges'
    ]
    
    for feat in required_features:
        if feat not in df_input.columns:
            # Handle missing feature by raising error or filling defaults
            raise ValueError(f"Required feature '{feat}' is missing from the input data.")
            
    # Run predictions
    predictions = pipeline.predict(df_input)
    probabilities = pipeline.predict_proba(df_input)[:, 1]
    
    # Append results
    df_result = df_input.copy()
    df_result['Churn_Prediction'] = predictions
    df_result['Churn_Probability'] = probabilities
    df_result['Churn_Label'] = df_result['Churn_Prediction'].map({1: 'Churn', 0: 'No Churn'})
    
    return df_result

if __name__ == "__main__":
    # Sample mock customer data
    sample_customers = [
        {
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 1,
            'PhoneService': 'No',
            'MultipleLines': 'No phone service',
            'InternetService': 'DSL',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 29.85,
            'TotalCharges': '29.85'
        },
        {
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 34,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'DSL',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'No',
            'DeviceProtection': 'Yes',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'One year',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Mailed check',
            'MonthlyCharges': 56.95,
            'TotalCharges': '1889.5'
        },
        {
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 0, # Empty total charges example
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'No',
            'OnlineSecurity': 'No internet service',
            'OnlineBackup': 'No internet service',
            'DeviceProtection': 'No internet service',
            'TechSupport': 'No internet service',
            'StreamingTV': 'No internet service',
            'StreamingMovies': 'No internet service',
            'Contract': 'Two year',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Mailed check',
            'MonthlyCharges': 20.0,
            'TotalCharges': ' '
        }
    ]
    
    try:
        results = predict_churn(sample_customers)
        print("\nPredictions for sample customers:")
        for idx, row in results.iterrows():
            print(f"Customer {idx+1}: Churn Probability: {row['Churn_Probability']:.4f} -> Prediction: {row['Churn_Label']}")
    except FileNotFoundError as e:
        print(f"Could not run test predictions: {e}")
    except Exception as e:
        print(f"Error during prediction: {e}")
