import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class TotalChargesCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer to clean the 'TotalCharges' column.
    It converts blank spaces and empty strings to NaN, casts the series to float,
    and imputes missing values with a default of 0.0 (since empty total charges
    correspond to tenure = 0).
    """
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Handle X if it is a DataFrame or a numpy array
        if isinstance(X, pd.DataFrame):
            col = X.iloc[:, 0]
        else:
            col = pd.Series(X.ravel())
        
        # Convert to numeric, coerce errors to NaN (converts spaces to NaN)
        col_numeric = pd.to_numeric(col, errors='coerce')
        # Fill NaN with 0.0 (since empty TotalCharges are for 0 tenure customers)
        col_numeric = col_numeric.fillna(0.0)
        
        # Reshape to 2D column array for scikit-learn pipelines
        return col_numeric.values.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(['TotalCharges'], dtype=object)
        return np.array(input_features, dtype=object)

def get_preprocessing_pipeline():
    """
    Constructs the preprocessor using ColumnTransformer.
    """
    # Numerical features to scale directly
    num_features = ['tenure', 'MonthlyCharges']
    # Numerical feature that needs custom string cleaning first
    clean_num_feature = ['TotalCharges']
    # Categorical features to encode
    cat_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]

    # Preprocessing pipelines for numeric columns
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing pipeline for TotalCharges
    total_charges_transformer = Pipeline(steps=[
        ('cleaner', TotalChargesCleaner()),
        ('scaler', StandardScaler())
    ])

    # Preprocessing pipeline for categorical columns
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Bundle preprocessing for all features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('total_charges', total_charges_transformer, clean_num_feature),
            ('cat', cat_transformer, cat_features)
        ],
        remainder='drop'  # Drop other columns like customerID
    )

    return preprocessor

def build_full_pipeline(classifier):
    """
    Combines the preprocessing pipeline and a classifier into a single Pipeline object.
    """
    preprocessor = get_preprocessing_pipeline()
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    return full_pipeline

if __name__ == "__main__":
    # Test constructing the pipeline
    from sklearn.ensemble import RandomForestClassifier
    pipeline = build_full_pipeline(RandomForestClassifier())
    print("Pipeline constructed successfully:")
    print(pipeline)
