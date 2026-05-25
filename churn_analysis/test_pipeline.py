import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add src to system path for running unit tests directly
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pipeline import TotalChargesCleaner, get_preprocessing_pipeline

class TestPipeline(unittest.TestCase):
    def test_total_charges_cleaner(self):
        """Test that TotalChargesCleaner handles float strings, spaces, and empty strings."""
        cleaner = TotalChargesCleaner()
        df = pd.DataFrame({'TotalCharges': ['123.45', ' ', '456.7', '']})
        
        cleaned = cleaner.fit_transform(df)
        expected = np.array([[123.45], [0.0], [456.7], [0.0]])
        
        np.testing.assert_array_almost_equal(cleaned, expected)

    def test_pipeline_preprocessing(self):
        """Test that the full ColumnTransformer handles a mock dataframe successfully."""
        mock_data = pd.DataFrame({
            'gender': ['Female', 'Male'],
            'SeniorCitizen': [0, 1],
            'Partner': ['Yes', 'No'],
            'Dependents': ['No', 'Yes'],
            'tenure': [10, 20],
            'PhoneService': ['Yes', 'No'],
            'MultipleLines': ['No', 'No phone service'],
            'InternetService': ['DSL', 'Fiber optic'],
            'OnlineSecurity': ['Yes', 'No'],
            'OnlineBackup': ['No', 'Yes'],
            'DeviceProtection': ['Yes', 'No'],
            'TechSupport': ['No', 'Yes'],
            'StreamingTV': ['Yes', 'No'],
            'StreamingMovies': ['No', 'Yes'],
            'Contract': ['Month-to-month', 'One year'],
            'PaperlessBilling': ['Yes', 'No'],
            'PaymentMethod': ['Electronic check', 'Mailed check'],
            'MonthlyCharges': [50.5, 75.0],
            'TotalCharges': ['505.0', '1500.0']
        })
        
        preprocessor = get_preprocessing_pipeline()
        transformed = preprocessor.fit_transform(mock_data)
        
        # Verify it transformed successfully (should have 2 rows)
        self.assertEqual(transformed.shape[0], 2)
        # Verify there are no NaNs in the transformed matrix
        self.assertFalse(np.isnan(transformed).any())

if __name__ == "__main__":
    unittest.main()
