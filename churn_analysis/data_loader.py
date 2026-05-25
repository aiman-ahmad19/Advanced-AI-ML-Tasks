import os
import urllib.request
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_FILE_PATH = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

def download_dataset():
    """Downloads the Telco Churn dataset from the raw URL to the data directory if not already present."""
    if not os.path.exists(DATA_DIR):
        print(f"Creating directory: {DATA_DIR}")
        os.makedirs(DATA_DIR)
    
    if not os.path.exists(DATA_FILE_PATH):
        print(f"Downloading dataset from {DATA_URL}...")
        try:
            urllib.request.urlretrieve(DATA_URL, DATA_FILE_PATH)
            print(f"Dataset successfully downloaded and saved to {DATA_FILE_PATH}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise e
    else:
        print(f"Dataset already exists locally at {DATA_FILE_PATH}")

def load_churn_data():
    """Loads the churn dataset as a pandas DataFrame, downloading it first if necessary."""
    download_dataset()
    print(f"Loading dataset from {DATA_FILE_PATH}...")
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    # Test loading the data
    df = load_churn_data()
    print(df.head())
