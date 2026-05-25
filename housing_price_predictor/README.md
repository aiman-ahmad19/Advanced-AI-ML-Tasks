# Multimodal House Price Predictor

A multimodal AI application that predicts house prices using both tabular data (square footage, bedrooms, bathrooms, etc.) and visual data (house images).

## What's in this project?

- **Flask Web Application** (`app.py`): Serves a web interface for interacting with the prediction models
- **Pre-trained Models** (`models/`):
  - `tabular_only_best.pth`: MLP model using only tabular features
  - `image_only_best.pth`: CNN model using only house images
  - `multimodal_best.pth`: Fusion model combining both tabular and image features
- **Dataset** (`data/`): 
  - House images (`images/`)
  - Tabular data in CSV format
- **Source Code** (`src/`): Model definitions, training, and evaluation scripts
- **Visualizations** (`plots/`): Training curves and evaluation metrics

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Create and activate a virtual environment (recommended)**

   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies**

   ```powershell
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Flask server**

   ```powershell
   python app.py
   ```

2. **Open your browser**

   Go to: `http://localhost:5000`

## Features

- **Real-time Predictions**: Get price predictions from three different models
- **Sample House Selection**: Choose from pre-loaded sample houses
- **Custom Inputs**: Adjust square footage, bedrooms, bathrooms, stories, quality, and pool status
- **Model Comparison**: View performance metrics (MAE, RMSE, R²) for all models
- **Training History**: See loss curves from model training

## Project Structure

```
task3/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── data/                   # Dataset (images and CSV files)
├── models/                 # Pre-trained models and scaler
├── src/                    # Source code (model, training, evaluation)
├── templates/              # HTML templates
├── plots/                  # Visualization plots
└── tests/                  # Test files
```

