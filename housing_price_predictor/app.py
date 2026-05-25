import os
import random
import joblib
import pandas as pd
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, render_template
from torchvision import transforms

from src.dataset import HousingDataset
from src.model import TabularOnlyPredictor, ImageOnlyPredictor, MultimodalHousePricePredictor

app = Flask(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for models and scaler
models = {}
scaler = None
test_df = None

# Image transformation
image_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def init_resources():
    global scaler, test_df
    # Load Scaler
    scaler_path = "models/tabular_scaler.joblib"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully.")
    else:
        print("Warning: Scaler not found. Run training first.")

    # Load Test CSV to query sample houses
    test_csv = "data/test.csv"
    if os.path.exists(test_csv):
        test_df = pd.read_csv(test_csv)
        print("Test data loaded successfully.")
    else:
        print("Warning: Test dataset csv not found.")

    # Load Models
    model_paths = {
        "tabular_only": (TabularOnlyPredictor(), "models/tabular_only_best.pth"),
        "image_only": (ImageOnlyPredictor(), "models/image_only_best.pth"),
        "multimodal": (MultimodalHousePricePredictor(), "models/multimodal_best.pth")
    }

    for name, (model_arch, path) in model_paths.items():
        if os.path.exists(path):
            model_arch.load_state_dict(torch.load(path, map_location=device))
            model_arch.to(device)
            model_arch.eval()
            models[name] = model_arch
            print(f"Model {name} loaded successfully.")
        else:
            print(f"Warning: Checkpoint {path} not found.")

# Initialize resources
init_resources()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory('plots', filename)

@app.route('/data/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('data/images', filename)

@app.route('/api/samples', methods=['GET'])
def get_samples():
    """Returns a list of 12 random houses from the test set for UI selection."""
    if test_df is None:
        return jsonify({"error": "Test dataset not available"}), 500
    
    # Select 12 random sample records
    samples = test_df.sample(n=min(12, len(test_df)), random_state=42).copy()
    
    # Map quality names
    quality_map = {1: "Budget", 2: "Standard", 3: "Premium"}
    samples['quality_name'] = samples['quality'].map(quality_map)
    
    return jsonify(samples.to_dict(orient='records'))

@app.route('/api/predict', methods=['POST'])
def predict():
    if not models:
        return jsonify({"error": "Models are not loaded on server."}), 500
    
    try:
        # 1. Parse Inputs
        # Supports JSON or multipart/form-data
        if request.is_json:
            data = request.json
            house_id = data.get('house_id')
            area = float(data.get('area'))
            bedrooms = float(data.get('bedrooms'))
            bathrooms = float(data.get('bathrooms'))
            stories = float(data.get('stories'))
            has_pool = float(data.get('has_pool'))
            quality = int(data.get('quality'))
            image_name = data.get('image_name') # For pre-loaded samples
        else:
            house_id = request.form.get('house_id')
            area = float(request.form.get('area'))
            bedrooms = float(request.form.get('bedrooms'))
            bathrooms = float(request.form.get('bathrooms'))
            stories = float(request.form.get('stories'))
            has_pool = float(request.form.get('has_pool'))
            quality = int(request.form.get('quality'))
            image_name = None

        # 2. Get/Process Image
        # If there's an uploaded file
        if 'image' in request.files:
            file = request.files['image']
            image = Image.open(file.stream).convert('RGB')
        # If selecting a pre-loaded sample house
        elif house_id is not None:
            img_path = f"data/images/house_{int(house_id)}.png"
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
            else:
                return jsonify({"error": f"Image file for house {house_id} not found."}), 404
        else:
            return jsonify({"error": "No image or house selection provided."}), 400

        # Apply image transformations
        img_tensor = image_transform(image).unsqueeze(0).to(device) # Shape: 1, 3, 128, 128

        # 3. Process Tabular features
        # Continuous: [area, bedrooms, bathrooms, stories]
        num_features = np.array([[area, bedrooms, bathrooms, stories]])
        if scaler is not None:
            scaled_num = scaler.transform(num_features)
        else:
            scaled_num = num_features
            
        # One-hot quality: levels [1, 2, 3]
        quality_onehot = [0.0, 0.0, 0.0]
        if quality in [1, 2, 3]:
            quality_onehot[quality - 1] = 1.0
            
        # Combine: scaled_num, has_pool, quality_onehot
        tab_features = np.hstack([scaled_num[0], [float(has_pool)], quality_onehot])
        tab_tensor = torch.tensor(tab_features, dtype=torch.float32).unsqueeze(0).to(device)

        # 4. Perform Predictions
        predictions = {}
        with torch.no_grad():
            for name, model in models.items():
                pred = model(img_tensor, tab_tensor)
                # Rescale back from $k to actual dollars
                predictions[name] = float(pred.item() * 1000.0)

        # 5. Extract actual price if query matches test dataset
        actual_price = None
        if house_id is not None and test_df is not None:
            match = test_df[test_df['house_id'] == int(house_id)]
            if not match.empty:
                actual_price = float(match.iloc[0]['price'])

        return jsonify({
            "predictions": predictions,
            "actual_price": actual_price
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Returns test set evaluation results and training history."""
    metrics_path = "models/evaluation_results.csv"
    history_path = "models/training_history.json"
    
    data = {}
    
    if os.path.exists(metrics_path):
        m_df = pd.read_csv(metrics_path, index_col=0)
        data["metrics"] = m_df.to_dict(orient='index')
        
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            data["history"] = json.load(f)
            
    return jsonify(data)

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
