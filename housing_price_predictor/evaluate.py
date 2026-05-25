import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset import HousingDataset
from model import TabularOnlyPredictor, ImageOnlyPredictor, MultimodalHousePricePredictor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_predictions(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, tabulars, prices in dataloader:
            images = images.to(device)
            tabulars = tabulars.to(device)
            
            outputs = model(images, tabulars)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(prices.numpy())
            
    return np.array(all_preds), np.array(all_targets)

def run_evaluation():
    # 1. Load scaler and test dataset
    scaler_path = "models/tabular_scaler.joblib"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please run train.py first.")
        
    scaler = joblib.load(scaler_path)
    test_csv = "data/test.csv"
    img_dir = "data/images"
    
    test_dataset = HousingDataset(csv_file=test_csv, img_dir=img_dir, scaler=scaler, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 2. Define models to load
    models_config = {
        "tabular_only": (TabularOnlyPredictor(), "models/tabular_only_best.pth"),
        "image_only": (ImageOnlyPredictor(), "models/image_only_best.pth"),
        "multimodal": (MultimodalHousePricePredictor(), "models/multimodal_best.pth")
    }
    
    results = {}
    predictions_dict = {}
    targets = None
    
    for name, (model, checkpoint_path) in models_config.items():
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}. Skipping...")
            continue
            
        print(f"Loading {name} model from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)
        
        preds, targets = get_predictions(model, test_loader, device)
        
        # Scale back to actual price (from thousands to raw dollars)
        preds_dollars = preds * 1000.0
        targets_dollars = targets * 1000.0
        
        # Calculate metrics in original dollars for better interpretability
        mae = mean_absolute_error(targets_dollars, preds_dollars)
        mse = mean_squared_error(targets_dollars, preds_dollars)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets_dollars, preds_dollars)
        
        results[name] = {
            "MAE ($)": mae,
            "RMSE ($)": rmse,
            "R2 Score": r2
        }
        
        predictions_dict[name] = preds_dollars
        
    if len(results) == 0:
        print("No models were evaluated because checkpoints were missing.")
        return
        
    targets_dollars = targets * 1000.0
    
    # 3. Create results dataframe
    results_df = pd.DataFrame(results).T
    print("\n=============================================")
    print("           TEST SET EVALUATION METRICS       ")
    print("=============================================")
    print(results_df.to_string(formatters={
        "MAE ($)": lambda x: f"${x:,.2f}",
        "RMSE ($)": lambda x: f"${x:,.2f}",
        "R2 Score": lambda x: f"{x:.4f}"
    }))
    print("=============================================\n")
    
    # Save metrics to csv
    results_df.to_csv("models/evaluation_results.csv")
    
    # Save predictions alongside actual price
    pred_df = pd.DataFrame({"Actual Price": targets_dollars})
    for name, preds in predictions_dict.items():
        pred_df[f"{name}_predicted"] = preds
    pred_df.to_csv("models/predictions_comparison.csv", index=False)
    
    # 4. Generate visualizations
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Bar Chart comparing MAE and RMSE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    model_names = [n.replace("_", " ").title() for n in results.keys()]
    maes = [results[m]["MAE ($)"] for m in results.keys()]
    rmses = [results[m]["RMSE ($)"] for m in results.keys()]
    
    # MAE Chart
    sns.barplot(x=model_names, y=maes, ax=ax1, hue=model_names, palette="viridis", legend=False)
    ax1.set_title("Mean Absolute Error (MAE) - Lower is Better", fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel("MAE (in USD $)", fontsize=12)
    # Add values on top of bars
    for i, v in enumerate(maes):
        ax1.text(i, v + (max(maes)*0.01), f"${v:,.0f}", ha='center', va='bottom', fontweight='bold', fontsize=11)
        
    # RMSE Chart
    sns.barplot(x=model_names, y=rmses, ax=ax2, hue=model_names, palette="magma", legend=False)
    ax2.set_title("Root Mean Squared Error (RMSE) - Lower is Better", fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel("RMSE (in USD $)", fontsize=12)
    for i, v in enumerate(rmses):
        ax2.text(i, v + (max(rmses)*0.01), f"${v:,.0f}", ha='center', va='bottom', fontweight='bold', fontsize=11)
        
    plt.tight_layout()
    metrics_plot_path = "plots/metrics_comparison.png"
    plt.savefig(metrics_plot_path, dpi=150)
    plt.close()
    print(f"Metrics comparison plot saved to {metrics_plot_path}")
    
    # Plot 2: Scatter Plot of Predictions vs Actual
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = ["royalblue", "darkorange", "forestgreen"]
    min_val = min(targets_dollars) * 0.9
    max_val = max(targets_dollars) * 1.1
    
    for idx, (name, preds) in enumerate(predictions_dict.items()):
        ax = axes[idx]
        ax.scatter(targets_dollars, preds, alpha=0.6, color=colors[idx], edgecolors='w', s=40)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Predictor")
        
        ax.set_title(f"{name.replace('_', ' ').title()}\n$R^2$: {results[name]['R2 Score']:.4f}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Actual Price ($)", fontsize=12)
        ax.set_ylabel("Predicted Price ($)", fontsize=12)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        
    plt.tight_layout()
    scatter_plot_path = "plots/predictions_scatter.png"
    plt.savefig(scatter_plot_path, dpi=150)
    plt.close()
    print(f"Scatter prediction plot saved to {scatter_plot_path}")
    
    # 5. Output key insight
    print("\n=============================================")
    print("                 INSIGHTS                    ")
    print("=============================================")
    best_model = min(results.keys(), key=lambda x: results[x]["RMSE ($)"])
    print(f"The best performing model is: **{best_model.replace('_', ' ').title()}**")
    
    if "multimodal" in results and "tabular_only" in results:
        improvement_mae = results["tabular_only"]["MAE ($)"] - results["multimodal"]["MAE ($)"]
        pct_improvement = (improvement_mae / results["tabular_only"]["MAE ($)"]) * 100
        print(f"Multimodal learning reduced the MAE by ${improvement_mae:,.2f} ({pct_improvement:.2f}%) compared to tabular-only baseline.")
    print("=============================================\n")

if __name__ == "__main__":
    run_evaluation()
