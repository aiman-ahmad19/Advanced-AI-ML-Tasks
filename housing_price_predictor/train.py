import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import HousingDataset
from model import TabularOnlyPredictor, ImageOnlyPredictor, MultimodalHousePricePredictor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, tabulars, prices in dataloader:
        images = images.to(device)
        tabulars = tabulars.to(device)
        prices = prices.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, tabulars)
        loss = criterion(outputs, prices)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, tabulars, prices in dataloader:
            images = images.to(device)
            tabulars = tabulars.to(device)
            prices = prices.to(device)
            
            outputs = model(images, tabulars)
            loss = criterion(outputs, prices)
            running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def run_training():
    # 1. Create Directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # 2. Split Dataset
    csv_file = "data/housing_data.csv"
    img_dir = "data/images"
    df = pd.read_csv(csv_file)
    
    # 70% Train, 15% Val, 15% Test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Save splits for consistency during evaluation
    train_csv = "data/train.csv"
    val_csv = "data/val.csv"
    test_csv = "data/test.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"Data split saved: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # 3. Instantiate Datasets and DataLoaders
    # Train fits the scaler
    train_dataset = HousingDataset(csv_file=train_csv, img_dir=img_dir, is_train=True)
    scaler = train_dataset.get_scaler()
    
    # Val and Test use the train scaler
    val_dataset = HousingDataset(csv_file=val_csv, img_dir=img_dir, scaler=scaler, is_train=False)
    test_dataset = HousingDataset(csv_file=test_csv, img_dir=img_dir, scaler=scaler, is_train=False)
    
    # Save scaler for future use (e.g. testing or deployment)
    scaler_path = "models/tabular_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Fitted scaler saved to {scaler_path}")
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Define Models to Train
    models_config = {
        "tabular_only": TabularOnlyPredictor(),
        "image_only": ImageOnlyPredictor(),
        "multimodal": MultimodalHousePricePredictor()
    }
    
    epochs = 25
    lr = 0.001
    weight_decay = 1e-4
    
    history = {}
    
    for name, model in models_config.items():
        print(f"\n==========================================")
        print(f"Training Model: {name}")
        print(f"==========================================")
        
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate_epoch(model, val_loader, criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Print epoch logs
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {train_loss:8.2f} | Val MSE: {val_loss:8.2f}")
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = f"models/{name}_best.pth"
                torch.save(model.state_dict(), checkpoint_path)
                
        history[name] = {
            "train": train_losses,
            "val": val_losses,
            "best_val_mse": best_val_loss,
            "best_val_rmse": np.sqrt(best_val_loss)
        }
        print(f"Finished {name} training. Best Val RMSE: {np.sqrt(best_val_loss):.2f} ($k)")
        
    # 5. Save training history
    history_path = "models/training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    # 6. Plot Loss Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_range = range(1, epochs + 1)
    
    for idx, (name, metrics) in enumerate(history.items()):
        ax = axes[idx]
        ax.plot(epochs_range, metrics["train"], label="Train MSE", color="royalblue", linewidth=2)
        ax.plot(epochs_range, metrics["val"], label="Val MSE", color="darkorange", linewidth=2)
        ax.set_title(f"{name.replace('_', ' ').title()} Loss", fontsize=14, fontweight='bold')
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel("MSE (Price in $k^2$)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=11)
        
    plt.tight_layout()
    plot_path = "plots/loss_curves.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nLoss curves saved to {plot_path}")

if __name__ == "__main__":
    run_training()
