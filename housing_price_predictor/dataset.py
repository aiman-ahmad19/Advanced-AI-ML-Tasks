import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

class HousingDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, scaler=None, is_train=True):
        """
        Args:
            csv_file (string): Path to the csv file with housing info.
            img_dir (string): Directory with all the house images.
            transform (callable, optional): Optional transform to be applied on an image.
            scaler (dict, optional): Dictionary containing fitted StandardScaler objects for numerical features.
            is_train (bool): Whether this dataset is for training (fits the scaler).
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.is_train = is_train
        
        # Define image transformation if not provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # Numerical features to scale
        self.num_cols = ['area', 'bedrooms', 'bathrooms', 'stories']
        # Categorical features to encode
        self.cat_cols = ['quality']
        # Binary features (already 0/1)
        self.bin_cols = ['has_pool']
        
        # Preprocess tabular data
        self.preprocess_tabular(scaler)
        
    def preprocess_tabular(self, scaler):
        # One-hot encode quality (levels: 1, 2, 3)
        quality_dummies = pd.get_dummies(self.df['quality'], prefix='quality')
        # Ensure all quality levels (1, 2, 3) are represented, even if not in this split
        for q in [1, 2, 3]:
            col_name = f'quality_{q}'
            if col_name not in quality_dummies.columns:
                quality_dummies[col_name] = 0
        quality_dummies = quality_dummies[[f'quality_{q}' for q in [1, 2, 3]]].astype(float)
        
        # Continuous numerical features
        num_features = self.df[self.num_cols].values
        
        if self.is_train:
            self.scaler = StandardScaler()
            scaled_num = self.scaler.fit_transform(num_features)
        else:
            self.scaler = scaler
            if self.scaler is not None:
                scaled_num = self.scaler.transform(num_features)
            else:
                scaled_num = num_features # Fallback if no scaler provided
                
        # Binary features
        bin_features = self.df[self.bin_cols].values.astype(float)
        
        # Combine tabular features
        # Columns will be: [scaled_area, scaled_bedrooms, scaled_bathrooms, scaled_stories, has_pool, quality_1, quality_2, quality_3]
        self.tabular_features = np.hstack([scaled_num, bin_features, quality_dummies.values])
        
        # Target variable: Price (in thousands of dollars for regression stability)
        self.prices = (self.df['price'].values / 1000.0).astype(np.float32)
        
    def get_scaler(self):
        return self.scaler

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_id = int(self.df.iloc[idx]['house_id'])
        img_path = os.path.join(self.img_dir, f"house_{img_id}.png")
        
        # Ensure image is in RGB format
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Get tabular features
        tabular = torch.tensor(self.tabular_features[idx], dtype=torch.float32)
        
        # Get price
        price = torch.tensor(self.prices[idx], dtype=torch.float32)
        
        return image, tabular, price

if __name__ == "__main__":
    # Test script if run directly
    dataset = HousingDataset(csv_file="data/housing_data.csv", img_dir="data/images")
    print(f"Dataset length: {len(dataset)}")
    img, tab, price = dataset[0]
    print(f"Image tensor shape: {img.shape}")
    print(f"Tabular tensor: {tab}")
    print(f"Price (in $k): {price.item()} ($ {price.item()*1000:.2f})")
