import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularMLP(nn.Module):
    def __init__(self, input_dim=8, output_dim=16):
        super(TabularMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class ImageCNN(nn.Module):
    def __init__(self, output_dim=16):
        super(ImageCNN, self).__init__()
        # Input: 3 x 128 x 128
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, output_dim)
        
    def forward(self, x):
        # 128x128 -> 64x64
        x = self.pool(F.relu(self.conv1(x)))
        # 64x64 -> 32x32
        x = self.pool(F.relu(self.conv2(x)))
        # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv3(x)))
        
        x = self.gap(x)
        x = torch.flatten(x, 1) # Squeeze to batch_size x 64
        x = F.relu(self.fc(x))
        return x

class MultimodalHousePricePredictor(nn.Module):
    def __init__(self, tab_input_dim=8, tab_out_dim=16, img_out_dim=16):
        super(MultimodalHousePricePredictor, self).__init__()
        self.tab_mlp = TabularMLP(input_dim=tab_input_dim, output_dim=tab_out_dim)
        self.img_cnn = ImageCNN(output_dim=img_out_dim)
        
        # Fused fully connected layers
        fused_dim = tab_out_dim + img_out_dim
        self.fc1 = nn.Linear(fused_dim, 16)
        self.fc2 = nn.Linear(16, 1)
        
    def forward(self, img, tab):
        tab_features = self.tab_mlp(tab)
        img_features = self.img_cnn(img)
        
        # Concatenate features along dimension 1 (features)
        fused = torch.cat((tab_features, img_features), dim=1)
        
        x = F.relu(self.fc1(fused))
        x = self.fc2(x) # Output single price in thousands
        return x.squeeze(1)

class TabularOnlyPredictor(nn.Module):
    def __init__(self, tab_input_dim=8, tab_out_dim=16):
        super(TabularOnlyPredictor, self).__init__()
        self.tab_mlp = TabularMLP(input_dim=tab_input_dim, output_dim=tab_out_dim)
        self.fc = nn.Linear(tab_out_dim, 1)
        
    def forward(self, img, tab):
        tab_features = self.tab_mlp(tab)
        x = self.fc(tab_features)
        return x.squeeze(1)

class ImageOnlyPredictor(nn.Module):
    def __init__(self, img_out_dim=16):
        super(ImageOnlyPredictor, self).__init__()
        self.img_cnn = ImageCNN(output_dim=img_out_dim)
        self.fc = nn.Linear(img_out_dim, 1)
        
    def forward(self, img, tab):
        img_features = self.img_cnn(img)
        x = self.fc(img_features)
        return x.squeeze(1)

if __name__ == "__main__":
    # Small test pass
    img = torch.randn(2, 3, 128, 128)
    tab = torch.randn(2, 8)
    
    model = MultimodalHousePricePredictor()
    pred = model(img, tab)
    print("Multimodal output shape:", pred.shape)
    
    model_tab = TabularOnlyPredictor()
    pred_tab = model_tab(img, tab)
    print("Tabular-only output shape:", pred_tab.shape)
    
    model_img = ImageOnlyPredictor()
    pred_img = model_img(img, tab)
    print("Image-only output shape:", pred_img.shape)
