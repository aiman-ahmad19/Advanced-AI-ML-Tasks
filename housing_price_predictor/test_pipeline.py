import os
import unittest
import torch
import pandas as pd
from PIL import Image

from src.data_generator import generate_house_image, generate_dataset
from src.dataset import HousingDataset
from src.model import MultimodalHousePricePredictor, TabularOnlyPredictor, ImageOnlyPredictor

class TestHousingMLPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate a small dataset for testing purposes
        cls.test_csv = "data/test_housing_data.csv"
        cls.test_img_dir = "data/test_images"
        
        # Create directories if they don't exist
        os.makedirs(cls.test_img_dir, exist_ok=True)
        
        # Create a single synthetic row for drawing test
        cls.dummy_row = pd.Series({
            'house_id': 9999,
            'area': 2500,
            'bedrooms': 3,
            'bathrooms': 2,
            'stories': 2,
            'has_pool': 1,
            'quality': 2,
            'price': 450000
        })
        cls.dummy_image_path = os.path.join(cls.test_img_dir, "house_9999.png")
        
    def test_image_generation(self):
        # Generate single house image
        generate_house_image(self.dummy_row, self.dummy_image_path)
        self.assertTrue(os.path.exists(self.dummy_image_path))
        
        # Open image and check dimensions
        img = Image.open(self.dummy_image_path)
        self.assertEqual(img.size, (128, 128))
        self.assertEqual(img.mode, "RGB")
        
    def test_dataset_loader(self):
        # Make sure CSV exists
        main_csv = "data/housing_data.csv"
        main_img_dir = "data/images"
        
        if not os.path.exists(main_csv):
            # Fallback to generating a mini dataset
            generate_dataset(num_samples=10, seed=42)
            
        dataset = HousingDataset(csv_file=main_csv, img_dir=main_img_dir, is_train=True)
        self.assertGreater(len(dataset), 0)
        
        img_tensor, tab_tensor, price_target = dataset[0]
        
        # Check shapes and types
        self.assertEqual(img_tensor.shape, (3, 128, 128))
        self.assertEqual(tab_tensor.shape, (8,))
        self.assertIsInstance(price_target.item(), float)
        
    def test_model_forward_passes(self):
        batch_size = 4
        img = torch.randn(batch_size, 3, 128, 128)
        tab = torch.randn(batch_size, 8)
        
        # Test Multimodal
        model = MultimodalHousePricePredictor()
        pred = model(img, tab)
        self.assertEqual(pred.shape, (batch_size,))
        
        # Test Tabular-only
        model_tab = TabularOnlyPredictor()
        pred_tab = model_tab(img, tab)
        self.assertEqual(pred_tab.shape, (batch_size,))
        
        # Test Image-only
        model_img = ImageOnlyPredictor()
        pred_img = model_img(img, tab)
        self.assertEqual(pred_img.shape, (batch_size,))

if __name__ == "__main__":
    unittest.main()
