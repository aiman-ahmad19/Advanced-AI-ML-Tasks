import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

def generate_house_image(row, image_path):
    """
    Generates a 128x128 image of a house representing the tabular features:
    - stories: 1, 2, or 3 (determines house height)
    - quality: 1 (Budget: tan), 2 (Standard: red), 3 (Premium: dark slate)
    - bedrooms: 1 to 5 (determines number of windows)
    - has_pool: 0 or 1 (draws a pool on the lawn)
    """
    width, height = 128, 128
    img = Image.new("RGB", (width, height), "#87CEEB") # Sky blue background
    draw = ImageDraw.Draw(img)
    
    # 1. Draw Sun
    draw.ellipse([100, 10, 115, 25], fill="#FFD700")
    
    # 2. Draw Grass (ground)
    draw.rectangle([0, 80, width, height], fill="#2E8B57") # Sea green grass
    
    # Extract row info
    stories = int(row['stories'])
    quality = int(row['quality'])
    bedrooms = int(row['bedrooms'])
    has_pool = int(row['has_pool'])
    
    # 3. Determine House dimensions and coordinates
    # Base is at y=80
    house_left = 32
    house_right = 96
    
    if stories == 1:
        house_top = 50
    elif stories == 2:
        house_top = 35
    else: # stories == 3
        house_top = 20
        
    # House body colors based on quality
    if quality == 1:
        body_color = "#D2B48C" # Tan (Budget)
    elif quality == 2:
        body_color = "#CD5C5C" # Indian Red (Standard)
    else:
        body_color = "#2F4F4F" # Dark Slate Gray (Premium)
        
    # Draw House Body
    draw.rectangle([house_left, house_top, house_right, 80], fill=body_color, outline="#333333", width=1)
    
    # 4. Draw Roof
    roof_color = "#5C4033" # Dark Brown
    roof_peak_x = (house_left + house_right) // 2
    roof_peak_y = house_top - 15
    draw.polygon([(house_left - 4, house_top), (roof_peak_x, roof_peak_y), (house_right + 4, house_top)], 
                 fill=roof_color, outline="#333333")
    
    # 5. Draw Door (centered on the first floor)
    door_width = 14
    door_height = 18
    door_left = roof_peak_x - door_width // 2
    door_right = roof_peak_x + door_width // 2
    door_top = 80 - door_height
    draw.rectangle([door_left, door_top, door_right, 80], fill="#5C2C16", outline="#333333")
    # Door knob
    draw.ellipse([door_right - 3, door_top + 9, door_right - 1, door_top + 11], fill="#FFD700")
    
    # 6. Define Window positions (x_center, y_center)
    # Windows are grouped by floors
    window_slots = []
    
    # Floor 1 slots (left and right of door)
    window_slots.extend([(44, 70), (84, 70)])
    
    # Floor 2 slots (if stories >= 2)
    if stories >= 2:
        window_slots.extend([(44, 50), (84, 50), (64, 50)])
        
    # Floor 3 slots (if stories == 3)
    if stories == 3:
        window_slots.extend([(44, 30), (84, 30), (64, 30)])
        
    # Draw windows up to number of bedrooms (cap at available slots)
    num_windows = min(bedrooms, len(window_slots))
    for i in range(num_windows):
        cx, cy = window_slots[i]
        w_size = 5 # half-width of window
        w_left = cx - w_size
        w_right = cx + w_size
        w_top = cy - w_size
        w_bottom = cy + w_size
        
        # Draw window frame (light blue glass)
        draw.rectangle([w_left, w_top, w_right, w_bottom], fill="#E0FFFF", outline="#333333")
        # Draw cross panes
        draw.line([cx, w_top, cx, w_bottom], fill="#333333")
        draw.line([w_left, cy, w_right, cy], fill="#333333")
        
    # 7. Draw Swimming Pool
    if has_pool == 1:
        # Draw blue pool in the foreground lawn
        pool_left = 98
        pool_top = 90
        pool_right = 124
        pool_bottom = 112
        draw.ellipse([pool_left, pool_top, pool_right, pool_bottom], fill="#1E90FF", outline="#00BFFF", width=1)
        
    img.save(image_path)

def generate_dataset(num_samples=1000, seed=42):
    """
    Generates tabular data and corresponding house images, saving them to disk.
    """
    np.random.seed(seed)
    
    # Generate tabular features
    area = np.random.uniform(1000, 5000, num_samples).astype(int)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], num_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05])
    
    # Bathrooms generally correlate with bedrooms
    bathrooms = []
    for b in bedrooms:
        max_bath = min(b, 4)
        bathrooms.append(np.random.choice(list(range(1, max_bath + 1))))
    bathrooms = np.array(bathrooms)
    
    stories = np.random.choice([1, 2, 3], num_samples, p=[0.45, 0.45, 0.1])
    has_pool = np.random.choice([0, 1], num_samples, p=[0.75, 0.25])
    quality = np.random.choice([1, 2, 3], num_samples, p=[0.3, 0.5, 0.2])
    
    # Calculate price based on a formula with some noise
    # Base price: $100k
    # Area: $150 per sqft
    # Bedrooms: $20k each
    # Bathrooms: $15k each
    # Stories: $30k each
    # Pool: $50k
    # Quality: $75k for standard (2), $150k for premium (3)
    base_price = 100000
    noise = np.random.normal(0, 10000, num_samples)
    
    prices = (
        base_price
        + area * 150
        + bedrooms * 20000
        + bathrooms * 15000
        + stories * 30000
        + has_pool * 50000
        + (quality - 1) * 75000
        + noise
    )
    # Keep prices rounded to nearest integer
    prices = np.round(prices).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'house_id': np.arange(num_samples),
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'has_pool': has_pool,
        'quality': quality,
        'price': prices
    })
    
    # Create folders
    os.makedirs("data/images", exist_ok=True)
    
    # Save CSV
    csv_path = "data/housing_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Tabular dataset saved to {csv_path}")
    
    # Generate images
    print("Generating images...")
    for idx, row in df.iterrows():
        img_path = f"data/images/house_{int(row['house_id'])}.png"
        generate_house_image(row, img_path)
        if (idx + 1) % 200 == 0:
            print(f"Generated {idx + 1}/{num_samples} images")
            
    print("Dataset generation completed successfully!")

if __name__ == "__main__":
    generate_dataset(num_samples=1000)
