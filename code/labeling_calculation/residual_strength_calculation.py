# final calculation for residual strength of all kind of attack v2
import cv2
import numpy as np
import pandas as pd
import os
import glob
from google.colab import drive

# Step 1: Mount Google Drive
drive.mount('/content/drive')

# Step 2: Define Paths
#/content/drive/MyDrive/Research/Research dataset/Final_Dataset/choliride attack/choliride attack segmented
#/content/drive/MyDrive/Research/Research dataset/Final_Dataset/choliride attack/chloride_attack_data_v2.csv
#/content/drive/MyDrive/Research/Research dataset/Final_Dataset/Sulphate attack/sulphate_segmented
#/content/drive/MyDrive/Research/Research dataset/Final_Dataset/Sulphate attack/sulphate_attack_data_V2.csv
segmented_save_dir = '/content/drive/MyDrive/Research/Research dataset/Final_Dataset/Sulphate attack/sulphate_segmented'
csv_save_path = '/content/drive/MyDrive/Research/Research dataset/Final_Dataset/Sulphate attack/sulphate_attack_data_V2.csv'

# Step 3: Fetch segmented image paths
image_paths = glob.glob(os.path.join(segmented_save_dir, '*'))
image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Found {len(image_paths)} segmented images.")

# Step 4: Define Color Conversion and Strength Calculation Functions
def rgb_to_xyz(r, g, b):
    """Convert RGB to CIE-XYZ."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    x = 0.41245 * r + 0.35758 * g + 0.18042 * b
    y = 0.21267 * r + 0.71516 * g + 0.07217 * b
    z = 0.01933 * r + 0.11919 * g + 0.95023 * b
    return x, y, z

def xyz_to_lab(x, y, z, x0=94.811, y0=100, z0=1017.304):
    """Convert CIE-XYZ to CIE-LAB with updated white reference values."""
    x, y, z = x / x0, y / y0, z / z0
    x = x ** (1/3) if x > 0.008856 else (7.787 * x + 16/116)
    y = y ** (1/3) if y > 0.008856 else (7.787 * y + 16/116)
    z = z ** (1/3) if z > 0.008856 else (7.787 * z + 16/116)
    l = 116 * y - 16 if y > 0.008856 else 903.3 * y
    a = 500 * (x - y)
    b = 200 * (y - z)
    return l, a, b

def calculate_residual_strength(l, b, reagent):
    """Calculate residual strength based on reagent using polynomial models."""
    if reagent == 'HCl':
        return max(0, 0.0083 * b**2 - 0.9127 * b + 39.311)
    elif reagent == 'NaCl':
        return max(0, -0.0183 * b**3 + 0.4962 * b**2 - 3.6565 * b + 43.551)
    elif reagent == 'H2SO4':
        return max(0, 0.0636 * b**3 - 1.2906 * b**2 + 6.1804 * b + 31.701)
    else:  # MgSO4
        return max(0, 0.116 * b**3 - 1.446 * b**2 + 5.336 * b + 32.897)

# Step 5: Process All Images
data = []

for image_path in image_paths:
    image_name = os.path.basename(image_path)
    print(f"Processing {image_name}...")

    # Load segmented image
    segmented_rgb = cv2.imread(image_path)
    if segmented_rgb is None:
        print(f"Skipping {image_name}: Could not load segmented image.")
        continue

    segmented_rgb = cv2.cvtColor(segmented_rgb, cv2.COLOR_BGR2RGB)

    # Create mask from non-zero pixels
    mask = np.any(segmented_rgb > 0, axis=2)
    valid_pixels = segmented_rgb[mask]

    if len(valid_pixels) == 0:
        print(f" - No valid pixels in {image_name}, skipping calculations.")
        continue

    l_values, b_values = [], []
    strengths_hcl, strengths_nacl, strengths_h2so4, strengths_mgso4 = [], [], [], []

    for r, g, b in valid_pixels:
        x, y, z = rgb_to_xyz(r, g, b)
        l, a, b_val = xyz_to_lab(x, y, z)
        l_values.append(l)
        b_values.append(b_val)

        strengths_hcl.append(calculate_residual_strength(l, b_val, 'HCl'))
        strengths_nacl.append(calculate_residual_strength(l, b_val, 'NaCl'))
        strengths_h2so4.append(calculate_residual_strength(l, b_val, 'H2SO4'))
        strengths_mgso4.append(calculate_residual_strength(l, b_val, 'MgSO4'))

    # Calculate average values
    avg_l = np.mean(l_values)
    avg_b = np.mean(b_values)
    avg_strength_hcl = np.mean(strengths_hcl)
    avg_strength_nacl = np.mean(strengths_nacl)
    avg_strength_h2so4 = np.mean(strengths_h2so4)
    avg_strength_mgso4 = np.mean(strengths_mgso4)

    print(f" - Average L*: {avg_l:.2f}")
    print(f" - Average b*: {avg_b:.2f}")
    print(f" - Average HCl Strength: {avg_strength_hcl:.2f}")
    print(f" - Average NaCl Strength: {avg_strength_nacl:.2f}")
    print(f" - Average H2SO4 Strength: {avg_strength_h2so4:.2f}")
    print(f" - Average MgSO4 Strength: {avg_strength_mgso4:.2f}")

    # Skip image if any strength is zero
    if 0 in [avg_strength_hcl, avg_strength_nacl, avg_strength_h2so4, avg_strength_mgso4]:
        print(f" - Skipping {image_name} due to zero residual strength in one or more reagents.")
        continue

    # 🚫 Skip image if MgSO4 strength is greater than 75
    if avg_strength_mgso4 > 75:
        print(f" - Skipping {image_name} due to MgSO4 residual strength > 75.")
        continue

    # Store the result for valid image
    data.append({
        'Image_Name': image_name,
        'Average_L*': avg_l,
        'Average_b*': avg_b,
        'Residual_Strength_HCl': avg_strength_hcl,
        'Residual_Strength_NaCl': avg_strength_nacl,
        'Residual_Strength_H2SO4': avg_strength_h2so4,
        'Residual_Strength_MgSO4': avg_strength_mgso4
    })

# Step 6: Save Results to CSV
df = pd.DataFrame(data)
df.to_csv(csv_save_path, index=False)
print(f"✅ Data saved to {csv_save_path}")
print(f"✅ Processed {len(data)} images successfully.")
print("✅ Sample Data Preview:")
print(df.head())