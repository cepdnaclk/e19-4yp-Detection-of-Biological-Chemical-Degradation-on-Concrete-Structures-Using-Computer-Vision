# #Chloride segmentation
# import cv2
# import numpy as np
# import os
# from google.colab import drive
# import glob

# # Step 1: Mount Google Drive
# drive.mount('/content/drive')

# # Step 2: Define Paths
# image_folder = '/content/drive/MyDrive/Research/Research dataset/choliride attack'  # Input images
# segmented_save_dir = '/content/drive/MyDrive/Research/Research dataset/choliride attack/choliride attack segmented'  # Save segmented

# # Create segmented folder if it doesn’t exist
# os.makedirs(segmented_save_dir, exist_ok=True)

# # Step 3: Fetch image paths
# image_paths = glob.glob(os.path.join(image_folder, '*'))
# image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
# print(f"Found {len(image_paths)} images.")

# # Step 4: Define Segmentation Function
# def segment_image(image_path):
#     """Improved segmentation for chloride-affected areas with better water and concrete exclusion."""
#     image = cv2.imread(image_path)
#     if image is None:
#         return None

#     image = cv2.resize(image, (512, 512))
#     image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     # Mask water areas (expanded range to include green to blue)
#     lower_water = np.array([50, 50, 50])
#     upper_water = np.array([120, 255, 255])
#     water_mask = cv2.inRange(image_hsv, lower_water, upper_water)

#     # Mask concrete areas (low saturation, neutral colors)
#     lower_concrete = np.array([0, 0, 20])
#     upper_concrete = np.array([180, 30, 200])
#     concrete_mask = cv2.inRange(image_hsv, lower_concrete, upper_concrete)

#     # Mask rust areas (reddish-brown, adjusted thresholds)
#     lower_rust = np.array([0, 50, 50])
#     upper_rust = np.array([70, 255, 255])
#     rust_mask = cv2.inRange(image_hsv, lower_rust, upper_rust)

#     # Remove water and concrete from rust mask
#     rust_mask[water_mask > 0] = 0
#     rust_mask[concrete_mask > 0] = 0

#     # Morphological clean-up (more aggressive to remove noise)
#     kernel = np.ones((5,5), np.uint8)
#     rust_mask = cv2.erode(rust_mask, kernel, iterations=2)
#     rust_mask = cv2.dilate(rust_mask, kernel, iterations=3)

#     rust_mask = rust_mask.astype(bool)

#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     segmented_rgb = image_rgb.copy()
#     segmented_rgb[~rust_mask] = 0

#     return segmented_rgb

# # Step 5: Process All Images and Save Segmented Results
# for image_path in image_paths:
#     image_name = os.path.basename(image_path)
#     print(f"Processing {image_name}...")

#     # Segment the image
#     segmented_rgb = segment_image(image_path)

#     # Skip if the image couldn't be loaded
#     if segmented_rgb is None:
#         print(f"Skipping {image_name}: Could not load image.")
#         continue

#     # Save the segmented image
#     segmented_save_path = os.path.join(segmented_save_dir, image_name)
#     cv2.imwrite(segmented_save_path, cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR))
#     print(f"Saved segmented image to {segmented_save_path}")

# print(f"✅ Processed {len(image_paths)} images successfully.")
# print(f"Segmented images saved to {segmented_save_dir}")

import cv2
import numpy as np
import os

def segment_chloride_image(image_path):
    """
    Segment chloride-affected regions in the image.
    Returns a segmented RGB image where non-corroded areas are blacked out.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    image = cv2.resize(image, (512, 512))
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define masks
    lower_water = np.array([50, 50, 50])
    upper_water = np.array([120, 255, 255])
    water_mask = cv2.inRange(image_hsv, lower_water, upper_water)

    lower_concrete = np.array([0, 0, 20])
    upper_concrete = np.array([180, 30, 200])
    concrete_mask = cv2.inRange(image_hsv, lower_concrete, upper_concrete)

    lower_rust = np.array([0, 50, 50])
    upper_rust = np.array([70, 255, 255])
    rust_mask = cv2.inRange(image_hsv, lower_rust, upper_rust)

    # Exclude water & concrete from rust areas
    rust_mask[water_mask > 0] = 0
    rust_mask[concrete_mask > 0] = 0

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    rust_mask = cv2.erode(rust_mask, kernel, iterations=2)
    rust_mask = cv2.dilate(rust_mask, kernel, iterations=3)
    rust_mask = rust_mask.astype(bool)

    # Apply mask
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segmented_rgb = image_rgb.copy()
    segmented_rgb[~rust_mask] = 0

    return segmented_rgb


def save_segmented_image(segmented_rgb, save_path):
    """
    Saves a segmented RGB image to the given path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR))