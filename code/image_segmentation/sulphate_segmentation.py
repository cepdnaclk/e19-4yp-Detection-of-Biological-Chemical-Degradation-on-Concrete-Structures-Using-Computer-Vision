# # Step 1: Install Dependencies
# !pip install opencv-python-headless

# # Step 2: Import Libraries
# import cv2
# import numpy as np
# import os
# from google.colab import drive
# import glob

# # Step 3: Mount Google Drive
# drive.mount('/content/drive')

# # Step 4: Define Paths
# image_folder = '/content/drive/MyDrive/Research/Research dataset/Sulphate attack'
# segmented_save_dir = os.path.join(image_folder, 'segmented_white_only')
# os.makedirs(segmented_save_dir, exist_ok=True)

# # Step 5: Get all image paths
# image_paths = glob.glob(os.path.join(image_folder, '*'))
# image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
# print(f"Found {len(image_paths)} images.")

# # Step 6: Define White-Sulfate Segmentation Function
# def segment_white_sulfate(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         return None

#     image = cv2.resize(image, (512, 512))

#     # Convert to LAB and apply CLAHE for contrast enhancement
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     l_clahe = clahe.apply(l)
#     lab_clahe = cv2.merge((l_clahe, a, b))
#     image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

#     # Convert to HSV for white region isolation
#     hsv = cv2.cvtColor(image_clahe, cv2.COLOR_BGR2HSV)
#     lower_white = np.array([0, 0, 190])
#     upper_white = np.array([180, 50, 255])
#     white_mask = cv2.inRange(hsv, lower_white, upper_white)

#     # Morphological operations to clean mask
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
#     white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

#     # Remove small contours
#     mask_clean = np.zeros_like(white_mask)
#     contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area > 200:  # adjust to retain smaller sulfate spots
#             cv2.drawContours(mask_clean, [cnt], -1, 255, -1)

#     # Apply mask to the original RGB image
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     result = image_rgb.copy()
#     result[mask_clean == 0] = 0

#     return result

# # Step 7: Segment all images
# for img_path in image_paths:
#     img_name = os.path.basename(img_path)
#     print(f"Processing {img_name}...")

#     segmented = segment_white_sulfate(img_path)

#     if segmented is None:
#         print(f"Skipping {img_name} — could not load.")
#         continue

#     save_path = os.path.join(segmented_save_dir, img_name)
#     cv2.imwrite(save_path, cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
#     print(f"✅ Saved: {save_path}")

# print(f"\n🎯 White sulfate segmentation completed for {len(image_paths)} images.")

import cv2
import numpy as np
import os

def segment_sulphate_image(image_path):
    """
    Segments white sulphate-affected regions from an image.
    Returns a segmented RGB image with only sulphate regions.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None

    image = cv2.resize(image, (512, 512))

    # Convert to LAB and apply CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Convert to HSV and isolate white regions
    hsv = cv2.cvtColor(image_clahe, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 190])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Filter small areas
    mask_clean = np.zeros_like(white_mask)
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            cv2.drawContours(mask_clean, [cnt], -1, 255, -1)

    # Apply mask to RGB image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segmented = image_rgb.copy()
    segmented[mask_clean == 0] = 0

    return segmented

def save_segmented_image(segmented_rgb, save_path):
    """
    Saves the segmented RGB image to the specified path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR))
