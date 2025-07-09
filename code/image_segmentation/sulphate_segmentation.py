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