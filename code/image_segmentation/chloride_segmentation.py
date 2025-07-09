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