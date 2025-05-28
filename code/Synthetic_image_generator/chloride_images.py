import cv2
import numpy as np
from PIL import Image, ImageFilter
import random
from skimage.util import random_noise
import matplotlib.pyplot as plt

# Load the base concrete image
sample_path = "concrete-image.jpg" 
sample_img = cv2.imread(sample_path)
sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

# Resize the image to output size
output_size = (800, 800)
base_texture = cv2.resize(sample_img, output_size)

# Copy base image to synthetic result
synthetic = base_texture.copy()

# Initialize cumulative mask to track overlaps
cumulative_mask = np.zeros((output_size[1], output_size[0]), dtype=np.float32)

# Number of clusters and patches
num_clusters = random.randint(1, 1)
cluster_centers = [(random.randint(300, output_size[0]-300), random.randint(200, output_size[1]-300)) for _ in range(num_clusters)]

# Define color stops for gradient
dark_center = np.array([94, 53, 28])   # Dark rust (center)
mid_color = np.array([187, 129, 59])    # Orange middle
edge_color = np.array([210, 191, 141])   # Salt deposit (edge)

def generate_path_mask(size, length=650, max_width=400):
    h, w = size
    mask = np.zeros((h, w), dtype=np.float32)
    x, y = int(w * 0.1), int(h // 2)
    points = [(x, y)]
    direction = np.pi / 8

    for _ in range(length):
        direction += (random.random() - 0.5) * 0.1
        step = 5
        x = np.clip(x + int(step * np.cos(direction)), 0, w-1)
        y = np.clip(y + int(step * np.sin(direction)), 0, h-1)
        points.append((x, y))

    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    for i, (px, py) in enumerate(points):
        width = max_width * (0.5 + 0.5 * np.sin(2 * np.pi * i / length)) * (0.7 + 0.6 * random.random())
        dist = np.sqrt((rr - py)**2 + (cc - px)**2)
        sigma = max(width / 3, 1e-5)
        contribution = np.exp(-(dist**2) / (2 * sigma**2))
        mask = np.maximum(mask, contribution)

    return np.clip(mask, 0, 1)

for cluster_x, cluster_y in cluster_centers:
    num_patches = random.randint(1,1)
    for _ in range(num_patches):
        patch_size = 800
        offset_x = cluster_x + int(np.random.normal(0, 350))
        offset_y = cluster_y + int(np.random.normal(0, 350))

        mask = generate_path_mask((patch_size, patch_size), length=750, max_width=50)
        m = mask[..., np.newaxis]

        # Suppress white formation on overlaps by darkening colors based on cumulative mask
        max_blend = 1  # Cap the effect of overlap
        overlap_region = cumulative_mask[offset_y - patch_size // 2: offset_y + patch_size // 2,
                                         offset_x - patch_size // 2: offset_x + patch_size // 2]

        if overlap_region.shape[:2] != mask.shape:
            # Handle border clipping
            x1 = max(0, offset_x - patch_size // 2)
            y1 = max(0, offset_y - patch_size // 2)
            x2 = min(output_size[0], x1 + patch_size)
            y2 = min(output_size[1], y1 + patch_size)
            mask_x1, mask_y1 = 0, 0
            mask_x2, mask_y2 = x2 - x1, y2 - y1

            if mask_x2 <= 0 or mask_y2 <= 0:
                continue

            mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]
            m = mask[..., np.newaxis]
            overlap_region = cumulative_mask[y1:y2, x1:x2]
        else:
            x1 = offset_x - patch_size // 2
            y1 = offset_y - patch_size // 2
            x2 = x1 + patch_size
            y2 = y1 + patch_size

        # Compute darkness factor (more overlap => darker)
        blend_strength = np.clip(overlap_region / max_blend, 0, 1)[..., np.newaxis]

        # Base interpolated patch
        base_patch = np.where(m > 0.5,
                              dark_center * (1 - (m - 0.5) * 2) + mid_color * ((m - 0.5) * 2),
                              mid_color * (1 - m * 2) + edge_color * (m * 2))
        base_patch = np.where(m == 0, edge_color, base_patch)

        # Add extra darkening on overlaps (gradual)
        final_patch = base_patch * (1 - blend_strength) + dark_center * blend_strength
        final_patch = np.clip(final_patch, 0, 255).astype(np.uint8)

        mask_cropped = m
        patch_cropped = final_patch
        base_region = synthetic[y1:y2, x1:x2].astype(np.float32)
        blended = base_region * (1 - mask_cropped) + patch_cropped * mask_cropped
        synthetic[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        # Update cumulative mask
        cumulative_mask[y1:y2, x1:x2] += mask.squeeze()

# Add speckle noise (grainy appearance)
def add_grain_noise(img):
    noisy = random_noise(img / 255.0, mode='speckle', mean=0.05, var=0.02)
    return (noisy * 255).astype(np.uint8)

synthetic = add_grain_noise(synthetic)

# Apply erosion and Gaussian blur for blending effect
synthetic_img = Image.fromarray(synthetic)
synthetic_img = synthetic_img.filter(ImageFilter.MinFilter(3))
synthetic_img = synthetic_img.filter(ImageFilter.GaussianBlur(1.5))

# Save and show
output_path = "chloride_attack_realistic.png"
synthetic_img.save(output_path)
synthetic_img.show()
print(f"✅ Saved more realistic chloride attack simulation: {output_path}")
