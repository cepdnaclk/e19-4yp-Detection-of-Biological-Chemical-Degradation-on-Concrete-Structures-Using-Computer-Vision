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
output_size = (2048, 2048)
base_texture = cv2.resize(sample_img, output_size)

# Copy base image to synthetic result
synthetic = base_texture.copy()

# Number of clusters and patches
num_clusters = random.randint(2, 3)
cluster_centers = [(random.randint(200, output_size[0]-200), random.randint(200, output_size[1]-200)) for _ in range(num_clusters)]

# Visualize one of the diffusion masks
visualized = False

for cluster_x, cluster_y in cluster_centers:
    num_patches = random.randint(5, 10)
    for _ in range(num_patches):
        offset_x = cluster_x + int(np.random.normal(0, 1500))  # closer patches
        offset_y = cluster_y + int(np.random.normal(0, 50))
        patch_radius = random.randint(600, 700)
        patch_size = patch_radius * 2
        sulphate_color = (230, 230, 220)

        # Generate smooth circular diffusion mask
        y, x = np.ogrid[:patch_size, :patch_size]
        center = (patch_radius, patch_radius)
        distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        sigma = patch_radius / 3 # larger sigma for smoother edges
        mask = np.exp(-(distance**2) / (2 * sigma**2))
        mask = np.clip(mask, 0, 1)  # Ensure mask is between 0 and 1
        mask[distance > patch_radius] = 0  # Outside the circle: zero

        # Optional: visualize one mask
        if not visualized:
            plt.figure(figsize=(4, 4))
            plt.imshow(mask, cmap='gray')
            plt.title(f"Sample Diffusion Mask (radius={patch_radius}, sigma={sigma:.1f})")
            plt.colorbar()
            plt.show()
            visualized = True

        patch = np.ones((patch_size, patch_size, 3), dtype=np.float32) * sulphate_color  # Keep in float32 for blending

        # Calculate placement boundaries
        x1 = max(0, offset_x - patch_radius)
        y1 = max(0, offset_y - patch_radius)
        x2 = min(output_size[0], x1 + patch_size)
        y2 = min(output_size[1], y1 + patch_size)

        mask_x1 = 0
        mask_y1 = 0
        mask_x2 = x2 - x1
        mask_y2 = y2 - y1

        if mask_x2 > 0 and mask_y2 > 0:
            mask_cropped = mask[mask_y1:mask_y2, mask_x1:mask_x2]
            patch_cropped = patch[mask_y1:mask_y2, mask_x1:mask_x2]
            alpha = mask_cropped[..., np.newaxis]

            # Convert base to float for blending
            base_patch = synthetic[y1:y2, x1:x2].astype(np.float32)

            # Blend with smooth alpha
            blended_patch = base_patch * (1 - alpha) + patch_cropped * alpha

            # Replace in synthetic image
            synthetic[y1:y2, x1:x2] = np.clip(blended_patch, 0, 255).astype(np.uint8)

# Add grain noise
def add_grain_noise(img):
    noisy = random_noise(img / 255.0, mode='speckle', mean=0.03, var=0.01)
    return (noisy * 255).astype(np.uint8)

synthetic = add_grain_noise(synthetic)

# Apply erosion and light blur for realism
synthetic_img = Image.fromarray(synthetic)
synthetic_img = synthetic_img.filter(ImageFilter.MinFilter(3))
synthetic_img = synthetic_img.filter(ImageFilter.GaussianBlur(2))

# Save and show
output_path = "sulphate_on_concrete_smooth_diffusion.png"
synthetic_img.save(output_path)
synthetic_img.show()
print(f"✅ Saved simulated sulphate patches with smooth diffusion: {output_path}")
