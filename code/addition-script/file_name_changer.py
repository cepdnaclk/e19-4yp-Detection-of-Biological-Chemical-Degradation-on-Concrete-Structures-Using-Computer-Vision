import os
import re

# Configuration
image_folder = './docs/images/Sulphate attack/'   # Adjust to your image folder
offset = 400

# Go through all files in the folder
for old_name in os.listdir(image_folder):
    old_path = os.path.join(image_folder, old_name)
    if not os.path.isfile(old_path):
        continue

    # Updated regex for 'images_8.jpg', 'images_22.jpeg', etc.
    match = re.match(r'(image_)(\d+)(\.\w+)', old_name, re.IGNORECASE)
    if match:
        prefix, number, extension = match.groups()
        new_number = int(number) + offset
        new_name = f"{prefix}{new_number}{extension.lower()}"
        new_path = os.path.join(image_folder, new_name)

        if not os.path.exists(new_path):
            print(f"Renaming {old_name} -> {new_name}")
            os.rename(old_path, new_path)
        else:
            print(f"Skipped {old_name} -> {new_name} already exists")
    else:
        print(f"Skipped (no match): {old_name}")
