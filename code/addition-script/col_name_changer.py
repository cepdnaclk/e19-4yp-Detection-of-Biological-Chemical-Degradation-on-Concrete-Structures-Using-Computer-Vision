import pandas as pd
import os
import re
import shutil

# CONFIGURATION
csv_file = 'sulphate_attack_data.csv'          # Your CSV file name
column_name = 'Image_Name'            # Column name containing the image filenames
offset = 400                         # Amount to add to the number

# Load the CSV
df = pd.read_csv(csv_file)

# Function to modify filename
def modify_filename(filename):
    match = re.search(r'(image_)(\d+)(\.\w+)', filename)
    if match:
        prefix, num, ext = match.groups()
        new_num = int(num) + offset
        return f"{prefix}{new_num}{ext}"
    return filename  # Return unchanged if pattern doesn't match

# Apply the function to the desired column
df[column_name] = df[column_name].apply(modify_filename)

# Save updated CSV
df.to_csv('updated_' + csv_file, index=False)
