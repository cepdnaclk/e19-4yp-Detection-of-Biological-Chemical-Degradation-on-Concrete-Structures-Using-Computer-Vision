import pandas as pd
import os
import re

# CONFIGURATION
csv_file = 'combined_attack_data.csv'      # CSV file name
column_name = 'Image_Name'                 # Column with image filenames
type_column = 'Attack_Type'             # Binary column (0 or 1)
offset = 400                               # Amount to add to image number

# Load CSV
df = pd.read_csv(csv_file)

# Function to modify filename only if type_of_attack == 1
def conditional_modify(row):
    filename = row[column_name]
    attack_type = row[type_column]
    
    if attack_type == 1:
        match = re.search(r'(image_)(\d+)(\.\w+)', filename)
        if match:
            prefix, num, ext = match.groups()
            new_num = int(num) + offset
            return f"{prefix}{new_num}{ext}"
    return filename  # Unchanged if type_of_attack != 1 or pattern doesn't match

# Apply row-wise
df[column_name] = df.apply(conditional_modify, axis=1)

# Save updated CSV
df.to_csv('updated_' + csv_file, index=False)
print("Updated CSV saved as:", 'updated_' + csv_file)
