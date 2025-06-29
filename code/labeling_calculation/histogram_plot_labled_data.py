import pandas as pd
import matplotlib.pyplot as plt
import os

# Step 1: Load the CSV file
#/content/drive/MyDrive/Research/Research dataset/Final_Dataset/choliride attack/chloride_attack_data_v2.csv
#/content/drive/MyDrive/Research/Research dataset/Final_Dataset/Sulphate attack/sulphate_attack_data_V2.csv
csv_path = '/content/drive/MyDrive/Research/Research dataset/Final_Dataset/choliride attack/chloride_attack_data_V2.csv'
df = pd.read_csv(csv_path)

# Step 2: Define the save path for the plot (same directory as CSV)
save_dir = os.path.dirname(csv_path)
plot_save_path = os.path.join(save_dir, 'histogram_plots_V2.png')

# Step 3: Set the style for a presentation-friendly look
plt.style.use('ggplot')  # Use a built-in Matplotlib style for a clean look
plt.rcParams['font.family'] = 'sans-serif'  # Use sans-serif as the base font
plt.rcParams['font.sans-serif'] = ['Helvetica', 'DejaVu Sans', 'Arial', 'sans-serif']  # Fallback font list
plt.rcParams['font.size'] = 12  # Base font size

# Step 4: Create histograms for each chemical attack
fig = plt.figure(figsize=(12, 10))  # Adjusted figure size for better spacing

# Add the main title
plt.suptitle("Residual Strength of chemical attack Distribution Under Chloride Chemical Attacks",
             fontsize=16, y=1.05, weight='bold')

# HCl Strength Distribution
plt.subplot(2, 2, 1)
plt.hist(df['Residual_Strength_HCl'], bins=10, color='#87CEEB', edgecolor='black', label='HCl')
plt.title('HCl Strength Distribution', fontsize=14)
plt.xlabel('Residual Strength HCl (MPa)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# NaCl Strength Distribution
plt.subplot(2, 2, 2)
plt.hist(df['Residual_Strength_NaCl'], bins=10, color='#87CEEB', edgecolor='black', label='NaCl')
plt.title('NaCl Strength Distribution', fontsize=14)
plt.xlabel('Residual Strength NaCl (MPa)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# H2SO4 Strength Distribution
plt.subplot(2, 2, 3)
plt.hist(df['Residual_Strength_H2SO4'], bins=10, color='#87CEEB', edgecolor='black', label='H2SO4')
plt.title('H2SO4 Strength Distribution', fontsize=14)
plt.xlabel('Residual Strength H2SO4 (MPa)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# MgSO4 Strength Distribution
plt.subplot(2, 2, 4)
plt.hist(df['Residual_Strength_MgSO4'], bins=10, color='#87CEEB', edgecolor='black', label='MgSO4')
plt.title('MgSO4 Strength Distribution', fontsize=14)
plt.xlabel('Residual Strength MgSO4 (MPa)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)  # Higher DPI for better quality in presentations
plt.close()

print(f"Histogram plot saved to {plot_save_path}")