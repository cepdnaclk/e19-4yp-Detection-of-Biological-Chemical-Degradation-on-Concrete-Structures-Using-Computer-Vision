import pandas as pd

# Define file paths
chloride_csv_path = '/content/drive/MyDrive/Research/Research dataset/Final_Dataset/choliride attack/chloride_attack_data.csv'
sulphate_csv_path = '/content/drive/MyDrive/Research/Research dataset/Final_Dataset/Sulphate attack/sulphate_attack_data_V2.csv'
combined_csv_output_path = '/content/drive/MyDrive/Research/Research dataset/Final_Dataset/combined_attack_data.csv'

# Load the CSV files
df_chloride = pd.read_csv(chloride_csv_path)
df_sulphate = pd.read_csv(sulphate_csv_path)

# Display available columns for each to verify
print("📘 Chloride CSV columns:", df_chloride.columns.tolist())
print("📙 Sulphate CSV columns:", df_sulphate.columns.tolist())

# Add attack type column
df_chloride['Attack_Type'] = 'Chloride'
df_sulphate['Attack_Type'] = 'Sulphate'

# Ensure both dataframes have the same column structure
common_columns = list(set(df_chloride.columns).intersection(set(df_sulphate.columns)))
df_chloride = df_chloride[common_columns]
df_sulphate = df_sulphate[common_columns]

# Combine both dataframes
df_combined = pd.concat([df_chloride, df_sulphate], ignore_index=True)

# Save combined CSV
df_combined.to_csv(combined_csv_output_path, index=False)

# Final outputs
print(f"✅ Combined dataset saved to: {combined_csv_output_path}")
print(f"✅ Total records combined: {len(df_combined)}")
print("📊 Sample preview of combined data:")
print(df_combined.head())
