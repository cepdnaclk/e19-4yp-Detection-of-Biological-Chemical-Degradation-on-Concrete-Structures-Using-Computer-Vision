import pandas as pd

# Load both CSVs
df1 = pd.read_csv('chloride_attack_data.csv')
df2 = pd.read_csv('updated_sulphate_attack_data.csv')

# Concatenate the dataframes
merged_df = pd.concat([df1, df2], ignore_index=True)

# Save the result
merged_df.to_csv('merged.csv', index=False)
