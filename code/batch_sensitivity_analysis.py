import os
import csv
import numpy as np
from Models.predict import predict_residual_strength
from tqdm import tqdm

def extract_image_pairs(folder_path):
    """
    Extract real and variant image pairs from the folder.
    Assumes filenames like image_1-real.jpg and image_1-var.jpg
    """
    images = os.listdir(folder_path)
    real_images = [img for img in images if '-real' in img]
    pairs = []

    for real_img in real_images:
        base_name = real_img.replace('-real', '').split('.')[0]
        real_path = os.path.join(folder_path, real_img)

        var_img = real_img.replace('-real', '-var')
        var_path = os.path.join(folder_path, var_img)

        if os.path.exists(var_path):
            pairs.append((base_name, real_path, var_path))
    return pairs

def main(folder_path):
    output_csv = os.path.join(folder_path, "comparison_results.csv")
    chemicals = ['NaCl', 'HCl', 'H2SO4', 'MgSO4']

    all_diff = []
    all_percent = []

    # Per-chemical tracking
    per_chemical_diffs = {chem: [] for chem in chemicals}
    per_chemical_percents = {chem: [] for chem in chemicals}

    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header
        header = ['Image']
        for chem in chemicals:
            header += [f'{chem} Real', f'{chem} Var']
        header += ['Avg Diff', 'Percentage Deviation (%)']
        writer.writerow(header)

        for name, real_img, var_img in tqdm(extract_image_pairs(folder_path), desc="Processing pairs"):
            real_preds = predict_residual_strength(real_img, attack_type=0)
            var_preds = predict_residual_strength(var_img, attack_type=0)

            real_preds = np.array(real_preds)
            var_preds = np.array(var_preds)

            diffs = np.abs(real_preds - var_preds)
            avg_diff = np.mean(diffs)
            percent_deviation = (avg_diff / np.mean(real_preds)) * 100 if np.mean(real_preds) != 0 else 0

            all_diff.append(avg_diff)
            all_percent.append(percent_deviation)

            for i, chem in enumerate(chemicals):
                per_chemical_diffs[chem].append(diffs[i])
                chem_percent = (diffs[i] / real_preds[i]) * 100 if real_preds[i] != 0 else 0
                per_chemical_percents[chem].append(chem_percent)

            row = [name]
            for r, v in zip(real_preds, var_preds):
                row += [r, v]
            row += [avg_diff, percent_deviation]
            writer.writerow(row)

    # Final output to console
    print("\n📊 Final Summary")
    print(f"➡️ Average Absolute Difference: {np.mean(all_diff)}")
    print(f"➡️ Average Percentage Deviation: {np.mean(all_percent)}%")

    print("\n📊 Per-Chemical Averages:")
    for chem in chemicals:
        avg_chem_diff = np.mean(per_chemical_diffs[chem])
        avg_chem_percent = np.mean(per_chemical_percents[chem])
        print(f"   {chem}: Avg Abs Diff = {avg_chem_diff}, Avg % Deviation = {avg_chem_percent}%")

    print(f"\n✅ CSV saved to: {output_csv}")

if __name__ == "__main__":
    folder = input("📂 Enter folder path containing image pairs: ").strip()
    if os.path.isdir(folder):
        main(folder)
    else:
        print("❌ Invalid folder path.")