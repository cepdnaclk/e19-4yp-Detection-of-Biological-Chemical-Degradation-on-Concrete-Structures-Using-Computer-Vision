import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

regression_cols = ['Residual_Strength_NaCl', 'Residual_Strength_HCl',
                   'Residual_Strength_H2SO4', 'Residual_Strength_MgSO4']

def load_and_clean_csv(csv_path):
    df = pd.read_csv(csv_path)
    df[regression_cols] = df[regression_cols].apply(pd.to_numeric, errors='coerce')
    return df.dropna(subset=regression_cols)

def load_data(df, image_dir, img_size=(224, 224)):
    X_img, X_attack, y = [], [], []
    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, row['Image_Name'])
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=img_size)
            img = img_to_array(img) / 255.0
            X_img.append(img)
            X_attack.append(row['Attack_Type'])
            y.append([row[col] for col in regression_cols])
    return np.array(X_img), np.array(X_attack).reshape(-1, 1), np.array(y, dtype=np.float32)

def normalize_targets(y):
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y)
    return y_scaled, scaler

def plot_results(y_true, y_pred, chemical_names=None):
    if chemical_names is None:
        chemical_names = regression_cols

    # Combined plot
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    combined_mae = mean_absolute_error(y_true_flat, y_pred_flat)
    print(f'Combined MAE: {combined_mae:.2f}')

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.5)
    plt.plot([y_true_flat.min(), y_true_flat.max()],
             [y_true_flat.min(), y_true_flat.max()], 'r--')
    plt.title('Combined - Predicted vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Individual plots
    for i, name in enumerate(chemical_names):
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
        plt.plot([y_true[:, i].min(), y_true[:, i].max()],
                 [y_true[:, i].min(), y_true[:, i].max()], 'r--')
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        print(f'{name} MAE: {mae:.2f}')
        plt.title(f'{name} - Predicted vs Actual')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_mae_history(history, model_name="Model"):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title(f'{model_name} MAE per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def preprocess_image(img_path, target_size=(224, 224)):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    img = load_img(img_path, target_size=target_size)
    img = img_to_array(img) / 255.0
    return img