import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def build_model(input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Transfer learning

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Regression head
    strength_output = layers.Dense(4, activation='linear', name='strengths')(x)

    model = models.Model(inputs=base_model.input, outputs=[strength_output])

    model.compile(
        optimizer='adam',
        loss={'strengths': 'mse'},
        metrics={'strengths': 'mae'}
    )
    
    return model

# Load and preprocess data
df = pd.read_csv('merged.csv')

# Convert regression columns to numeric and drop rows with invalid data
for col in ['Residual_Strength_NaCl', 'Residual_Strength_HCl', 'Residual_Strength_H2SO4', 'Residual_Strength_MgSO4']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['Residual_Strength_NaCl', 'Residual_Strength_HCl', 'Residual_Strength_H2SO4', 'Residual_Strength_MgSO4'])

# def load_data(df, image_dir='images/', img_size=(224, 224)): #used for actual image analysis
def load_data(df, image_dir='segmented_images/', img_size=(224, 224)):
    X, y_reg = [], []
    for _, row in df.iterrows():
        img = load_img(image_dir + row['Image_Name'], target_size=img_size)
        img = img_to_array(img) / 255.0
        X.append(img)
        strengths = [
            row['Residual_Strength_NaCl'],
            row['Residual_Strength_HCl'],
            row['Residual_Strength_H2SO4'],
            row['Residual_Strength_MgSO4']
        ]
        y_reg.append(strengths)
    return np.array(X), np.array(y_reg, dtype=np.float32)

X, y_reg = load_data(df)

# Normalize y_reg
scaler = MinMaxScaler()
y_reg_normalized = scaler.fit_transform(y_reg)

print('Labels dtype:', y_reg.dtype)  # Should print float32
print('Sample labels:', y_reg[:5])

X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg_normalized, test_size=0.2, random_state=42
)

model = build_model()

history = model.fit(
    X_train,
    y_reg_train,
    validation_data=(X_test, y_reg_test),
    epochs=100,
    batch_size=8
)


# # Plot training & validation MAE
# plt.plot(history.history['mae'], label='Train MAE')
# plt.plot(history.history['val_mae'], label='Val MAE')
# plt.title('Residual Strength MAE')
# plt.legend()
# plt.show()

#Predict and denormalize (add this after training and plotting)
y_pred_normalized = model.predict(X_test)

# Denormalize both predicted and true values
y_pred = scaler.inverse_transform(y_pred_normalized)
y_true = scaler.inverse_transform(y_reg_test)

#Calculate MAE for each chemical
from sklearn.metrics import mean_absolute_error

chemical_names = ['NaCl', 'HCl', 'H2SO4', 'MgSO4']
for i, name in enumerate(chemical_names):
    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
    print(f'{name} MAE: {mae:.2f}')

# for i, name in enumerate(chemical_names):
#     plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.6)
#     plt.plot([y_true[:, i].min(), y_true[:, i].max()],
#              [y_true[:, i].min(), y_true[:, i].max()], 'r--')
#     plt.title(f'{name} - Predicted vs Actual')
#     plt.xlabel('Actual')
#     plt.ylabel('Predicted')
#     plt.show()