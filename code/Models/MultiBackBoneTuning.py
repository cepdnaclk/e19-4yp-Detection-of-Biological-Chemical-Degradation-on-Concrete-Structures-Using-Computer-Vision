import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.applications import (
    DenseNet121, EfficientNetB0, InceptionV3, MobileNetV2, ResNet50
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ----------- Load Data -------------
regression_cols = ['Residual_Strength_NaCl', 'Residual_Strength_HCl',
                   'Residual_Strength_H2SO4', 'Residual_Strength_MgSO4']

def load_and_clean_csv(csv_path):
    df = pd.read_csv(csv_path)
    df[regression_cols] = df[regression_cols].apply(pd.to_numeric, errors='coerce')
    return df.dropna(subset=regression_cols)

def load_data(df, image_dir, img_size=(224, 224)):
    X_img, X_attack, y = [], [], []
    for _, row in df.iterrows():
        img_path = image_dir + row['Image_Name']
        img = load_img(img_path, target_size=img_size)
        img = img_to_array(img) / 255.0
        X_img.append(img)
        X_attack.append(row['type_of_attack'])
        y.append([row[col] for col in regression_cols])
    return np.array(X_img), np.array(X_attack).reshape(-1, 1), np.array(y, dtype=np.float32)

def normalize_targets(y):
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y)
    return y_scaled, scaler

# ------------- Build Model with Hyperparams ------------
def build_model(hp):
    backbone_name = hp.Choice('backbone', ['DenseNet121', 'EfficientNetB0', 'InceptionV3', 'MobileNetV2', 'ResNet50'])
    input_shape = (224, 224, 3)

    if backbone_name == 'DenseNet121':
        base_model = DenseNet121(include_top=False, input_shape=input_shape, weights='imagenet')
    elif backbone_name == 'EfficientNetB0':
        base_model = EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')
    elif backbone_name == 'InceptionV3':
        base_model = InceptionV3(include_top=False, input_shape=input_shape, weights='imagenet')
    elif backbone_name == 'MobileNetV2':
        base_model = MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet')
    elif backbone_name == 'ResNet50':
        base_model = ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')

    base_model.trainable = False

    image_input = Input(shape=input_shape, name='image')
    attack_input = Input(shape=(1,), name='attack')

    x = base_model(image_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Concatenate()([x, attack_input])
    x = layers.Dense(hp.Int('dense_units', min_value=128, max_value=512, step=64), activation='relu')(x)
    x = layers.Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1))(x)
    output = layers.Dense(4, name='strengths')(x)

    model = Model(inputs=[image_input, attack_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('lr', [1e-3, 5e-4, 1e-4])),
        loss='mse',
        metrics=['mae']
    )
    return model

# -------------- Main Execution ------------------
df = load_and_clean_csv('data/combined_attack_data.csv')
X_img, X_attack, y = load_data(df, image_dir='data/segmented_images/')
y_scaled, scaler = normalize_targets(y)

X_img_train, X_img_test, X_attack_train, X_attack_test, y_train, y_test = train_test_split(
    X_img, X_attack, y_scaled, test_size=0.2, random_state=42
)

# Tuning
tuner = kt.Hyperband(
    build_model,
    objective='val_mae',
    max_epochs=20,
    factor=3,
    directory='tuner_logs',
    project_name='multi_backbone_regression'
)

tuner.search(
    {'image': X_img_train, 'attack': X_attack_train},
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=8,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)]
)

best_model = tuner.get_best_models(1)[0]
history = best_model.fit(
    {'image': X_img_train, 'attack': X_attack_train},
    y_train,
    validation_data=({'image': X_img_test, 'attack': X_attack_test}, y_test),
    epochs=30,
    batch_size=8
)

# Evaluate and plot
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Training vs Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

y_pred = best_model.predict({'image': X_img_test, 'attack': X_attack_test})
y_pred_denorm = scaler.inverse_transform(y_pred)
y_true_denorm = scaler.inverse_transform(y_test)

chemical_names = ['NaCl', 'HCl', 'H2SO4', 'MgSO4']
for i, name in enumerate(chemical_names):
    mae = mean_absolute_error(y_true_denorm[:, i], y_pred_denorm[:, i])
    print(f"{name} MAE: {mae:.2f}")
