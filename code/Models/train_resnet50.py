import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from common.utils import load_and_clean_csv, load_data, normalize_targets, plot_results, plot_mae_history

df = load_and_clean_csv('data/combined_attack_data.csv')
X_img, X_attack, y = load_data(df, image_dir='data/segmented_images/')
y_scaled, scaler = normalize_targets(y)

X_img_train, X_img_test, X_attack_train, X_attack_test, y_train, y_test = train_test_split(
    X_img, X_attack, y_scaled, test_size=0.2, random_state=42
)

base_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
base_model.trainable = False

image_input = tf.keras.Input(shape=(224, 224, 3))
attack_input = tf.keras.Input(shape=(1,))

x = base_model(image_input, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Concatenate()([x, attack_input])
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(4)(x)

model = models.Model([image_input, attack_input], output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit([X_img_train, X_attack_train], y_train,
                    validation_data=([X_img_test, X_attack_test], y_test),
                    epochs=30, batch_size=8)
plot_mae_history(history, "ResNet50 (Frozen)")

base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='mse', metrics=['mae'])

fine_history = model.fit([X_img_train, X_attack_train], y_train,
                         validation_data=([X_img_test, X_attack_test], y_test),
                         epochs=20, batch_size=8)
plot_mae_history(fine_history, "ResNet50 (Fine-tuned)")

y_pred_scaled = model.predict([X_img_test, X_attack_test])
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test)

plot_results(y_true, y_pred, ['NaCl', 'HCl', 'H2SO4', 'MgSO4'])