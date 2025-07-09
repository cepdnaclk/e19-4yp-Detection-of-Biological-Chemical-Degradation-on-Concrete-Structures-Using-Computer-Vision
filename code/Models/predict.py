import os
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from Models.common.utils import preprocess_image

MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__),'saved_models', 'densenet_attack_predictor.h5')
)
SCALER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__),'saved_models', 'target_scaler.pkl')
)

def predict_residual_strength(img_path, attack_type):
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)

    image = preprocess_image(img_path, target_size=(224, 224))
    image = np.expand_dims(image, axis=0)
    attack_input = np.array([[attack_type]])

    scaled_prediction = model.predict([image, attack_input])
    prediction = scaler.inverse_transform(scaled_prediction)

    return prediction[0]