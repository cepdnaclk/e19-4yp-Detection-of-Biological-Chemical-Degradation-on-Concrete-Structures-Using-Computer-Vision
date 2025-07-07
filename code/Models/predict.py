import os
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from Models.common.utils import preprocess_image

# Define model and scaler paths
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'saved_models', 'densenet_attack_predictor.h5')
)
SCALER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'saved_models', 'target_scaler.pkl')
)

def load_regression_model(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """Load the trained regression model and its scaler."""
    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict(model, scaler, image_np_array, attack_type):
    """
    Predict the residual strengths.
    - image_np_array: a 3D image array (H, W, C) already resized and normalized
    - attack_type: integer (0 for chloride, 1 for sulphate, etc.)
    Returns a list of 4 predicted values.
    """
    image = np.expand_dims(image_np_array, axis=0)  # shape: (1, 224, 224, 3)
    attack_input = np.array([[attack_type]])        # shape: (1, 1)
    scaled_prediction = model.predict([image, attack_input])
    prediction = scaler.inverse_transform(scaled_prediction)
    return prediction[0]
