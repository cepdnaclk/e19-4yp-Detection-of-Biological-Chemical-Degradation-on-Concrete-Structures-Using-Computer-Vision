# import os
# import numpy as np
# from tensorflow.keras.models import load_model
# import joblib
# from common.utils import preprocess_image  # Update if needed

# # Define model and scaler paths relative to this file
# MODEL_PATH = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), 'saved_models', 'densenet_attack_predictor.h5')
# )
# SCALER_PATH = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), 'saved_models', 'target_scaler.pkl')
# )


# def predict_residual_strength(img_path, attack_type):
#     # Load model and scaler
#     model = load_model(MODEL_PATH, compile=False)
#     scaler = joblib.load(SCALER_PATH)


#     # Preprocess image
#     image = preprocess_image(img_path, target_size=(224, 224))
#     image = np.expand_dims(image, axis=0)  # shape (1, 224, 224, 3)

#     # Prepare attack input
#     attack_input = np.array([[attack_type]])  # shape (1, 1)

#     # Predict
#     scaled_prediction = model.predict([image, attack_input])
#     prediction = scaler.inverse_transform(scaled_prediction)

#     return prediction[0]

# if __name__ == "__main__":
#     img_path = input("Enter image path: ").strip()
#     attack_type = int(input("Enter attack type (0 for chloride, 1 for sulphate): ").strip())

#     result = predict_residual_strength(img_path, attack_type)
#     print("Predicted Residual Strengths (NaCl, HCl, H2SO4, MgSO4):", result)

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