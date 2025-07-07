import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

# === Import segmentation modules ===
from image_segmentation.chloride_segmentation import segment_chloride_image
from image_segmentation.sulphate_segmentation import segment_sulphate_image

# === Import classification and regression modules ===
from Models.classification_model.classification import load_classification_model, classify_image
from Models.predict import load_regression_model, predict

# === Convert RGB image to model input format ===
def prepare_for_densenet(np_image):
    image = cv2.resize(np_image, (224, 224)) / 255.0
    return img_to_array(image)

# === Main Pipeline ===
def main():
    img_path = input("📂 Enter full path to image: ").strip()
    if not os.path.exists(img_path):
        print("Image path does not exist.")
        return

    #Load classification model and classify
    print("\n🔍 Classifying image...")
    clf_model = load_classification_model()
    pred_class = classify_image(clf_model, img_path)
    print(f"🧠 Predicted Class: {pred_class}")

    #If biological, exit early
    if pred_class == "bio-degradation":
        print("Biological degradation detected.")
        return

    # Segment based on class
    print("Segmenting image...")
    if pred_class == "chloride-attack":
        segmented_img = segment_chloride_image(img_path)
        attack_type = 0
    elif pred_class == "sulphate-attack":
        segmented_img = segment_sulphate_image(img_path)
        attack_type = 1
    else:
        print("Unknown degradation type for segmentation.")
        return

    if segmented_img is None:
        print("Segmentation failed.")
        return

    #Predict using regression model
    print("⚙️ Predicting residual strength...")
    image_array = prepare_for_densenet(segmented_img)
    reg_model, scaler = load_regression_model()
    residuals = predict(reg_model, scaler, image_array, attack_type)

    #Final Output
    print("\n🎯 Final Output:")
    print(f"Class: {pred_class}")
    print(f"Predicted Residual Strengths:")
    print(f"    NaCl   : {residuals[0]:.2f}")
    print(f"    HCl    : {residuals[1]:.2f}")
    print(f"    H2SO4  : {residuals[2]:.2f}")
    print(f"    MgSO4  : {residuals[3]:.2f}")

if __name__ == "__main__":
    main()