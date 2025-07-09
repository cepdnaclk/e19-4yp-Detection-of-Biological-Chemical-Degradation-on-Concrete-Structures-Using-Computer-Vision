import os
from tkinter import Tk, filedialog

# Image segmentation
from image_segmentation.chloride_segmentation import segment_chloride_image, save_segmented_image as save_chloride_image
from image_segmentation.sulphate_segmentation import segment_sulphate_image, save_segmented_image as save_sulphate_image

# Classification
from Models.classification_model.classify import load_classification_model, classify_image

# Prediction
from Models.predict import predict_residual_strength


def choose_image():
    """Open file dialog to choose an image."""
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image", filetypes=[("Image files", "*.jpg *.png *.jpeg")]
    )
    return file_path


def main():
    print("===== Concrete Degradation Detection Pipeline =====")
    image_path = choose_image()

    if not image_path:
        print("No image selected. Exiting.")
        return

    print(f"Selected image: {image_path}")

    # Load classifier
    print("Loading classification model...")
    model = load_classification_model()

    # Predict class
    predicted_class = classify_image(model, image_path)
    print(f"Predicted class: {predicted_class}")

    if predicted_class == 'bio-degradation':
        print("Final Output: Biological degradation detected.")
        return

    # Create output directory
    output_dir = os.path.join("segmented_outputs")
    os.makedirs(output_dir, exist_ok=True)

    if predicted_class == 'chloride-attack':
        segmented = segment_chloride_image(image_path)
        if segmented is None:
            print("Segmentation failed.")
            return

        segmented_path = os.path.join(output_dir, "chloride_segmented.png")
        save_chloride_image(segmented, segmented_path)
        print(f"Chloride segmentation saved at: {segmented_path}")

        print("Predicting residual strength...")
        result = predict_residual_strength(segmented_path, attack_type=0)
        print("Predicted Residual Strengths (NaCl, HCl, H2SO4, MgSO4):", result)

    elif predicted_class == 'sulphate-attack':
        segmented = segment_sulphate_image(image_path)
        if segmented is None:
            print("Segmentation failed.")
            return

        segmented_path = os.path.join(output_dir, "sulphate_segmented.png")
        save_sulphate_image(segmented, segmented_path)
        print(f"Sulphate segmentation saved at: {segmented_path}")

        print("Predicting residual strength...")
        result = predict_residual_strength(segmented_path, attack_type=1)
        print("Predicted Residual Strengths (NaCl, HCl, H2SO4, MgSO4):", result)


if __name__ == "__main__":
    main()