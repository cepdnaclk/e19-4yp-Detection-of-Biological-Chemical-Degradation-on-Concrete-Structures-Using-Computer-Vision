import os
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np

from image_segmentation.chloride_segmentation import segment_chloride_image, save_segmented_image as save_chloride_image
from image_segmentation.sulphate_segmentation import segment_sulphate_image, save_segmented_image as save_sulphate_image
from Models.classification_model.classify import load_classification_model, classify_image
from Models.predict import predict_residual_strength


class ConcreteDegradationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Concrete Degradation Detection")
        self.root.geometry("1080x880")
        self.root.configure(bg="#e9f1f7")

        self.model = load_classification_model()
        self.selected_image_path = None

        self.setup_base_ui()

    def setup_base_ui(self):
        # Title
        tk.Label(self.root, text="Concrete Degradation Detection",
                 font=("Segoe UI", 24, "bold"), bg="#e9f1f7", fg="#2c3e50").pack(pady=20)

        # Upload Button
        ttk.Button(self.root, text="Select Image", command=self.choose_image).pack(pady=10)

        # Loading Label
        self.loading_label = tk.Label(self.root, text="", font=("Segoe UI", 12), fg="#2980b9", bg="#e9f1f7")
        self.loading_label.pack()

        # Main result frame (re-created on each use)
        self.result_frame = None

    def reset_ui(self):
        if self.result_frame:
            self.result_frame.destroy()

        self.result_frame = tk.Frame(self.root, bg="#ffffff", padx=30, pady=30, relief=tk.RIDGE, borderwidth=2)
        self.result_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

    def choose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.selected_image_path = file_path
            self.reset_ui()
            self.show_loading("Processing... Please wait")
            self.display_selected_image(file_path)
            threading.Thread(target=self.process_image).start()

    def show_loading(self, msg):
        self.loading_label.config(text=msg)

    def hide_loading(self):
        self.loading_label.config(text="")

    def display_selected_image(self, path):
        self.original_image = Image.open(path)
        self.original_image.thumbnail((320, 320))
        self.tk_image = ImageTk.PhotoImage(self.original_image)

    def display_segmented_image(self, path):
        segmented_image = Image.open(path)
        segmented_image.thumbnail((320, 320))
        return ImageTk.PhotoImage(segmented_image)

    def process_image(self):
        path = self.selected_image_path
        predicted_class = classify_image(self.model, path)
        print(f"[DEBUG] Predicted class: {predicted_class}")
        self.root.after(0, self.build_dynamic_ui, predicted_class)

    def build_dynamic_ui(self, predicted_class):
        self.reset_ui()

        # Classification label
        tk.Label(self.result_frame, text=f"Predicted Class: {predicted_class.title()}",
                 font=("Segoe UI", 16, "bold"), bg="#ffffff", fg="#34495e").pack(pady=10)

        # Image area frame
        image_frame = tk.Frame(self.result_frame, bg="#ffffff")
        image_frame.pack(pady=10)

        # Original image
        tk.Label(image_frame, text="Original Image", font=("Segoe UI", 12, "bold"),
                 bg="#ffffff").grid(row=0, column=0, pady=5)
        tk.Label(image_frame, image=self.tk_image, bg="#ffffff", relief=tk.SOLID, borderwidth=1)\
            .grid(row=1, column=0, padx=20)

        # If bio-degradation, no segmentation or residual strengths
        if predicted_class == 'bio-degradation':
            self.hide_loading()
            return

        # Segmentation and residual strength
        if predicted_class == 'chloride-attack':
            segmented = segment_chloride_image(self.selected_image_path)
            save_path = os.path.join("segmented_outputs", "chloride_segmented.png")
            save_chloride_image(segmented, save_path)
            attack_type = 0
        elif predicted_class == 'sulphate-attack':
            segmented = segment_sulphate_image(self.selected_image_path)
            save_path = os.path.join("segmented_outputs", "sulphate_segmented.png")
            save_sulphate_image(segmented, save_path)
            attack_type = 1
        else:
            self.hide_loading()
            return

        os.makedirs("segmented_outputs", exist_ok=True)

        # Segmented image display
        segmented_img = self.display_segmented_image(save_path)
        tk.Label(image_frame, text="Segmented Image", font=("Segoe UI", 12, "bold"),
                 bg="#ffffff").grid(row=0, column=1, pady=5)
        tk.Label(image_frame, image=segmented_img, bg="#ffffff", relief=tk.SOLID, borderwidth=1)\
            .grid(row=1, column=1, padx=20)
        self.segmented_img_ref = segmented_img  # prevent garbage collection

        # Predict residual strength values
        residuals = predict_residual_strength(save_path, attack_type)
        print(f"[DEBUG] Residual strength values: {residuals}")

        self.display_strengths(residuals)

        self.hide_loading()

    def display_strengths(self, values):
        # Accept list, tuple, or numpy ndarray with length 4
        if not (isinstance(values, (list, tuple, np.ndarray)) and len(values) == 4):
            print("[ERROR] Residual strength prediction returned invalid data format.")
            print(f"Received type: {type(values)}, length: {len(values) if hasattr(values, '__len__') else 'N/A'}")
            return

        # Convert numpy array to list for safe iteration
        if isinstance(values, np.ndarray):
            values = values.tolist()

        chem_names = ["NaCl", "HCl", "H₂SO₄", "MgSO₄"]
        chem_colors = ["#3498db", "#e67e22", "#9b59b6", "#2ecc71"]

        tk.Label(self.result_frame, text="Residual Strengths (MPa)",
                 font=("Segoe UI", 16, "bold"), bg="#ffffff", fg="#2c3e50").pack(pady=(25, 10))

        strength_frame = tk.Frame(self.result_frame, bg="#ffffff")
        strength_frame.pack()

        # Create side-by-side chemical strength displays
        for i, (chem, val, color) in enumerate(zip(chem_names, values, chem_colors)):
            chem_frame = tk.Frame(strength_frame, bg="#ffffff", padx=25, pady=15, relief=tk.GROOVE, borderwidth=1)
            chem_frame.grid(row=0, column=i, padx=15)

            tk.Label(chem_frame, text=chem, font=("Segoe UI", 14, "bold"),
                     fg=color, bg="#ffffff").pack()
            tk.Label(chem_frame, text=f"{val:.2f} MPa", font=("Segoe UI", 16, "bold"),
                     fg="#2d3436", bg="#ffffff").pack()


if __name__ == "__main__":
    root = tk.Tk()
    app = ConcreteDegradationApp(root)
    root.mainloop()