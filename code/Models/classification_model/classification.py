import os
from PIL import Image
import torch
from torchvision import transforms
import timm

# Global constants
MODEL_NAME = "efficientnetv2_s"
CLASS_NAMES = ['bio-degradation', 'chloride-attack', 'sulphate-attack']
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'best_efficientnetv2_s_seed42.pt')
)
# Load classification model once and reuse
def load_classification_model(model_path=MODEL_PATH):
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Classify a single image using the loaded model
def classify_image(model, img_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    image = transform(Image.open(img_path).convert('RGB')).unsqueeze(0)

    with torch.no_grad():
        pred = model(image).argmax().item()
        return CLASS_NAMES[pred]
