# import os
# from PIL import Image
# import torch
# from torchvision import transforms
# import timm

# img_path = "test-image-2.jpg"
# model_name = "efficientnetv2_s"
# class_names = ['bio-degradation','chloride-attack', 'sulphate-attack']

# # Build the model path relative to this file
# model_path = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), '..', 'saved_models', f"best_{model_name}_seed42.pt")
# )

# # Load model
# model = timm.create_model(model_name, pretrained=False, num_classes=3)
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# model.eval()

# # Preprocess
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
# ])
# img = transform(Image.open(img_path).convert('RGB')).unsqueeze(0)
# with torch.no_grad():
#     pred = model(img).argmax().item()
#     print(f"Prediction: {class_names[pred]}")

import os
from PIL import Image
import torch
from torchvision import transforms
import timm

CLASS_NAMES = ['bio-degradation', 'chloride-attack', 'sulphate-attack']

def load_classification_model(model_name="efficientnetv2_s"):
    model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'saved_models', f"best_{model_name}_seed42.pt")
    )
    model = timm.create_model(model_name, pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def classify_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img = transform(Image.open(image_path).convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        pred = model(img).argmax().item()
        return CLASS_NAMES[pred]