from PIL import Image
import torch
from torchvision import transforms
import timm

img_path = "test-image-1.jpg"
model_name = "efficientnetv2_s"
class_names = ['bio-degradation','chloride-attack', 'sulphate-attack']

# Load model
model = timm.create_model(model_name, pretrained=False, num_classes=3)
model.load_state_dict(torch.load(f"../saved_models/best_{model_name}_seed42.pt"))
model.eval()

# Preprocess
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])
img = transform(Image.open(img_path).convert('RGB')).unsqueeze(0)
with torch.no_grad():
    pred = model(img).argmax().item()
    print(f"Prediction: {class_names[pred]}")