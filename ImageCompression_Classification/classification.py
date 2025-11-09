# Image classification
from PIL import Image
import torch
from torchvision import models, transforms

# model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

img_path = r".\apple.jpg"
img = Image.open(img_path).convert("RGB")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

img_t = preprocess(img).unsqueeze(0)

with torch.no_grad():
    out = model(img_t)
_, index = torch.max(out, 1)

labels = models.ResNet18_Weights.DEFAULT.meta["categories"]
print("Class:", labels[index[0]])
