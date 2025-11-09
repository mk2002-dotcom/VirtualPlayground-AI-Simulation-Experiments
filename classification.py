from PIL import Image
import torch
from torchvision import models, transforms

# モデル
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# 画像読み込み
# img_path = r"C:\Users\narik\code\apple.jpg"
img_path = r".\apple.jpg"
img = Image.open(img_path).convert("RGB")

# 前処理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

img_t = preprocess(img).unsqueeze(0)

# 推論
with torch.no_grad():
    out = model(img_t)
_, index = torch.max(out, 1)

# クラス名取得
labels = models.ResNet18_Weights.DEFAULT.meta["categories"]
print("予測クラス:", labels[index[0]])
