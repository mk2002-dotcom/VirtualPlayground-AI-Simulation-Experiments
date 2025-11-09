# image classification
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1️モデル準備
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# 最後の畳み込み層の名前
target_layer = model.layer4[1].conv2

# 2️入力画像
img_path = r"C:\Users\narik\code\sample.png"
img = Image.open(img_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0)

# 3️hookで特徴と勾配を取得
features = []
gradients = []

def forward_hook(module, input, output):
    features.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# 4️順伝播と逆伝播
output = model(input_tensor)
pred_class = output.argmax().item()

model.zero_grad()
output[0, pred_class].backward()

# 5️Grad-CAM計算
grads = gradients[0].mean(dim=(2, 3), keepdim=True)
cam = torch.sum(grads * features[0], dim=1).squeeze()
cam = torch.relu(cam)
cam = cam - cam.min()
cam = cam / cam.max()

# 7x7 → 224x224 にアップサンプリング
cam = torch.nn.functional.interpolate(
    cam.unsqueeze(0).unsqueeze(0), 
    size=(224, 224), 
    mode='bilinear', 
    align_corners=False
)

# 余分な次元を削除
cam = cam.squeeze().detach().numpy()  # shape = (224, 224)

# ヒートマップ作成
img_np = np.array(img.resize((224, 224))) / 255.0
heatmap = plt.cm.jet(cam)[..., :3]  # shape = (224, 224, 3)

# 合成して表示
result = 0.4 * heatmap + 0.6 * img_np
plt.imshow(result)
plt.axis("off")
plt.show()


