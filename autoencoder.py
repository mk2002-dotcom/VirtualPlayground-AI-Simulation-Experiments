import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# --- 1. 画像読み込み。学習してないので使えない！
img = Image.open("images.jpg").convert("L")
transform = T.Compose([T.Resize((64,64)), T.ToTensor()])
x = transform(img).unsqueeze(0)  # shape: (1,1,64,64)

class GrayAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,4,2,1), nn.ReLU(),
            nn.Conv2d(16,32,4,2,1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32,16,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(16,1,4,2,1), nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)   # 圧縮表現
        out = self.decoder(z) # 復元
        return out, z

net = GrayAutoencoder()

# --- 3. 推論
with torch.no_grad():
    out, z = net(x)

print("圧縮後の潜在ベクトルサイズ:", z.shape)

# --- 4. 保存 ---
T.ToPILImage()(out.squeeze()).save("reconstructed.jpg")