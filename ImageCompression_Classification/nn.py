# compression with neural network
import torch
import matplotlib.pyplot as plt
from PIL import Image
from compressai.zoo import bmshj2018_factorized
from torchvision import transforms

# loading a trained model
net = bmshj2018_factorized(quality=1, pretrained=True).eval()

img = Image.open('images.jpg').convert('L')
img = img.convert('RGB')
x = transforms.ToTensor()(img).unsqueeze(0)

# compression
out = net.compress(x)
strings = out["strings"]
shape = out["shape"]

# restoration
x_hat = net.decompress(strings, shape)["x_hat"].clamp(0, 1)

Image.fromarray((x_hat.squeeze().permute(1, 2, 0).detach().numpy()*255).astype("uint8")).save("output.png")
