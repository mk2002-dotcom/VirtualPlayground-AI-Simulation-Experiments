import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorly as tl
from tensorly.decomposition import tensor_train
from tensorly.tt_tensor import tt_to_tensor

img = Image.open('images.jpg').convert('L').resize((128, 128))
data = np.array(img, dtype=np.float32) / 255.0  # shape = (128,128)

# Tensor Train
r = 10
ranks = [1] + [r] * (data.ndim - 1) + [1]
print("data.ndim =", data.ndim, "=> ranks =", ranks)
tt_tensor = tensor_train(data, rank=ranks)

core_list = getattr(tt_tensor, 'core_list', None) or getattr(tt_tensor, 'factors', None)
compressed_size = sum(np.prod(core.shape) for core in core_list)
original_size = np.prod(data.shape)
print(f"圧縮率: {compressed_size/original_size:.3f}")

reconstructed = tt_to_tensor(tt_tensor)

# plot
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(data, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Reconstructed")
plt.imshow(reconstructed, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()