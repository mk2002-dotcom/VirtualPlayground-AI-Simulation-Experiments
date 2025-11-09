# ink diffusion
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter

# グリッド
nx, ny = 150, 150
steps = 150

# 初期インク（少し広めに）
u0 = np.zeros((nx, ny))
X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
u0 += np.exp(-((X - nx/2)**2 + (Y - ny/2)**2) / 500)

# animation
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(u0, cmap='Blues', vmin=0, vmax=1, interpolation='gaussian')
ax.axis('off')
plt.tight_layout()

def update(frame):
    # 拡散範囲を時間とともに大きく
    sigma0 = 1.0
    sigma = sigma0 + frame * 0.5

    # 拡散（＝時間発展）
    u = gaussian_filter(u0, sigma=sigma)

    # 全体のフェード（インクがゆっくり薄まる）
    u *= np.exp(-frame / 400)

    im.set_array(u)
    return [im]

ani = FuncAnimation(fig, update, frames=steps, interval=60, blit=True)
plt.show()
