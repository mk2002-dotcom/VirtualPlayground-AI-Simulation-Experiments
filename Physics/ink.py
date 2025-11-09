# ink diffusion
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter

nx, ny = 150, 150
steps = 150

u0 = np.zeros((nx, ny))
X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
u0 += np.exp(-((X - nx/2)**2 + (Y - ny/2)**2) / 500)

# Animation
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(u0, cmap='Blues', vmin=0, vmax=1, interpolation='gaussian')
ax.axis('off')
plt.tight_layout()

def update(frame):
    sigma0 = 1.0
    sigma = sigma0 + frame * 0.5
    u = gaussian_filter(u0, sigma=sigma)
    u *= np.exp(-frame / 400)
    im.set_array(u)
    return [im]

ani = FuncAnimation(fig, update, frames=steps, interval=60, blit=True)
plt.show()
