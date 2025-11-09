# Body patterns of living things
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----- parameters -----
size = 200
Du, Dv = 0.16, 0.08  # diffusion parameter
F, k = 0.060, 0.062  # Gray-Scott parameter

# ----- initial -----
U = np.ones((size, size))
V = np.zeros((size, size))

# insert V
r = 20
U[size//2 - r:size//2 + r, size//2 - r:size//2 + r] = 0.50
V[size//2 - r:size//2 + r, size//2 - r:size//2 + r] = 0.25

# noize
U += 0.05 * np.random.random((size, size))
V += 0.05 * np.random.random((size, size))

# ----- update function -----
def laplacian(Z):
    return (
        -4*Z
        + np.roll(Z, (0, -1), (0, 1))
        + np.roll(Z, (0, 1), (0, 1))
        + np.roll(Z, (-1, 0), (0, 1))
        + np.roll(Z, (1, 0), (0, 1))
    )

def update(frame):
    global U, V
    for _ in range(10):
        Lu, Lv = laplacian(U), laplacian(V)
        UVV = U * V * V
        U += (Du * Lu - UVV + F * (1 - U)) * 1.0
        V += (Dv * Lv + UVV - (F + k) * V) * 1.0
    im.set_data(V)
    return [im]

# ----- visualize -----
fig, ax = plt.subplots()
im = ax.imshow(V, interpolation='bilinear', vmin=0, vmax=1)
ax.set_axis_off()
fig.suptitle("Gray-Scott Reaction-Diffusion")

anim = FuncAnimation(fig, update, frames=200, interval=10)
plt.show()