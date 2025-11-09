# Interference
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


nx, ny = 200, 200
x = np.linspace(-5, 5, nx)
y = np.linspace(-5, 5, ny)
X, Y = np.meshgrid(x, y)

src1 = (-2.0, 0)
src2 = (2.0, 0)

# parameters
k = 5.0       # wave number
omega = 1.0
t = 0.0

r1 = np.sqrt((X - src1[0])**2 + (Y - src1[1])**2)
r2 = np.sqrt((X - src2[0])**2 + (Y - src2[1])**2)
Z = np.sin(k*r1) + np.sin(k*r2)

# Animation
fig, ax = plt.subplots(figsize=(6,6))
ax.set_axis_off()
img = ax.imshow(Z, extent=[-5,5,-5,5], cmap='Blues', vmin=-2, vmax=2)

def animate(frame):
    t = frame * 0.3
    r1 = np.sqrt((X - src1[0])**2 + (Y - src1[1])**2)
    r2 = np.sqrt((X - src2[0])**2 + (Y - src2[1])**2)
    Z = np.sin(k*r1 - omega*t) + np.sin(k*r2 - omega*t)
    img.set_data(Z)
    return [img]

anim = animation.FuncAnimation(fig, animate, frames=100, interval=50, blit=True)
plt.show()