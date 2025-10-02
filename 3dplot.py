# 3d plot or contour
import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# data
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = (X - Y**2)**2 + (Y - 1)**2

# plot
surf = ax.plot_surface(X, Y, Z, cmap="viridis")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

fig.savefig("3d_plot.png")