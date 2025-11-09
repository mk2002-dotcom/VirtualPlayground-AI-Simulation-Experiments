# Contour plot
import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots()

# data
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = (X - Y**2)**2 + (Y - 1)**2

# plot
contours = ax.contour(X, Y, Z, levels=10)
ax.set_title("Contour Plot")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.show()

# save
fig.savefig("contour.png")