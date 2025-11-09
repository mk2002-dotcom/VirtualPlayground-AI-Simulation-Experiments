# 2D plot
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x**2)

fig, ax = plt.subplots()

# plot
ax.plot(x, y, label='sin(x)', color='blue')
ax.set_title('y = sin(x)')
ax.set_xlabel('x')
ax.set_ylabel('sin(x)')
ax.grid(True)
ax.legend()

plt.show()

