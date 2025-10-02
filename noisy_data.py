# Make noisy data
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return np.sin(x)

std = 0.1
ndata = 100
x = np.linspace(0, 10, ndata)
y = func(x) + np.random.randn(ndata) * std

# save
np.savetxt('noisy_data.txt', np.column_stack((x, y)))

# plot
plt.figure()
plt.scatter(x, y, label="Noisy data", s=20)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()