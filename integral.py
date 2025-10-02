# Integral using Monte Carlo method
import numpy as np


def func(x):
    return x**2 - 1


iter = 100000
min_x = 0
max_x = 1

x = np.linspace(min_x, max_x, iter)
max_y = max(func(x))
min_y = min(func(x))
x_rand = np.random.uniform(min_x, max_x, size=iter)
y_rand = np.random.uniform(min_y, max_y, size=iter)

# count points between 0 and func
in_plus  = (0 < y_rand) & (y_rand < func(x_rand))
in_minus = (func(x_rand) < y_rand) & (y_rand < 0)

# bool to int
in_plus  = in_plus.astype(int)
in_minus = in_minus.astype(int)

# (max_x - min_x) * (max_y - min_y) is the total area
# The result must be -0.666...
result = (max_x - min_x) * (max_y - min_y) * np.mean(in_plus - in_minus)
print(f"Result = {result}")
