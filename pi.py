# Calculate pi using Monte Carlo method
import numpy as np

iter = 2000000
x = np.random.uniform(-1, 1, size=iter)
y = np.random.uniform(-1, 1, size=iter)

num_in = x**2 + y**2 < 1
result = 4 * np.mean(num_in)
print(f'pi = {result}')
