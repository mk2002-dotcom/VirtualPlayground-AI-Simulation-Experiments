# least squares method
import numpy as np
import matplotlib.pyplot as plt


# mat @ coeff = vec
def ls(x, y, order):
    ndata = len(y)
    vec = np.zeros(order)
    mat = np.zeros((order, order))
    
    for i in range(order):
        for k in range(ndata):
            vec[i] += y[k] * x[k]**i
            
    for i in range(order):
        for j in range(order):
            for k in range(ndata):
                mat[i, j] += x[k]**(i + j)
    
    return np.linalg.inv(mat) @ vec
    

order = 4
x = [0, 1, 2, 3, 4, 5, 6]
y = [1, 3.3, 6.8, 14, 20.8, 31.1, 47]
coeff  = ls(x, y, order)
print(coeff)

# plot
fig, ax = plt.subplots()
x_plot = np.linspace(min(x), max(x), 100)
y_plot = np.zeros(len(x_plot))
for i in range(len(y_plot)):
    for k in range(len(coeff)):
        y_plot[i] += coeff[k] * x_plot[i]**k

ax.plot(x, y, marker = '.', linestyle='', label = 'Data')
ax.plot(x_plot, y_plot, marker = '', linestyle='-', label = 'Fit')
ax.legend()
fig.savefig('ls.png')
plt.show()