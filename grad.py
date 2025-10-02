# Gradient Descent
# Determine parameters to fit the data with a function
import numpy as np
import torch


# The function form is assumed
def func(x, a, b):
    return a * x**2 + b


def gd(x, y):
    iter = 1000
    rate = 0.001
    a = torch.tensor(-2.0, requires_grad=True)
    b = torch.tensor(-2.0, requires_grad=True)

    # use the gradient of loss
    for k in range(iter):
        loss = 0
        for i in range(len(y)):
            loss += (func(x[i], a, b) - y[i]) ** 2

        loss.backward()
        a.data -= rate * a.grad.data
        b.data -= rate * b.grad.data
        a.grad.zero_()
        b.grad.zero_()

    return a, b


# a must be 1, b must be 2
x = [0, 1, 2, 3, 4, 5]
y = [2, 2.9, 5.9, 11.1, 18, 26.8]

a, b = gd(x, y)
print(f"a = {a}")
print(f"b = {b}")
