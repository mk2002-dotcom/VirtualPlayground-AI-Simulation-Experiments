# The simplest minimum search methods
# Determine parameters to fit the data with a function
import numpy as np


# The function form is assumed
def func(x, a, b):
    return a * x**2 + b


# The Simplest minimum search
def min_search(x, y):
    a_list = np.linspace(-2, 2, 100)
    b_list = np.linspace(-2, 2, 100)

    loss_min = 10000
    for i in range(len(a_list)):
        a_test = a_list[i]

        for j in range(len(b_list)):
            b_test = b_list[j]
            loss = 0

            for k in range(len(x)):
                loss += (func(x[k], a_test, b_test) - y[k]) ** 2
            if loss < loss_min:
                a = a_test
                b = b_test
                loss_min = loss

    if loss_min == 10000:
        print("Warning")

    return a, b


# Gradient descent
def gd(x, y):
    iter = 100000
    rate = 0.001
    a, b = -2, -2

    # loss_a and loss_b are the partial derivatives of loss
    for i in range(iter):
        loss_a, loss_b = 0, 0
        for k in range(len(x)):
            loss_a += (func(x[k], a, b) - y[k]) * x[k] ** 2
            loss_b += func(x[k], a, b) - y[k]
        a -= rate * loss_a
        b -= rate * loss_b

    return a, b


x = [0, 1, 2, 3, 4, 5]
y = [2, 2.9, 5.9, 11.1, 18, 26.8]

a, b = min_search(x, y)
print(f"a = {a}")
print(f"b = {b}")
