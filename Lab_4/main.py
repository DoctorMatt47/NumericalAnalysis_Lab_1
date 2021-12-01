import numpy as np
from numpy.polynomial import polynomial as p
from numpy.polynomial import Polynomial as P
import matplotlib.pyplot as plt
import math


def func(z):
    return math.e ** z + z**3


def lagrange(y, x):
    poly = [0]

    for i in range(0, y.size):
        poly_add = [y[i]]

        for j in range(0, x.size):
            if i != j:
                poly_add = p.polymul(poly_add, [-x[j], 1] / (x[i] - x[j]))

        poly = p.polyadd(poly, poly_add)

    return poly


def newton(y, x):
    diffs = np.zeros((y.size, y.size))
    diffs[:, 0] = y
    h = x[1] - x[0]

    for j in range(1, y.size):
        for i in range(0, y.size - j):
            diffs[i, j] = (diffs[i, j - 1] - diffs[i + 1, j - 1]) / (-h * j)

    # Forward formula
    poly = [0]

    for j in range(0, x.size):
        poly_add = [diffs[0, j]]

        for i in range(0, j):
            poly_add = p.polymul(poly_add, [-x[i], 1])

        poly = p.polyadd(poly, poly_add)

    return poly


def figure(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.scatter(x, y)


def main():
    a, b, n = -4, 4, 20
    x = np.linspace(a, b, n)
    y = func(x)
    real = np.linspace(a, b, 100)

    lagrange_poly = lagrange(y, x)
    print("Lagrange:", lagrange_poly)
    figure(x, y)
    plt.plot(real, p.polyval(real, lagrange_poly))
    plt.show()

    newton_poly = newton(y, x)
    print("Newton:", newton_poly)
    figure(x, y)
    plt.plot(real, p.polyval(real, newton_poly))
    plt.show()

if __name__ == '__main__':
    main()