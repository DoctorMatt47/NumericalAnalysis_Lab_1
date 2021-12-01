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


def spline(y, x):
    tridiag_a = np.zeros((x.size - 2, x.size - 2))
    b = np.empty(x.size - 2)

    for i in range(0, x.size - 2):
        b[i] = (y[i + 2] - y[i + 1]) / (x[i + 2] - x[i + 1]) - (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        if i > 0:
            tridiag_a[i, i - 1] = (x[i + 1] - x[i]) / 6
        tridiag_a[i, i] = (x[i + 2] - x[i]) / 3
        if i < x.size - 3:
            tridiag_a[i, i + 1] = (x[i + 2] - x[i + 1]) / 6

    m = np.linalg.solve(tridiag_a, b)
    m = np.insert(m, 0, 0)
    m = np.insert(m, m.size, 0)

    polys = []

    for i in range(0, x.size - 1):
        poly = P([0])
        h = x[i + 1] - x[i]
        poly += P([x[i + 1], -1]) ** 3 * (m[i] / (6 * h))
        poly += P([-x[i], 1]) ** 3 * (m[i + 1] / (6 * h))
        poly += P([x[i + 1], -1]) * ((y[i] - m[i] * h * h / 6) / h)
        poly += P([-x[i], 1]) * ((y[i + 1] - m[i + 1] * h * h / 6) / h)

        polys.append(poly.coef)

    return polys


def show_spline(spline_polys, n):
    for i in range(0, n - 1):
        x_lower = -4 + (8 / (n - 1)) * i
        x_upper = -4 + (8 / (n - 1)) * (i + 1)
        x_interval = np.linspace(x_lower, x_upper, 50)
        print("{0} for [{1}; {2}]".format(spline_polys[i], x_lower, x_upper))
        plt.plot(x_interval, p.polyval(x_interval, spline_polys[i]))
    plt.show()


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
    a, b, n = -4, 4, 9
    x = np.linspace(a, b, n)
    y = func(x)
    real = np.linspace(-4, 4, 100)

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

    print("Spline:")
    spline_poly = spline(y, x)
    figure(x, y)
    show_spline(spline_poly, n)


if __name__ == '__main__':
    main()