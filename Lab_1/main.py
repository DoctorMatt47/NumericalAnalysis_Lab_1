import math


def relaxation(f):
    eps = 1e-3
    x = 1.0
    lmd = 0.1
    prev_x = x + 1
    count = 0
    while abs(prev_x - x) >= eps:
        prev_x = x
        x -= f(x) * lmd
        count += 1
    print("Relaxation method count:", count)
    return x


def bisection(f):
    eps = 1e-3
    a = 0.0
    b = 1.0
    c = 0
    count = 0
    while abs(a - b) > eps:
        c = (a + b) / 2
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
        count += 1
    print("Bisection method count:", count)
    return c


def newtone(f, d_f):
    eps = 1e-3
    a = 0.0
    b = 1.0
    x0 = f(a) * a if d_f(a) > 0 else b
    xn = x0 - f(x0) / d_f(x0)
    count = 0
    while abs(x0 - xn) > eps:
        x0 = xn
        xn -= f(x0) / d_f(x0)
        count += 1
    print("Newton method count:", count)
    return xn

def func(x):
    """ 5 * x^3 - 2 * x^2 * sin(x) - 2 / 5 """
    return 5 * x ** 3 - 2 * x ** 2 * math.sin(x) - 2 / 5


def d_func(x):
    """ 15 * x^2 - 4 * x * sin(x) - 2 * x^2 * cos(x) """
    return 15 * x ** 2 - 4 * x * math.sin(x) - 2 * x ** 2 * math.cos(x)


def main():
    print()
    print(f"Simple iteration method: {relaxation(func)}")
    print(f"Bisection method: {bisection(func)}")
    print(f"Newton method: {newtone(func, d_func)}")


if __name__ == '__main__':
    main()
