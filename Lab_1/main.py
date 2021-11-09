import math
import bisection
import newton
import simpleIteration


def func(x):
    """ 5 * x^3 - 2 * x^2 * sin(x) - 2 / 5 """
    return 5 * x ** 3 - 2 * x ** 2 * math.sin(x) - 2 / 5


def d_func(x):
    """ 15 * x^2 - 4 * x * sin(x) - 2 * x^2 * cos(x) """
    return 15 * x ** 2 - 4 * x * math.sin(x) - 2 * x ** 2 * math.cos(x)


def main():
    print()
    print(f"Bisection method: {bisection.execute(func)}")
    print(f"Newton method: {newton.execute(func, d_func)}")
    print(f"Simple iteration method: {simpleIteration.execute(func)}")


if __name__ == '__main__':
    main()
