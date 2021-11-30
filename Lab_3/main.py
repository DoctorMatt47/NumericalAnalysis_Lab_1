import numpy as np
import math


def simple_system(x):
    return np.array([x[0]**2 / x[1]**2 - math.cos(x[1]) - 2, x[0]**2 + x[1]**2 - 6])


def system(x):
    result = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        result[i] = 0
        for j in range(x.shape[0]):
            if j == i:
                result[i] += x[j] ** 3
            else:
                result[i] += x[j] ** 2

        for j in range(x.shape[0]):
            if j == i:
                result[i] -= (j + 1) ** 3
            else:
                result[i] += (j + 1) ** 2

    return result


def simple_jacoby(x):
    return np.array([[(2 * x[0]) / x[1]**2, math.sin(x[1]) - 2 * x[0]**2 / x[1]**3],
                     [2 * x[0], 2 * x[1]]])

def jacoby(x):
    result = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            result[i, j] = 0
            if j == i:
                result[i, j] += 3 * x[j] ** 2
            else:
                result[i, j] += 2 * x[j]


    return result


def newtone(system, jacoby, x, max_steps=1e3, eps=1e-6):
    system_x = system(x)
    counter = 0
    while abs(np.linalg.norm(system_x, ord=2)) > eps and counter < max_steps:
        x -= np.linalg.solve(jacoby(x), system_x)
        system_x = system(x)
        counter += 1

    return x


def newtone_modified(system, jacoby, x, max_steps=1e3, eps=1e-6):
    system_x = system(x)
    counter = 0
    while abs(np.linalg.norm(system_x, ord=2)) > eps and counter < max_steps:
        jacoby_inv = np.linalg.inv(jacoby(x))
        x -= jacoby_inv @ system_x
        system_x = system(x)
        counter += 1

    return x


def relaxation(system, x, tau=1e-3, max_steps=1e3, eps=1e-6):
    system_x = system(x)
    counter = 0
    while abs(np.linalg.norm(system_x, ord=2)) > eps and counter < max_steps:
        x -= tau * system_x
        system_x = system(x)
        counter += 1

    return x

def main():
    x = relaxation(simple_system, np.array([1., 1.]), tau=0.1)
    print("Relaxation:")
    print("x: ", x)
    print("System(x): ", simple_system(x))

    x = newtone(simple_system, simple_jacoby, np.array([1., 1.]))
    print("\nNewtone:")
    print("x: ", x)
    print("System(x): ", simple_system(x))

    x = newtone_modified(simple_system, simple_jacoby, np.array([1., 1.]))
    print("\nNewtone modified:")
    print("x: ", x)
    print("System(x): ", simple_system(x))

    x_array = np.array([1., 1., 1., 1.])
    x = relaxation(system, x_array)
    print("Relaxation:")
    print("x: ", x)
    print("System(x): ", system(x))

    x = newtone(system, jacoby, x_array)
    print("\nNewtone:")
    print("x: ", x)
    print("System(x): ", system(x))

    x = newtone_modified(system, jacoby, x_array)
    print("\nNewtone modified:")
    print("x: ", x)
    print("System(x): ", system(x))


if __name__ == '__main__':
    main()