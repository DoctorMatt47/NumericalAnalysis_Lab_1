import numpy as np
import math


def get_hilbert_matrix(size):
    x = np.arange(1, size + 1) + np.arange(0, size)[:, np.newaxis]
    return 1.0 / x


def get_dominant_matrix(size):
    rand = np.random.randn(size, size)
    absolute = np.abs(rand)
    sums = absolute.sum(axis=1) - absolute.diagonal()
    add = np.diagflat(sums * np.sign(rand.diagonal()))
    return rand + add


def get_symmetric_matrix(size):
    b = np.random.randint(-10, 10, (size, size))
    return b + b.T


def get_matrix(size):
    return np.random.rand(size, size)


def get_vector(size):
    return np.random.rand(size)


def gauss(a, b):
    a = a.copy()
    b = b.copy()

    for i in range(0, a.shape[0]):
        max_i = i + np.argmax(a[range(i, a.shape[1]), i])
        max_el = a[max_i, i]

        # Swaps max_i and i
        if max_i != i:
            a[[max_i, i], :] = a[[i, max_i], :]
            b[[max_i, i]] = b[[i, max_i]]

        a[i, :] /= max_el
        b[i] /= max_el

        # Does zeros under i row
        for row_index in range(i + 1, a.shape[1]):
            multiplier = a[row_index, i]
            a[row_index, :] -= a[i, :] * multiplier
            b[row_index] -= b[i] * multiplier

    x = np.empty_like(b)

    # Finds x
    for i in range(0, a.shape[0]):
        row_index = x.size - 1 - i
        x[row_index] = b[row_index]
        for j in range(0, i):
            x[row_index] -= a[row_index, a.shape[1] - 1 - j] * x[x.size - 1 - j]

    return x


def jacoby(A, b, max_steps=50, eps=1e-06):
    x = np.zeros(len(A[0]))

    D = np.diag(A)
    R = A - np.diagflat(D)

    for i in range(int(max_steps)):
        x_prev = x
        x = (b - np.dot(R, x)) / D
        if np.allclose(x, x_prev, atol=eps):
            return x

    return x


def seidel(A, b, max_steps=1e+3, eps=1e-06):
    x = np.zeros(len(A[0]))

    for k in range(int(max_steps)):
        x_prev = x.copy()

        for i in range(0, A.shape[0]):
            d = b[i]

            for j in range(0, A.shape[1]):
                if i != j:
                    d -= A[i, j] * x[j]
            x[i] = d / A[i, i]

        if np.allclose(x, x_prev, atol=eps):
            return x

    return x


def jacobi_rotation(a, epsilon=1e-6):
    u_prod = np.eye(a.shape[0])

    # Quadratic sum of non-diagonal elements
    t = 0
    for i in range(a.shape[0]):
        for j in range(i, a.shape[1]):
            if i != j:
                t += 2 * (a[i, j] ** 2)

    while t > epsilon:
        # Find max abs non-diagonal element
        max_elem, max_i, max_j = 0, 0, 0
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if i != j and abs(a[i, j]) > abs(max_elem):
                    max_elem = a[i, j]
                    max_i, max_j = i, j

        if abs(a[max_i, max_i] - a[max_j, max_j]) < epsilon:
            phi = math.pi / 4
        else:
            phi = math.atan(2 * max_elem / (a[max_i, max_i] - a[max_j, max_j])) / 2

        u = np.eye(a.shape[0])
        u[max_i, max_i] = math.cos(phi)
        u[max_i, max_j] = math.sin(phi)
        u[max_j, max_i] = -math.sin(phi)
        u[max_j, max_j] = math.cos(phi)
        t -= 2 * (max_elem ** 2)

        a = u @ a @ u.transpose()
        u_prod = u_prod @ u.transpose()

    return np.diag(a), u_prod


def main():
    print()

    a = get_dominant_matrix(4)
    b = get_vector(4)
    print("a =\n", a)
    print("b = ", b)
    print()

    print("Gauss:", gauss(a, b))
    print("Seidel:", seidel(a, b))
    print("Jacoby:", jacoby(a, b))
    print("Numpy:", np.linalg.solve(a, b))
    print()

    a = get_symmetric_matrix(3)
    print("a =\n", a)
    values, vectors = jacobi_rotation(a)
    print("\nJacobi rotation:")
    print("Eigen values: ", values)
    print("Eigen vectors:\n", vectors)
    values, vectors = np.linalg.eig(a)
    print("\nNumpy:")
    print("Eigen values: ", values)
    print("Eigen vectors:\n", vectors)



if __name__ == '__main__':
    main()
