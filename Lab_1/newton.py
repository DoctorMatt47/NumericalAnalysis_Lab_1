def execute(func, d_func):
    eps = 0.00001
    a = 0.0
    b = 1.0
    x0 = func(a) *  a if d_func(a) > 0 else b
    xn = x0 - func(x0) / d_func(x0)
    while (abs(x0 - xn) > eps):
        x0 = xn
        xn = x0 - func(x0) / d_func(x0)
    return xn