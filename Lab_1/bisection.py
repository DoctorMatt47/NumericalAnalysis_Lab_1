def execute(func):
    eps = 0.00001
    a = 0.0
    b = 1.0
    c = 0
    while abs(a - b) > eps:
        c = (a + b) / 2
        if func(c) * func(a) < 0:
            b = c
        else:
            a = c
    return c
