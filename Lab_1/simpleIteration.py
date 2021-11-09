def execute(func):
    eps = 0.00001
    x = 1.0
    lmd = 0.1
    prev_x = x + 1
    while abs(prev_x - x) >= eps:
        prev_x = x
        x = x - func(x) * lmd
    return x
