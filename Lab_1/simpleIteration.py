def execute(func):
    eps = 0.00001
    x = 1.0
    lmd = 0.1
    prevX = x + 1
    while (abs(prevX - x) >= eps):
        prevX = x
        x = x - func(x) * lmd
    return x