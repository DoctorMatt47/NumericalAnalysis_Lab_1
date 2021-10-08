#            double Function(double x) => 5 * Math.Pow(x, 3) - 2 * Math.Pow(x, 2) * Math.Sin(x) - (double) 2 / 5;
#            double DFunction(double x) => 15 * Math.Pow(x, 2) - 4 * Math.Sin(x) * x - 2 * Math.Pow(x, 2) * Math.Cos(x);
#            double D2Function(double x) => 2 * Math.Pow(x, 2) * Math.Sin(x) - 4 * Math.Sin(x) + 30 * x - 8 * x * Math.Cos(x);
import math
import bisection
import newton

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
    print()
    pass

if __name__ == '__main__':
    main()