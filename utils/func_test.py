import numpy as np

def zdt1_function(x):
    """
    ZDT1 test function for multi-objective optimization
    f1 = x1
    f2 = g * (1 - sqrt(f1/g))
    where g = 1 + 9 * sum(x2...xn)/(n-1)
    """
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    f2 = g * (1 - np.sqrt(f1 / g))
    return np.array([f1, f2])

def sphere_function(x):
    return np.sum(x**2)

def rastrigin_function(x):
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def negative_sphere(x):
        return -np.sum(x**2)