import numpy as np

def zdt1_function(x):
    """
    ZDT1 test function for multi-objective optimization
    f1 = x1
    f2 = g * (1 - sqrt(f1/g))
    where g = 1 + 9 * sum(x2...xn)/(n-1)
    
    Args:
        x (array-like): Input vector of decision variables
        
    Returns:
        numpy.ndarray: Array containing two objective function values [f1, f2]
    """
    n = len(x)
    f1 = x[0]  # First objective function: simply the first variable
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)  # Constraint function g calculation
    f2 = g * (1 - np.sqrt(f1 / g))  # Second objective function
    return np.array([f1, f2])

def sphere_function(x):
    """
    Sphere function - a simple unimodal test function for optimization
    f(x) = sum(xi^2) for all i
    
    Args:
        x (array-like): Input vector of decision variables
        
    Returns:
        float: Sum of squares of all variables (single objective value)
    """
    return np.sum(x**2)

def rastrigin_function(x):
    """
    Rastrigin function - a non-convex multimodal test function for optimization
    f(x) = A*n + sum(xi^2 - A*cos(2*pi*xi)) for all i
    where A is typically 10
    
    Args:
        x (array-like): Input vector of decision variables
        
    Returns:
        float: Rastrigin function value (single objective value)
    """
    A = 10  # Amplitude of the cosine component
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def negative_sphere(x):
    """
    Negative Sphere function - the negative of the sphere function
    f(x) = -sum(xi^2) for all i
    Used for maximization problems (by converting to minimization)
    
    Args:
        x (array-like): Input vector of decision variables
        
    Returns:
        float: Negative sum of squares of all variables (single objective value)
    """
    return -np.sum(x**2)