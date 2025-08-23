"""
Test file for SimulatedAnnealingOptimizer
"""
import numpy as np
from src import create_solver


def test_sphere_function_minimization():
    """Test simulated annealing on sphere function (minimization)"""
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='SimulatedAnnealingOptimizer',
        objective_func=sphere_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=False,
        max_temperatures=50,
        equilibrium_steps=100
    )
    
    _, best = method.solver(
        max_iter=50
    )
    
    # Should find a solution close to [0, 0] with fitness near 0
    print(f"Best fitness: {best.fitness}")
    print(f"Best position: {best.position}")
    assert best.fitness < 0.5
    assert np.all(np.abs(best.position) < 1.0)


def test_rastrigin_function_minimization():
    """Test simulated annealing on Rastrigin function (minimization)"""
    def rastrigin_function(x):
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    method = create_solver(
        solver_name='SimulatedAnnealingOptimizer',
        objective_func=rastrigin_function,
        lb=-5.12,
        ub=5.12,
        dim=2,
        maximize=False,
        max_temperatures=50,
        equilibrium_steps=100
    )
    
    _, best = method.solver(
        max_iter=50
    )
    
    # Should find a solution with reasonably low fitness
    print(f"Best fitness: {best.fitness}")
    print(f"Best position: {best.position}")
    assert best.fitness < 10.0


def test_maximization():
    """Test simulated annealing on maximization problem"""
    def negative_sphere(x):
        return -np.sum(x**2)
    
    method = create_solver(
        solver_name='SimulatedAnnealingOptimizer',
        objective_func=negative_sphere,
        lb=-2.0,
        ub=2.0,
        dim=2,
        maximize=True,
        max_temperatures=50,
        equilibrium_steps=100
    )
    
    _, best = method.solver(
        max_iter=50
    )
    
    # Should find a solution with fitness near 0 (maximizing negative sphere)
    print(f"Best fitness: {best.fitness}")
    print(f"Best position: {best.position}")
    assert best.fitness > -0.5


def test_camelback_function():
    """Test simulated annealing on six-hump camelback function"""
    def camelback_function(x):
        x1, x2 = x
        term1 = (4 - 2.1*x1**2 + x1**4/3) * x1**2
        term2 = x1*x2
        term3 = (-4 + 4*x2**2) * x2**2
        return term1 + term2 + term3
    
    method = create_solver(
        solver_name='SimulatedAnnealingOptimizer',
        objective_func=camelback_function,
        lb=-3.0,
        ub=3.0,
        dim=2,
        maximize=False,
        max_temperatures=100,
        equilibrium_steps=200
    )
    
    _, best = method.solver(
        max_iter=100
    )
    
    # Should find one of the global minima around -1.0316
    print(f"Best fitness: {best.fitness}")
    print(f"Best position: {best.position}")
    assert best.fitness < -0.5  # Should be better than most local minima


if __name__ == "__main__":
    print("Testing SimulatedAnnealingOptimizer...")
    
    print("\n1. Testing sphere function minimization:")
    test_sphere_function_minimization()
    
    print("\n2. Testing Rastrigin function minimization:")
    test_rastrigin_function_minimization()
    
    print("\n3. Testing maximization problem:")
    test_maximization()
    
    print("\n4. Testing camelback function:")
    test_camelback_function()
    
    print("\nAll tests passed! ✅")
