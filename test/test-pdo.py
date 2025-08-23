import numpy as np
import pytest
from src import create_solver


def test_sphere_function_minimization():
    """Test PDO on sphere function (minimization)"""
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='PrairieDogsOptimizer',
        objective_func=sphere_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=False
    )
    
    _, best = method.solver(
        search_agents_no=50,
        max_iter=100
    )
    
    # Should find a solution close to [0, 0] with fitness near 0
    assert best.fitness < 0.1
    assert np.all(np.abs(best.position) < 0.5)


def test_rastrigin_function_minimization():
    """Test PDO on Rastrigin function (minimization)"""
    def rastrigin_function(x):
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    method = create_solver(
        solver_name='PrairieDogsOptimizer',
        objective_func=rastrigin_function,
        lb=-5.12,
        ub=5.12,
        dim=2,
        maximize=False
    )
    
    _, best = method.solver(
        search_agents_no=50,
        max_iter=100
    )
    
    # Should find a solution with reasonably low fitness
    assert best.fitness < 5.0


def test_maximization():
    """Test PDO on maximization problem"""
    def negative_sphere(x):
        return -np.sum(x**2)
    
    method = create_solver(
        solver_name='PrairieDogsOptimizer',
        objective_func=negative_sphere,
        lb=-2.0,
        ub=2.0,
        dim=2,
        maximize=True
    )
    
    _, best = method.solver(
        search_agents_no=50,
        max_iter=100
    )
    
    # Should find a solution with fitness near 0 (maximizing negative sphere)
    assert best.fitness > -0.1


def test_custom_parameters():
    """Test PDO with custom parameters"""
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='PrairieDogsOptimizer',
        objective_func=sphere_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=False,
        rho=0.01,  # Custom parameter
        eps_pd=0.2,  # Custom parameter
        eps=1e-12  # Custom parameter
    )
    
    _, best = method.solver(
        search_agents_no=30,
        max_iter=50
    )
    
    # Should still converge reasonably well
    assert best.fitness < 1.0


def test_boundary_handling():
    """Test that PDO respects boundary constraints"""
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='PrairieDogsOptimizer',
        objective_func=sphere_function,
        lb=-1.0,
        ub=1.0,
        dim=3,
        maximize=False
    )
    
    _, best = method.solver(
        search_agents_no=20,
        max_iter=30
    )
    
    # All positions should be within bounds
    assert np.all(best.position >= -1.0)
    assert np.all(best.position <= 1.0)


if __name__ == "__main__":
    # Run the tests
    test_sphere_function_minimization()
    test_rastrigin_function_minimization()
    test_maximization()
    test_custom_parameters()
    test_boundary_handling()
    print("All tests passed!")
