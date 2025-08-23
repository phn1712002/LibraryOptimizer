import numpy as np
import pytest
from src import create_solver


def test_sphere_function_minimization():
    """Test SFLA on sphere function (minimization)"""
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='ShuffledFrogLeapingOptimizer',
        objective_func=sphere_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=False
    )
    
    _, best = method.solver(
        search_agents_no=50,
        max_iter=50
    )
    
    # Should find a solution close to [0, 0] with fitness near 0
    assert best.fitness < 1.0
    assert np.all(np.abs(best.position) < 2.0)


def test_rastrigin_function_minimization():
    """Test SFLA on Rastrigin function (minimization)"""
    def rastrigin_function(x):
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    method = create_solver(
        solver_name='ShuffledFrogLeapingOptimizer',
        objective_func=rastrigin_function,
        lb=-5.12,
        ub=5.12,
        dim=2,
        maximize=False
    )
    
    _, best = method.solver(
        search_agents_no=50,
        max_iter=50
    )
    
    # Should find a solution with reasonably low fitness
    assert best.fitness < 10.0


def test_maximization():
    """Test SFLA on maximization problem"""
    def negative_sphere(x):
        return -np.sum(x**2)
    
    method = create_solver(
        solver_name='ShuffledFrogLeapingOptimizer',
        objective_func=negative_sphere,
        lb=-2.0,
        ub=2.0,
        dim=2,
        maximize=True
    )
    
    _, best = method.solver(
        search_agents_no=50,
        max_iter=50
    )
    
    # Should find a solution with fitness near 0 (maximizing negative sphere)
    assert best.fitness > -1.0


def test_custom_parameters():
    """Test SFLA with custom parameters"""
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='ShuffledFrogLeapingOptimizer',
        objective_func=sphere_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=False,
        n_memeplex=4,
        memeplex_size=8,
        fla_q=3,
        fla_alpha=2,
        fla_beta=3,
        fla_sigma=1.5
    )
    
    _, best = method.solver(
        search_agents_no=32,  # 4 memeplexes × 8 frogs = 32
        max_iter=30
    )
    
    # Should find a reasonable solution
    assert best.fitness < 2.0


def test_boundary_handling():
    """Test that SFLA respects variable bounds"""
    def boundary_test_function(x):
        # Penalize solutions that go out of bounds
        penalty = 0
        if np.any(x < -1.0) or np.any(x > 1.0):
            penalty = 1000 * np.sum(np.maximum(0, np.abs(x) - 1.0))
        return np.sum(x**2) + penalty
    
    method = create_solver(
        solver_name='ShuffledFrogLeapingOptimizer',
        objective_func=boundary_test_function,
        lb=-1.0,
        ub=1.0,
        dim=3,
        maximize=False
    )
    
    _, best = method.solver(
        search_agents_no=30,
        max_iter=40
    )
    
    # Solution should be within bounds
    assert np.all(best.position >= -1.0)
    assert np.all(best.position <= 1.0)
    assert best.fitness < 1.0


if __name__ == "__main__":
    # Run tests
    test_sphere_function_minimization()
    test_rastrigin_function_minimization()
    test_maximization()
    test_custom_parameters()
    test_boundary_handling()
    print("All tests passed!")
