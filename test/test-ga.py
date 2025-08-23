"""
Test file for GeneticAlgorithmOptimizer.
"""
import numpy as np
import pytest
from src import create_solver


def test_sphere_function_minimization():
    """Test GeneticAlgorithmOptimizer on sphere function (minimization)."""
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='GeneticAlgorithmOptimizer',
        objective_func=sphere_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=False,
        num_groups=3,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    _, best = method.solver(
        search_agents_no=20,
        max_iter=50
    )
    
    # Should find a solution close to [0, 0] with fitness near 0
    assert best.fitness < 0.5
    assert np.all(np.abs(best.position) < 1.0)


def test_rastrigin_function_minimization():
    """Test GeneticAlgorithmOptimizer on Rastrigin function (minimization)."""
    def rastrigin_function(x):
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    method = create_solver(
        solver_name='GeneticAlgorithmOptimizer',
        objective_func=rastrigin_function,
        lb=-5.12,
        ub=5.12,
        dim=2,
        maximize=False,
        num_groups=2,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    _, best = method.solver(
        search_agents_no=30,
        max_iter=100
    )
    
    # Should find a solution with reasonably low fitness
    assert best.fitness < 10.0


def test_maximization():
    """Test GeneticAlgorithmOptimizer on maximization problem."""
    def negative_sphere(x):
        return -np.sum(x**2)
    
    method = create_solver(
        solver_name='GeneticAlgorithmOptimizer',
        objective_func=negative_sphere,
        lb=-2.0,
        ub=2.0,
        dim=2,
        maximize=True,
        num_groups=2,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    _, best = method.solver(
        search_agents_no=20,
        max_iter=50
    )
    
    # Should find a solution with fitness near 0 (maximizing negative sphere)
    assert best.fitness > -0.5

if __name__ == "__main__":
    # Run the tests
    test_sphere_function_minimization()
    test_rastrigin_function_minimization()
    test_maximization()
    print("All tests passed!")
