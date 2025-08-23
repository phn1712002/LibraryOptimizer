import numpy as np
import pytest
from src import create_solver

def test_cuckoo_search_sphere_function():
    """Test Cuckoo Search on sphere function (minimization)"""
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='CuckooSearchOptimizer',
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

def test_cuckoo_search_rastrigin_function():
    """Test Cuckoo Search on Rastrigin function (minimization)"""
    def rastrigin_function(x):
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    method = create_solver(
        solver_name='CuckooSearchOptimizer',
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

def test_cuckoo_search_maximization():
    """Test Cuckoo Search on maximization problem"""
    def negative_sphere(x):
        return -np.sum(x**2)
    
    method = create_solver(
        solver_name='CuckooSearchOptimizer',
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

def test_cuckoo_search_custom_parameters():
    """Test Cuckoo Search with custom parameters"""
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='CuckooSearchOptimizer',
        objective_func=sphere_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=False,
        pa=0.3,  # Custom discovery rate
        beta=1.8  # Custom Levy exponent
    )
    
    _, best = method.solver(
        search_agents_no=50,
        max_iter=100
    )
    
    # Should find a solution close to [0, 0] with fitness near 0
    assert best.fitness < 0.1
    assert np.all(np.abs(best.position) < 0.5)

def test_cuckoo_search_history():
    """Test that Cuckoo Search returns proper history"""
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='CuckooSearchOptimizer',
        objective_func=sphere_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=False
    )
    
    history, best = method.solver(
        search_agents_no=50,
        max_iter=50
    )
    
    # History should have correct length
    assert len(history) == 50
    
    # Best solution should be the last in history
    assert np.allclose(history[-1].position, best.position)
    assert np.isclose(history[-1].fitness, best.fitness)
    
    # Fitness should improve over time (for minimization)
    for i in range(1, len(history)):
        assert history[i].fitness <= history[i-1].fitness

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
