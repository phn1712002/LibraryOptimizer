import numpy as np
import pytest
from src import create_solver

def test_jaya_sphere_function():
    '''Test JAYA on sphere function (minimization)'''
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='JAYAOptimizer',
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

def test_jaya_rastrigin_function():
    '''Test JAYA on Rastrigin function (minimization)'''
    def rastrigin_function(x):
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    method = create_solver(
        solver_name='JAYAOptimizer',
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
    
    # Should find a solution with fitness reasonably low
    assert best.fitness < 5.0

def test_jaya_maximization():
    '''Test JAYA on maximization problem'''
    def negative_sphere(x):
        return -np.sum(x**2)
    
    method = create_solver(
        solver_name='JAYAOptimizer',
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

def test_jaya_high_dimension():
    '''Test JAYA on higher dimensional problem'''
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='JAYAOptimizer',
        objective_func=sphere_function,
        lb=-10.0,
        ub=10.0,
        dim=10,
        maximize=False
    )
    
    _, best = method.solver(
        search_agents_no=100,
        max_iter=200
    )
    
    # Should find a reasonable solution in higher dimensions
    assert best.fitness < 1.0

def test_jaya_bounds_respect():
    '''Test that JAYA respects variable bounds'''
    def simple_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='JAYAOptimizer',
        objective_func=simple_function,
        lb=-1.0,
        ub=1.0,
        dim=3,
        maximize=False
    )
    
    _, best = method.solver(
        search_agents_no=30,
        max_iter=50
    )
    
    # All positions should be within bounds
    assert np.all(best.position >= -1.0)
    assert np.all(best.position <= 1.0)

if __name__ == "__main__":
    # Run tests manually for quick verification
    test_jaya_sphere_function()
    test_jaya_rastrigin_function()
    test_jaya_maximization()
    test_jaya_high_dimension()
    test_jaya_bounds_respect()
    print("All JAYA tests passed!")
