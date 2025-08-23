import numpy as np
import pytest
from src import create_solver


def test_mgo_sphere_function():
    """Test MGO on sphere function (minimization)"""
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='MossGrowthOptimizer',
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


def test_mgo_rastrigin_function():
    """Test MGO on Rastrigin function (minimization)"""
    def rastrigin_function(x):
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    method = create_solver(
        solver_name='MossGrowthOptimizer',
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


def test_mgo_maximization():
    """Test MGO on maximization problem"""
    def negative_sphere(x):
        return -np.sum(x**2)
    
    method = create_solver(
        solver_name='MossGrowthOptimizer',
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


def test_mgo_custom_parameters():
    """Test MGO with custom parameters"""
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='MossGrowthOptimizer',
        objective_func=sphere_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=False,
        w=1.5,  # Custom inertia weight
        rec_num=8,  # Custom record number
        divide_num=1,  # Custom divide number
        d1=0.3  # Custom probability threshold
    )
    
    _, best = method.solver(
        search_agents_no=30,
        max_iter=50
    )
    
    # Should still converge reasonably well
    assert best.fitness < 0.5


def test_mgo_boundary_handling():
    """Test MGO boundary handling"""
    def boundary_test_function(x):
        # Function that penalizes solutions near boundaries
        boundary_penalty = np.sum(np.maximum(0, x - 4.9) + np.maximum(0, -4.9 - x))
        return np.sum(x**2) + 1000 * boundary_penalty
    
    method = create_solver(
        solver_name='MossGrowthOptimizer',
        objective_func=boundary_test_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=False
    )
    
    _, best = method.solver(
        search_agents_no=40,
        max_iter=80
    )
    
    # Solution should stay within bounds
    assert np.all(best.position >= -5.0)
    assert np.all(best.position <= 5.0)
    # And should avoid boundary penalties
    assert best.fitness < 1.0


if __name__ == "__main__":
    # Run the tests
    test_mgo_sphere_function()
    test_mgo_rastrigin_function()
    test_mgo_maximization()
    test_mgo_custom_parameters()
    test_mgo_boundary_handling()
    print("All MGO tests passed!")
