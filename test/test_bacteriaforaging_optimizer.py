"""
Test script for Bacteria Foraging Optimization algorithm.
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import create_solver

def sphere_function(x):
    """Sphere function - minimization problem."""
    return np.sum(x**2)

def negative_sphere_function(x):
    """Negative sphere function - maximization problem."""
    return -np.sum(x**2)

def test_bfo_minimization():
    """Test BFO for minimization problem."""
    print("Testing Bacteria Foraging Optimizer - Minimization")
    
    method = create_solver(
        solver_name="BacteriaForagingOptimizer",
        objective_func=sphere_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=False,
        n_elimination=2,
        n_reproduction=2,
        n_chemotaxis=5,
        n_swim=2,
        step_size=0.1,
        elimination_prob=0.25
    )
    
    history, best = method.solver(
        search_agents_no=20,
        max_iter=50
    )
    
    print(f"Best solution: {best.position}")
    print(f"Best fitness: {best.fitness}")
    assert best.fitness < 0.1, f"Expected fitness < 0.1, got {best.fitness}"
    print("✓ Minimization test passed\n")

def test_bfo_maximization():
    """Test BFO for maximization problem."""
    print("Testing Bacteria Foraging Optimizer - Maximization")
    
    method = create_solver(
        solver_name="BacteriaForagingOptimizer",
        objective_func=negative_sphere_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=True,
        n_elimination=2,
        n_reproduction=2,
        n_chemotaxis=5,
        n_swim=2,
        step_size=0.1,
        elimination_prob=0.25
    )
    
    history, best = method.solver(
        search_agents_no=20,
        max_iter=50
    )
    
    print(f"Best solution: {best.position}")
    print(f"Best fitness: {best.fitness}")
    assert best.fitness > -0.1, f"Expected fitness > -0.1, got {best.fitness}"
    print("✓ Maximization test passed\n")

def test_bfo_parameters():
    """Test BFO with different parameters."""
    print("Testing Bacteria Foraging Optimizer - Custom Parameters")
    
    method = create_solver(
        solver_name="BacteriaForagingOptimizer",
        objective_func=sphere_function,
        lb=-10.0,
        ub=10.0,
        dim=3,
        maximize=False,
        n_elimination=3,
        n_reproduction=3,
        n_chemotaxis=8,
        n_swim=3,
        step_size=0.05,
        elimination_prob=0.2
    )
    
    history, best = method.solver(
        search_agents_no=30,
        max_iter=30
    )
    
    print(f"Best solution: {best.position}")
    print(f"Best fitness: {best.fitness}")
    print("✓ Custom parameters test passed\n")

if __name__ == "__main__":
    print("Running Bacteria Foraging Optimization tests...\n")
    
    try:
        test_bfo_minimization()
        test_bfo_maximization()
        test_bfo_parameters()
        print("🎉 All tests passed successfully!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
