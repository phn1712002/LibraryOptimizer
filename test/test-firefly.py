import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import create_solver

def test_sphere_function():
    '''Test Firefly Algorithm on sphere function (minimization)'''
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='FireflyOptimizer',
        objective_func=sphere_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=False
    )
    
    _, best = method.solver(
        search_agents_no=30,
        max_iter=100
    )
    
    # Should find a solution close to [0, 0] with fitness near 0
    print(f"Sphere function test - Best fitness: {best.fitness:.6f}")
    print(f"Sphere function test - Best position: {best.position}")
    assert best.fitness < 0.1
    assert np.all(np.abs(best.position) < 0.5)
    print("✓ Sphere function test passed")

def test_rastrigin_function():
    '''Test Firefly Algorithm on Rastrigin function (minimization)'''
    def rastrigin_function(x):
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    method = create_solver(
        solver_name='FireflyOptimizer',
        objective_func=rastrigin_function,
        lb=-5.12,
        ub=5.12,
        dim=2,
        maximize=False
    )
    
    _, best = method.solver(
        search_agents_no=30,
        max_iter=100
    )
    
    # Should find a solution with fitness reasonably low
    print(f"Rastrigin function test - Best fitness: {best.fitness:.6f}")
    print(f"Rastrigin function test - Best position: {best.position}")
    assert best.fitness < 5.0
    print("✓ Rastrigin function test passed")

def test_maximization():
    '''Test Firefly Algorithm on maximization problem'''
    def negative_sphere(x):
        return -np.sum(x**2)
    
    method = create_solver(
        solver_name='FireflyOptimizer',
        objective_func=negative_sphere,
        lb=-2.0,
        ub=2.0,
        dim=2,
        maximize=True
    )
    
    _, best = method.solver(
        search_agents_no=30,
        max_iter=100
    )
    
    # Should find a solution with fitness near 0 (maximizing negative sphere)
    print(f"Maximization test - Best fitness: {best.fitness:.6f}")
    print(f"Maximization test - Best position: {best.position}")
    assert best.fitness > -0.1
    print("✓ Maximization test passed")

def test_custom_parameters():
    '''Test Firefly Algorithm with custom parameters'''
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='FireflyOptimizer',
        objective_func=sphere_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=False,
        alpha=0.3,
        betamin=0.1,
        gamma=0.5,
        alpha_reduction=True,
        alpha_delta=0.95
    )
    
    _, best = method.solver(
        search_agents_no=30,
        max_iter=50
    )
    
    print(f"Custom parameters test - Best fitness: {best.fitness:.6f}")
    print(f"Custom parameters test - Best position: {best.position}")
    assert best.fitness < 0.5
    print("✓ Custom parameters test passed")

if __name__ == "__main__":
    print("Running Firefly Algorithm tests...")
    print("=" * 50)
    
    try:
        test_sphere_function()
        test_rastrigin_function()
        test_maximization()
        test_custom_parameters()
        print("=" * 50)
        print("All tests passed! ✓")
    except Exception as e:
        print(f"Test failed: {e}")
        raise
