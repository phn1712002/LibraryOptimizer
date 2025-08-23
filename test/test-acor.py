import numpy as np
from src import create_solver

def test_sphere_function_minimization():
    """Test ACOR on sphere function (minimization)"""
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='AntColonyOptimizer',
        objective_func=sphere_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=False
    )
    
    _, best_solution = method.solver(
        search_agents_no=10,
        max_iter=50
    )
    
    # Should find a solution close to [0, 0] with fitness near 0
    print(f"Best position: {best_solution.position}")
    print(f"Best fitness: {best_solution.fitness}")
    assert best_solution.fitness < 0.1
    assert np.all(np.abs(best_solution.position) < 0.5)

def test_sphere_function_maximization():
    """Test ACOR on negative sphere function (maximization)"""
    def negative_sphere(x):
        return -np.sum(x**2)
    
    method = create_solver(
        solver_name='AntColonyOptimizer',
        objective_func=negative_sphere,
        lb=-2.0,
        ub=2.0,
        dim=2,
        maximize=True
    )
    
    _, best_solution = method.solver(
        search_agents_no=10,
        max_iter=50
    )
    
    # Should find a solution with fitness near 0 (maximizing negative sphere)
    print(f"Best position: {best_solution.position}")
    print(f"Best fitness: {best_solution.fitness}")
    assert best_solution.fitness > -0.1

def test_rastrigin_function():
    """Test ACOR on Rastrigin function (minimization)"""
    def rastrigin_function(x):
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    method = create_solver(
        solver_name='AntColonyOptimizer',
        objective_func=rastrigin_function,
        lb=-5.12,
        ub=5.12,
        dim=2,
        maximize=False
    )
    
    _, best_solution = method.solver(
        search_agents_no=20,
        max_iter=100
    )
    
    # Should find a solution with reasonably low fitness
    print(f"Best position: {best_solution.position}")
    print(f"Best fitness: {best_solution.fitness}")
    assert best_solution.fitness < 5.0

def test_custom_parameters():
    """Test ACOR with custom parameters"""
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='AntColonyOptimizer',
        objective_func=sphere_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=False,
        q=0.3,  # Custom intensification factor
        zeta=0.8  # Custom deviation-distance ratio
    )
    
    _, best_solution = method.solver(
        search_agents_no=15,
        max_iter=50
    )
    
    print(f"Best position: {best_solution.position}")
    print(f"Best fitness: {best_solution.fitness}")
    assert best_solution.fitness < 0.2

if __name__ == "__main__":
    print("Testing Ant Colony Optimization for Continuous Domains (ACOR)")
    print("=" * 60)
    
    print("\n1. Testing sphere function minimization:")
    test_sphere_function_minimization()
    print("✓ Passed")
    
    print("\n2. Testing sphere function maximization:")
    test_sphere_function_maximization()
    print("✓ Passed")
    
    print("\n3. Testing Rastrigin function:")
    test_rastrigin_function()
    print("✓ Passed")
    
    print("\n4. Testing custom parameters:")
    test_custom_parameters()
    print("✓ Passed")
    
    print("\nAll tests passed! 🎉")
