import numpy as np
from src import create_solver


def test_sphere_function_minimization():
    """Test AEO on sphere function (minimization)."""
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='ArtificialEcosystemOptimizer',
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
    """Test AEO on Rastrigin function (minimization)."""
    def rastrigin_function(x):
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    method = create_solver(
        solver_name='ArtificialEcosystemOptimizer',
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
    """Test AEO on maximization problem."""
    def negative_sphere(x):
        return -np.sum(x**2)
    
    method = create_solver(
        solver_name='ArtificialEcosystemOptimizer',
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


def test_boundary_handling():
    """Test that AEO respects boundary constraints."""
    def simple_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='ArtificialEcosystemOptimizer',
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


def test_convergence():
    """Test that AEO shows improvement over iterations."""
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='ArtificialEcosystemOptimizer',
        objective_func=sphere_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=False
    )
    
    history, best = method.solver(
        search_agents_no=30,
        max_iter=50
    )
    
    # Should have correct number of history entries
    assert len(history) == 50
    
    # Best solution should be better or equal to initial best
    assert best.fitness <= history[0].fitness


if __name__ == "__main__":
    # Run the tests
    test_sphere_function_minimization()
    test_rastrigin_function_minimization()
    test_maximization()
    test_boundary_handling()
    test_convergence()
    print("All tests passed!")
