import numpy as np
import pytest
from src import create_solver


def test_bat_optimizer_sphere_function():
    """Test BAT optimizer on sphere function (minimization)"""
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='BatOptimizer',
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


def test_bat_optimizer_rastrigin_function():
    """Test BAT optimizer on Rastrigin function (minimization)"""
    def rastrigin_function(x):
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    method = create_solver(
        solver_name='BatOptimizer',
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


def test_bat_optimizer_maximization():
    """Test BAT optimizer on maximization problem"""
    def negative_sphere(x):
        return -np.sum(x**2)
    
    method = create_solver(
        solver_name='BatOptimizer',
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


def test_bat_optimizer_custom_parameters():
    """Test BAT optimizer with custom parameters"""
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='BatOptimizer',
        objective_func=sphere_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=False,
        fmin=0.1,
        fmax=1.9,
        alpha=0.95,
        gamma=0.85,
        ro=0.3
    )
    
    _, best = method.solver(
        search_agents_no=30,
        max_iter=50
    )
    
    # Should find a reasonable solution
    assert best.fitness < 1.0


def test_bat_optimizer_history_tracking():
    """Test that BAT optimizer properly tracks history"""
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='BatOptimizer',
        objective_func=sphere_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=False
    )
    
    history, best = method.solver(
        search_agents_no=20,
        max_iter=30
    )
    
    # History should have correct length
    assert len(history) == 30
    
    # Best solution should be the last in history
    assert np.allclose(history[-1].position, best.position)
    assert history[-1].fitness == best.fitness
    
    # Fitness should generally improve over time (not strictly monotonic due to stochastic nature)
    final_fitness = history[-1].fitness
    initial_fitness = history[0].fitness
    assert final_fitness <= initial_fitness


if __name__ == "__main__":
    # Run tests
    test_bat_optimizer_sphere_function()
    test_bat_optimizer_rastrigin_function()
    test_bat_optimizer_maximization()
    test_bat_optimizer_custom_parameters()
    test_bat_optimizer_history_tracking()
    print("All BAT optimizer tests passed!")
