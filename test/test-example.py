import numpy as np
from src import create_solver

def test_sphere_function():
    '''Test method on sphere function (minimization)'''
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='Name',
        objective_func=sphere_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=False
    )
    
    _, best = method.solver(
        search_agents_no=100,
        max_iter=100
    )
    
    # Should find a solution close to [0, 0] with fitness near 0
    assert best.fitness < 0.1
    assert np.all(np.abs(best.position) < 0.5)

def test_rastrigin_function():
    '''Test method on Rastrigin function (minimization)'''
    def rastrigin_function(x):
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    method = create_solver(
        solver_name='Name',
        objective_func=rastrigin_function,
        lb=-5.12,
        ub=5.12,
        dim=2,
        maximize=False
    )
    
    _, best = method.solver(
        search_agents_no=100,
        max_iter=100
    )
    
    # Should find a solution with fitness reasonably low
    assert best.fitness < 5.0

def test_maximization():
    '''Test method on maximization problem'''
    def negative_sphere(x):
        return -np.sum(x**2)
    
    method = create_solver(
        solver_name='Name',
        objective_func=negative_sphere,
        lb=-2.0,
        ub=2.0,
        dim=2,
        maximize=True
    )
    
    _, best = method.solver(
        search_agents_no=100,
        max_iter=100
    )
    
    # Should find a solution with fitness near 0 (maximizing negative sphere)
    assert best.fitness > -0.1

if __name__ == '__main__':
    test_sphere_function()
    test_rastrigin_function()
    test_maximization()
    print('All tests passed!')
