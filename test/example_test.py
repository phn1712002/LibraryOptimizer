import argparse
import numpy as np
import sys
import os

# Add the current directory to Python path to allow imports from utils and src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src import create_solver, show_solvers
from utils.func_test import sphere_function, rastrigin_function, negative_sphere, zdt1_function, zdt5_function

# Global variable to store the algorithm name
SOLVER_NAME = "{SOLVER_NAME}"

def test_sphere_function():
    method = create_solver(
        solver_name=SOLVER_NAME,
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
    method = create_solver(
        solver_name=SOLVER_NAME,
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
    method = create_solver(
        solver_name=SOLVER_NAME,
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

def test_multiobjective_zdt1():
    method = create_solver(
        solver_name=SOLVER_NAME,
        objective_func=zdt1_function,
        lb=np.array([0.0, 0.0]),
        ub=np.array([1.0, 1.0]),
        dim=2,
        archive_size=50,
        maximize=False
    )
    
    history_archive, final_archive = method.solver(
        search_agents_no=100,
        max_iter=100
    )
    
    # Should find a diverse set of non-dominated solutions
    assert len(final_archive) > 0
    assert len(final_archive[0].multi_fitness) == 2
    
    # Check that solutions are within bounds
    for solution in final_archive:
        assert np.all(solution.position >= 0.0)
        assert np.all(solution.position <= 1.0)
        assert len(solution.multi_fitness) == 2

def test_multiobjective_zdt5():
    method = create_solver(
        solver_name=SOLVER_NAME,
        objective_func=zdt5_function,
        lb=np.array([0.0] * 2),
        ub=np.array([1.0] * 2),
        dim=2,
        archive_size=100,
        maximize=False
    )
    
    history_archive, final_archive = method.solver(
        search_agents_no=200,
        max_iter=200
    )
    
    # Should find a diverse set of non-dominated solutions
    assert len(final_archive) > 0
    assert len(final_archive[0].multi_fitness) == 3

def run_all_tests():
    '''Run all tests and report results'''
    test_results = {}
    
    try:
        test_sphere_function()
        test_results['sphere_function'] = 'PASSED'
        print("✓ Sphere function test passed")
    except Exception as e:
        test_results['sphere_function'] = f'FAILED: {e}'
        print(f"✗ Sphere function test failed: {e}")
    
    try:
        test_rastrigin_function()
        test_results['rastrigin_function'] = 'PASSED'
        print("✓ Rastrigin function test passed")
    except Exception as e:
        test_results['rastrigin_function'] = f'FAILED: {e}'
        print(f"✗ Rastrigin function test failed: {e}")
    
    try:
        test_maximization()
        test_results['maximization'] = 'PASSED'
        print("✓ Maximization test passed")
    except Exception as e:
        test_results['maximization'] = f'FAILED: {e}'
        print(f"✗ Maximization test failed: {e}")
    
    try:
        test_multiobjective_zdt1()
        test_results['multiobjective_zdt1'] = 'PASSED'
        print("✓ Multi-objective ZDT1 test passed")
    except Exception as e:
        test_results['multiobjective_zdt1'] = f'FAILED: {e}'
        print(f"✗ Multi-objective ZDT1 test failed: {e}")
    
    try:
        test_multiobjective_zdt5()
        test_results['multiobjective_zdt5'] = 'PASSED'
        print("✓ Multi-objective ZDT5 test passed")
    except Exception as e:
        test_results['multiobjective_zdt5'] = f'FAILED: {e}'
        print(f"✗ Multi-objective ZDT5 test failed: {e}")
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result == 'PASSED' else "✗ FAILED"
        print(f"{test_name:30} {status}")
    
    # Return True if all tests passed
    return all(result == 'PASSED' for result in test_results.values())

def main():
    parser = argparse.ArgumentParser(description='Test optimization algorithms')
    parser.add_argument('-name', type=str, help='Name of the algorithm to test')
    parser.add_argument('-list', action='store_true', help='Show available solvers')
    
    args = parser.parse_args()
    
    if args.list:
        show_solvers()
        return
    
    if args.name:
        # Test specific algorithm
        SOLVER_NAME = args.name
        print(f"Testing algorithm: {SOLVER_NAME}")
        
        # Update the global SOLVER_NAME for the test functions
        globals()['SOLVER_NAME'] = SOLVER_NAME
        
        # Run all tests with the specified algorithm
        success = run_all_tests()
        if success:
            print(f'\n🎉 All tests passed for {SOLVER_NAME}!')
        else:
            print(f'\n❌ Some tests failed for {SOLVER_NAME}!')
    else:
        # Default behavior: run all tests with placeholder name
        print("No algorithm specified. Running all tests with placeholder name...")
        success = run_all_tests()
        if success:
            print('\n🎉 All tests passed!')
        else:
            print('\n❌ Some tests failed!')

if __name__ == '__main__':
    main()
