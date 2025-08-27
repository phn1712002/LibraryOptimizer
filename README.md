# Library Optimizer

A comprehensive Python library for metaheuristic optimization algorithms, featuring 20+ state-of-the-art optimization algorithms with both single-objective and multi-objective capabilities, all with a unified interface.

### And especially there is a library for MATLAB: [Libraries Optimizer for MATLAB](/matlab/)


## Features

- **20+ Optimization Algorithms**: Grey Wolf Optimizer, Particle Swarm Optimization, Artificial Bee Colony, Whale Optimization, and more
- **Multi-Objective Support**: Automatic generation of multi-objective versions from single-objective algorithms
- **Unified Interface**: Consistent API across all algorithms for both single and multi-objective optimization
- **Visualization**: Built-in progress tracking and convergence plotting
- **Benchmark Functions**: Ready-to-use test functions for evaluation (single and multi-objective)
- **Extensible**: Easy to add new algorithms following established patterns
- **Type Hints**: Full type annotations for better development experience
- **Automatic Detection**: System automatically detects objective function type and selects appropriate solver

## Installation

```bash
git clone https://github.com/phn1712002/LibraryOptimizer
cd LibraryOptimizer && pip install -e . 
```

## Quick Start

### Single-Objective Optimization

```python
import numpy as np
from LibraryOptimizer import create_solver

# Define objective function (Sphere function - single objective)
def sphere_function(x):
    return np.sum(x**2)

# Create optimizer instance
optimizer = create_solver(
    solver_name='GreyWolfOptimizer',
    objective_func=sphere_function,
    lb=-5.0,  # Lower bound
    ub=5.0,   # Upper bound
    dim=10,   # Problem dimension
    maximize=False  # Minimization problem
)

# Run optimization
history, best_solution = optimizer.solver(
    search_agents_no=50,  # Population size
    max_iter=100          # Maximum iterations
)

print(f"Best solution: {best_solution.position}")
print(f"Best fitness: {best_solution.fitness}")
```

### Multi-Objective Optimization

```python
import numpy as np
from LibraryOptimizer import create_solver

# Define multi-objective function (ZDT1 benchmark)
def zdt1_function(x):
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    h = 1 - np.sqrt(f1 / g)
    f2 = g * h
    return np.array([f1, f2])

# Create optimizer instance - system automatically detects multi-objective function
optimizer = create_solver(
    solver_name='GreyWolfOptimizer',  # Same name, auto-detects multi-objective
    objective_func=zdt1_function,
    lb=np.array([0.0, 0.0]),  # Lower bounds
    ub=np.array([1.0, 1.0]),  # Upper bounds
    dim=2,                    # Problem dimension
    archive_size=100          # Archive size for multi-objective optimization
)

# Run optimization
history_archive, final_archive = optimizer.solver(
    search_agents_no=100,     # Population size
    max_iter=200              # Maximum iterations
)

print(f"Found {len(final_archive)} non-dominated solutions")
print(f"First solution: {final_archive[0].position} -> {final_archive[0].multi_fitness}")
```

## Available Algorithms

All algorithms are available in both single-objective and multi-objective versions:

- Grey Wolf Optimizer (GWO) / Multi-Objective GWO
- Whale Optimization Algorithm (WOA) / Multi-Objective WOA
- Particle Swarm Optimization (PSO) / Multi-Objective PSO
- Artificial Bee Colony (ABC) / Multi-Objective ABC
- Ant Colony Optimization (ACO) / Multi-Objective ACO
- Bat Algorithm / Multi-Objective Bat
- Artificial Ecosystem-based Optimization (AEO) / Multi-Objective AEO
- Cuckoo Search (CS) / Multi-Objective CS
- Dingo Optimization Algorithm (DOA) / Multi-Objective DOA
- Firefly Algorithm / Multi-Objective Firefly
- JAYA Algorithm / Multi-Objective JAYA
- Modified Social Group Optimization (MSGO) / Multi-Objective MSGO
- Moss Growth Optimization (MGO) / Multi-Objective MGO
- Shuffled Frog Leaping Algorithm (SFLA) / Multi-Objective SFLA
- Teaching-Learning-based Optimization (TLBO) / Multi-Objective TLBO
- Prairie Dogs Optimization (PDO) / Multi-Objective PDO
- Simulated Annealing (SA) / Multi-Objective SA
- Genetic Algorithm (GA) / Multi-Objective GA
- More ....

The system automatically selects the appropriate version based on your objective function type.

## Documentation

Full documentation is available at [GitHub Repository](https://github.com/phn1712002/LibraryOptimizer).

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Adding New Algorithms

To add a new optimization algorithm, follow the template in `rules/single-objective.md`.
If you need to implement multiple fitness, follow the template in `rules/multi-objective.md`.
Or the system will automatically generate the multi-objective version of your algorithm using the automatic generation system described in `rules/auto-generation-multi.md`.

### Example Test Script

The library includes a comprehensive test script (`test/example_test.py`) that allows you to test specific optimization algorithms with various benchmark functions:

```bash
# Show available algorithms
python test/example_test.py -list

# Test a specific algorithm (e.g., GreyWolfOptimizer)
python test/example_test.py -name GreyWolfOptimizer

# Test another algorithm (e.g., ParticleSwarmOptimizer)  
python test/example_test.py -name ParticleSwarmOptimizer
```

The test script runs the following comprehensive tests:
- **Sphere Function**: Tests basic minimization capabilities
- **Rastrigin Function**: Tests performance on a multimodal function
- **Maximization**: Tests algorithm's ability to maximize functions
- **Multi-objective ZDT1**: Tests multi-objective optimization with 2 objectives
- **Multi-objective ZDT5**: Tests multi-objective optimization with 3 objectives

Each test validates that the algorithm can find reasonable solutions and provides detailed pass/fail reporting.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please consider citing:

```bibtex
@software{LibraryOptimizer,
  author = {phn1712002},
  title = {Library Optimizer: A Python Library for Metaheuristic Optimization with Multi-Objective Support},
  year = {2025},
  url = {https://github.com/phn1712002/LibraryOptimizer}
}
```

## Contact

- Author: phn1712002
- Email: phn1712002@gmail.com
- GitHub: [phn1712002](https://github.com/phn1712002)

## Acknowledgments

This library builds upon research in metaheuristic optimization and implements algorithms from various scientific publications.

## Languages
- [English](README.md)
- [Vietnamese](/docs/vi/README_VI.md)