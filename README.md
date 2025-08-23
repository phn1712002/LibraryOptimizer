# Library Optimizer

A comprehensive Python library for metaheuristic optimization algorithms, featuring 18+ state-of-the-art optimization algorithms with a unified interface.

## Features

- **18+ Optimization Algorithms**: Grey Wolf Optimizer, Particle Swarm Optimization, Artificial Bee Colony, Whale Optimization, and more
- **Unified Interface**: Consistent API across all algorithms
- **Visualization**: Built-in progress tracking and convergence plotting
- **Benchmark Functions**: Ready-to-use test functions for evaluation
- **Extensible**: Easy to add new algorithms following established patterns
- **Type Hints**: Full type annotations for better development experience

## Installation

```bash
git clone https://github.com/phn1712002/LibraryOptimizer
pip install -e . 
```

## Quick Start

```python
import numpy as np
from library_optimizer import create_solver

# Define objective function (Sphere function)
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

## Available Algorithms

- Grey Wolf Optimizer (GWO)
- Whale Optimization Algorithm (WOA)
- Particle Swarm Optimization (PSO)
- Artificial Bee Colony (ABC)
- Ant Colony Optimization (ACO)
- Bat Algorithm
- Artificial Ecosystem-based Optimization (AEO)
- Cuckoo Search (CS)
- Dingo Optimization Algorithm (DOA)
- Firefly Algorithm
- JAYA Algorithm
- Modified Social Group Optimization (MSGO)
- Moss Growth Optimization (MGO)
- Shuffled Frog Leaping Algorithm (SFLA)
- Teaching-Learning-based Optimization (TLBO)
- Prairie Dogs Optimization (PDO)
- Simulated Annealing (SA)
- Genetic Algorithm (GA)

## Documentation

Full documentation is available at [GitHub Repository](https://github.com/HoangggNam/LibraryOptimizer).

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Adding New Algorithms

To add a new optimization algorithm, follow the template in `rules/new-algorithm.md` and ensure:

1. Inherit from the base `Solver` class
2. Implement the required interface
3. Add comprehensive tests
4. Update the registry in `src/__init__.py`

## Testing

Run the test suite:

```bash
python -m pytest test/ -v
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please consider citing:

```bibtex
@software{LibraryOptimizer,
  author = {HoangggNam},
  title = {Library Optimizer: A Python Library for Metaheuristic Optimization},
  year = {2025},
  url = {https://github.com/HoangggNam/LibraryOptimizer}
}
```

## Contact

- Author: HoangggNam
- Email: phn1712002@gmai.com
- GitHub: [HoangggNam](https://github.com/HoangggNam)

## Acknowledgments

This library builds upon research in metaheuristic optimization and implements algorithms from various scientific publications.
