# Rules for Writing New Algorithms for LibraryOptimizer

## Basic Structure

Each new algorithm must follow these rules to ensure consistency with the library:

### 1. File and Class Naming
- **File name**: `algorithm_name_optimizer.py` (snake_case)
- **Class name**: `AlgorithmNameOptimizer` (PascalCase)
- **Member class** (if needed): `AlgorithmMember` (PascalCase, simple)

### 2. Inheritance from Base Class
```python
from .core import Solver, Member
```

### 3. Algorithm Class Structure
```python
class AlgorithmNameOptimizer(Solver):
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize)
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set solver name
        self.name_solver = "Algorithm Name Optimizer"
        
        # Set default parameters
        self.param1 = kwargs.get('param1', default_value)
        self.param2 = kwargs.get('param2', default_value)
```

### 4. Custom Member Class (if needed)
```python
class AlgorithmMember(Member):
    def __init__(self, position: np.ndarray, fitness: float, additional_attr=None):
        super().__init__(position, fitness)
        self.additional_attr = additional_attr  # Additional attributes
    
    def copy(self):
        return AlgorithmMember(self.position.copy(), self.fitness, self.additional_attr)
    
    def __str__(self):
        return f"Position: {self.position} - Fitness: {self.fitness} - Additional: {self.additional_attr}"
```

### 5. Main Solver Method
```python
def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
    # 1. Initialize population
    population = self._init_population(search_agents_no)
    
    # 2. Initialize best solution
    sorted_population, _ = self._sort_population(population)
    best_solution = sorted_population[0].copy()
    
    # 3. Initialize history
    history_step_solver = []
    
    # 4. Start solver (show progress bar)
    self._begin_step_solver(max_iter)
    
    # 5. Main optimization loop
    for iter in range(max_iter):
        # Update parameters (if needed)
        # a = 2 - iter * (2 / max_iter)  # Example
        
        # Update each search agent
        for i in range(search_agents_no):
            # Position update logic
            new_position = self._update_position(population[i], iter, max_iter)
            new_position = np.clip(new_position, self.lb, self.ub)
            
            # Evaluate new fitness
            new_fitness = self.objective_func(new_position)
            
            # Compare and update
            if self._is_better(AlgorithmMember(new_position, new_fitness), population[i]):
                population[i].position = new_position
                population[i].fitness = new_fitness
        
        # Update best solution
        sorted_population, _ = self._sort_population(population)
        current_best = sorted_population[0]
        if self._is_better(current_best, best_solution):
            best_solution = current_best.copy()
        
        # Save history
        history_step_solver.append(best_solution.copy())
        
        # Call callback
        self._callbacks(iter, max_iter, best_solution)
    
    # 6. End solver
    self.history_step_solver = history_step_solver
    self.best_solver = best_solution
    self._end_step_solver()
    
    return history_step_solver, best_solution
```

### 6. Population Initialization Method
```python
def _init_population(self, search_agents_no) -> List:
    population = []
    for _ in range(search_agents_no):
        position = np.random.uniform(self.lb, self.ub, self.dim)
        fitness = self.objective_func(position)
        # Use custom Member class if needed
        population.append(AlgorithmMember(position, fitness, additional_attr=value))
    return population
```

### 7. Register Algorithm
In `src/__init__.py`, add to registry:
```python
from .algorithm_name_optimizer import AlgorithmNameOptimizer

_SOLVER_REGISTRY: Dict[str, Type[Solver]] = {
    # ... existing solvers
    "AlgorithmNameOptimizer": AlgorithmNameOptimizer,
}
```

## Available Methods from Solver Class

### Utility Methods
- `_is_better(member1, member2)`: Compare two solutions
- `sort_population(population, maximize)`: Sort population
- `_begin_step_solver(max_iter)`: Initialize progress bar
- `_callbacks(iter, max_iter, best)`: Update progress
- `_end_step_solver()`: Close progress bar and plot history

### Available Attributes
- `self.lb`, `self.ub`: Lower and upper bounds
- `self.dim`: Problem dimension
- `self.maximize`: Optimization direction (True: maximize, False: minimize)
- `self.objective_func`: Objective function

## Code Style Rules

### Import
```python
import numpy as np
from typing import Callable, Union, Tuple, List
from .core import Solver, Member
from utils.general import sort_population, roulette_wheel_selection  # If needed
```

### Type Hints
Always use complete type hints for public methods.

### Documentation
Add docstring for class and main methods:
```python
class AlgorithmNameOptimizer(Solver):
    """
    Brief description of the algorithm.
    
    Parameters:
    -----------
    objective_func : Callable
        Objective function to optimize
    lb : Union[float, np.ndarray]
        Lower bounds for variables
    ub : Union[float, np.ndarray]
        Upper bounds for variables  
    dim : int
        Problem dimension
    maximize : bool, optional
        Optimization direction, default is True (maximize)
    **kwargs
        Additional algorithm parameters
    """
```

### Boundary Handling
Always use `np.clip()` to ensure positions stay within bounds:
```python
new_position = np.clip(new_position, self.lb, self.ub)
```

### Copy Objects
Always use `.copy()` when storing solutions to avoid reference issues:
```python
best_solution = current_best.copy()
history_step_solver.append(best_solution.copy())
```

## Example References

Check existing files for reference:
- `greywolf_optimizer.py` - Simple algorithm
- `particleswarm_optimizer.py` - Algorithm with custom Member class
- `artificialbeecolony_optimizer.py` - Complex algorithm with multiple phases
- `whale_optimizer.py` - Algorithm with multiple strategies

## Best Practices

1. **Performance**: Use NumPy vectorization when possible
2. **Memory**: Avoid creating unnecessary large arrays
3. **Readability**: Use clear variable names, add comments for complex logic
4. **Modularity**: Separate logic into individual methods if needed
5. **Error handling**: Handle special cases (division by zero, etc.)
6. **Reproducibility**: Ensure results are reproducible (seed random if needed)

By following these rules, new algorithms will integrate well with the existing library and be easy to maintain.

## Testing Guidelines

### 1. Test File Structure
Each algorithm must have corresponding test files following this pattern:
- **Test file name**: `test-algorithm_name.py` (snake_case)
- **Test functions**: Use descriptive names starting with `test_`

### 2. Standard Test Functions
Every algorithm should be tested with these standard benchmark functions:

```python
def test_sphere_function():
    '''Test method on sphere function (minimization)'''
    def sphere_function(x):
        return np.sum(x**2)
    
    method = create_solver(
        solver_name='AlgorithmNameOptimizer',
        objective_func=sphere_function,
        lb=-5.0,
        ub=5.0,
        dim=2,
        maximize=False
    )
    
    _, best_position, best_fitness = method.solver(
        search_agents_no=100,
        max_iter=100
    )
    
    # Should find a solution close to [0, 0] with fitness near 0
    assert best_fitness < 0.1
    assert np.all(np.abs(best_position) < 0.5)

def test_rastrigin_function():
    '''Test method on Rastrigin function (minimization)'''
    def rastrigin_function(x):
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    method = create_solver(
        solver_name='AlgorithmNameOptimizer',
        objective_func=rastrigin_function,
        lb=-5.12,
        ub=5.12,
        dim=2,
        maximize=False
    )
    
    _, best_position, best_fitness = method.solver(
        search_agents_no=100,
        max_iter=100
    )
    
    # Should find a solution with fitness reasonably low
    assert best_fitness < 5.0

def test_maximization():
    '''Test method on maximization problem'''
    def negative_sphere(x):
        return -np.sum(x**2)
    
    method = create_solver(
        solver_name='AlgorithmNameOptimizer',
        objective_func=negative_sphere,
        lb=-2.0,
        ub=2.0,
        dim=2,
        maximize=True
    )
    
    _, best_position, best_fitness = method.solver(
        search_agents_no=100,
        max_iter=100
    )
    
    # Should find a solution with fitness near 0 (maximizing negative sphere)
    assert best_fitness > -0.1
```

### 3. Running Tests
```bash
# Run specific test file
python -m pytest test/test-algorithm_name.py -v

# Run all tests
python -m pytest test/ -v

# Run with coverage
python -m pytest test/ --cov=src --cov-report=html
```

### 4. Test Assertions
- Use meaningful assertions that verify algorithm behavior
- Test both minimization and maximization problems
- Verify boundary handling and constraint satisfaction
- Check convergence properties

### Best Practices for Utility Functions

1. **Import correctly**: Always import from `utils.general`
2. **Type safety**: Utility functions validate input types where appropriate
3. **Reusability**: Use these functions instead of reimplementing common operations
4. **Consistency**: Ensures uniform behavior across different algorithms
5. **Documentation**: All utility functions include comprehensive docstrings

### Adding New Utility Functions

When adding new utility functions to `utils/general.py`:

1. **Function signature**: Use proper type hints
2. **Documentation**: Add comprehensive docstrings
3. **Testing**: Include tests for new utility functions
4. **Backward compatibility**: Ensure existing functionality is not broken

```python
def new_utility_function(param1: Type, param2: Type) -> ReturnType:
    """
    Brief description of the function.
    
    Parameters:
    -----------
    param1 : Type
        Description of param1
    param2 : Type  
        Description of param2
        
    Returns:
    --------
    ReturnType
        Description of return value
    """
    # Implementation
    pass
```

## Performance Testing

### Benchmark Functions
Test algorithms with standard benchmark functions:
- Sphere function (convex, easy)
- Rastrigin function (multimodal, difficult)
- Ackley function (multimodal, moderate)
- Rosenbrock function (non-convex, difficult)

### Performance Metrics
- Convergence speed
- Solution quality
- Computational efficiency
- Memory usage


## Code Quality

### Linting and Formatting
```bash
# Run flake8 for code quality
flake8 src/ --max-line-length=120

# Run black for code formatting
black src/

# Run isort for import sorting
isort src/
```

### Type Checking
```bash
# Run mypy for type checking
mypy src/ --ignore-missing-imports
```

By following these comprehensive guidelines, new algorithms will be well-tested, maintainable, and consistent with the library's standards.

*This document was last updated: 2025-01-23*
