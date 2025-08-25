# Rules for Writing New Algorithms for LibraryOptimizer

## Basic Structure

Each new algorithm must follow these rules to ensure consistency with the library:

### 1. File and Class Naming
- **File name**: `algorithm_name_optimizer.py` (snake_case)
- **Class name**: `AlgorithmNameOptimizer` (PascalCase)
- **Member class** (if needed): `AlgorithmMember` (PascalCase, simple) - Only create custom Member classes when additional attributes are required

### 2. Inheritance from Base Class
```python
import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
# Note: Use inherited utilities from Solver class instead of importing from utils.general
# Only import specific utilities that are not available through inheritance
```

### 3. Algorithm Class Structure
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
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize)
        
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set solver name (descriptive)
        self.name_solver = "Descriptive Algorithm Name"
        
        # Set algorithm-specific parameters with defaults
        self.param1 = kwargs.get('param1', default_value)
        self.param2 = kwargs.get('param2', default_value)
        # ... more parameters as needed
```

### 4. Custom Member Class (Only when needed)
```python
class AlgorithmMember(Member):
    def __init__(self, position: np.ndarray, fitness: float, additional_attr=None):
        super().__init__(position, fitness)
        self.additional_attr = additional_attr  # Additional algorithm-specific attributes
    
    def copy(self):
        return AlgorithmMember(self.position.copy(), self.fitness, self.additional_attr)
    
    def __str__(self):
        return f"Position: {self.position} - Fitness: {self.fitness} - Additional: {self.additional_attr}"
```

### 5. Main Solver Method (Standard Signature)
```python
def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
    """
    Main optimization method.
    
    Parameters:
    -----------
    search_agents_no : int
        Number of search agents (population size)
    max_iter : int
        Maximum number of iterations
        
    Returns:
    --------
    Tuple[List, Member]
        History of best solutions and the best solution found
    """
    # Initialize storage variables
    history_step_solver = []
    
    # Initialize population
    population = self._init_population(search_agents_no)
    
    # Initialize best solution using _sort_population
    sorted_population, _ = self._sort_population(population)
    best_solution = sorted_population[0].copy()
    
    # Start solver (show progress bar)
    self._begin_step_solver(max_iter)
    
    # Main optimization loop
    for iter in range(max_iter):
        # Algorithm-specific update logic here
        
        # Update best solution
        sorted_population, _ = self._sort_population(population)
        current_best = sorted_population[0]
        if self._is_better(current_best, best_solution):
            best_solution = current_best.copy()
        
        # Save history
        history_step_solver.append(best_solution.copy())
        
        # Call callback for progress tracking
        self._callbacks(iter, max_iter, best_solution)
    
    # End solver
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
        # Use custom Member class if needed, otherwise use base Member
        population.append(Member(position, fitness))  # or AlgorithmMember if custom
    return population
```

### 7. Utility Methods (Use inherited utilities)
```python
# Use inherited methods from Solver class instead of importing external functions
# For example, use self._sort_population() instead of importing sort_population
# Use self._is_better() for comparison, self._begin_step_solver() for progress tracking, etc.

# If you need to override a method, inherit and extend functionality
def _init_population(self, search_agents_no) -> List:
    # Override to provide custom initialization if needed
    # Otherwise, use the inherited method
    return super()._init_population(search_agents_no)
```

### 8. Register Algorithm
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
- `_sort_population(population)`: Sort population using utility function
- `_begin_step_solver(max_iter)`: Initialize progress bar
- `_callbacks(iter, max_iter, best)`: Update progress
- `_end_step_solver()`: Close progress bar and plot history

### Available Attributes
- `self.lb`, `self.ub`: Lower and upper bounds
- `self.dim`: Problem dimension
- `self.maximize`: Optimization direction (True: maximize, False: minimize)
- `self.objective_func`: Objective function
- `self.kwargs`: Additional parameters passed during initialization

## Multi-Objective Support

### Automatic Multi-Objective Generation
When creating a new single-objective algorithm, the system can automatically generate a multi-objective version. Follow these steps:

1. **Create the single-objective algorithm** following the standard pattern
2. **Create multi-objective version** in `src/multiobjective/` directory:
   ```python
   # File: src/multiobjective/algorithm_name_optimizer.py
   import numpy as np
   from typing import Callable, Union, Tuple, List
   from ._core import MultiObjectiveSolver, MultiObjectiveMember
   
   class AlgorithmMultiMember(MultiObjectiveMember):
       def __init__(self, position: np.ndarray, fitness: np.ndarray, additional_attr=None):
           super().__init__(position, fitness)
           self.additional_attr = additional_attr
       
       def copy(self):
           new_member = AlgorithmMultiMember(self.position.copy(), self.multi_fitness.copy(), 
                                           self.additional_attr)
           new_member.dominated = self.dominated
           new_member.grid_index = self.grid_index
           new_member.grid_sub_index = self.grid_sub_index
           return new_member
   
   class MultiObjectiveAlgorithmNameOptimizer(MultiObjectiveSolver):
       def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                    ub: Union[float, np.ndarray], dim: int, **kwargs):
           super().__init__(objective_func, lb, ub, dim, **kwargs)
           self.name_solver = "Multi-Objective Algorithm Name Optimizer"
           # Copy algorithm-specific parameters
           self.param1 = kwargs.get('param1', default_value)
       
       def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[AlgorithmMultiMember]]:
           # Implementation using algorithm-specific logic
           # and inherited multi-objective utilities
           pass
   ```

3. **Update registry mapping** in `src/__init__.py`:
   ```python
   _MULTI_OBJECTIVE_MAPPING: Dict[str, str] = {
       "AlgorithmNameOptimizer": "MultiObjectiveAlgorithmNameOptimizer",
       # ... existing mappings
   }
   ```

4. **Add import** in `src/__init__.py`:
   ```python
   from .multiobjective.algorithm_name_optimizer import *
   ```

5. **Update registry** in `src/__init__.py`:
   ```python
   _SOLVER_REGISTRY: Dict[str, Type[Solver]] = {
       # ... existing solvers
       "MultiObjectiveAlgorithmNameOptimizer": MultiObjectiveAlgorithmNameOptimizer,
   }
   ```

### Key Multi-Objective Concepts
- Use `MultiObjectiveSolver` and `MultiObjectiveMember` as base classes
- Fitness is stored in `multi_fitness` array, `fitness` is for compatibility
- Solutions are compared using `_dominates()` instead of `_is_better()`
- Return archive of non-dominated solutions instead of single best solution
- Use grid-based selection for leader selection from archive

See `rules/multi-objective.md` for detailed multi-objective implementation guidelines.

## Code Style Rules

### Import Structure
```python
# Standard library imports
import numpy as np
from typing import Callable, Union, Tuple, List

# Third-party imports (if needed)
import matplotlib.pyplot as plt

# Local imports (relative)
from ._core import Solver, Member
# Note: Use inherited utilities from Solver class instead of importing from utils.general
# Only import specific utilities that are not available through inheritance
```

### Type Hints
Always use complete type hints for public methods.

### Documentation
Comprehensive docstrings for classes and main methods following the established pattern.

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

## Implementation Patterns from Existing Code

### Pattern 1: Simple Algorithm (GWO, Whale)
- No custom Member class
- Basic population initialization
- Simple update rules in main loop

### Pattern 2: Algorithm with Custom Member (PSO, ABC)
- Custom Member class with additional attributes
- Specialized initialization
- Complex update logic requiring additional state

### Pattern 3: Complex Algorithm (SFLA, MGO)
- Multiple phases or complex update strategies
- Additional utility methods
- Sophisticated parameter handling

## Best Practices

1. **Performance**: Use NumPy vectorization when possible
2. **Memory**: Avoid creating unnecessary large arrays
3. **Readability**: Use clear variable names, add comments for complex logic
4. **Modularity**: Separate complex logic into individual methods
5. **Error handling**: Handle special cases (division by zero, etc.)
6. **Reproducibility**: Ensure results are reproducible

## Testing Guidelines

### Standard Test Functions
Every algorithm should be tested with standard benchmark functions:
- Sphere function (minimization)
- Rastrigin function (minimization) 
- Negative sphere (maximization)

### Test Structure
```python
def test_sphere_function():
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
    
    _, best = method.solver(
        search_agents_no=100,
        max_iter=100
    )
    
    assert best.fitness < 0.1
    assert np.all(np.abs(best.position) < 0.5)
```

## Utility Functions Available

Note: These utility functions should be accessed through inherited methods from the Solver class rather than imported directly:
- `self._sort_population()`: For sorting population (inherited from Solver)
- `self._is_better()`: For comparing solutions (inherited from Solver)
- `self._begin_step_solver()`: For progress tracking (inherited from Solver)
- `self._callbacks()`: For iteration callbacks (inherited from Solver)
- `self._end_step_solver()`: For finalizing solver (inherited from Solver)

Only import specific utilities from `utils/general.py` if they are not available through inheritance and are truly needed for the algorithm.

## Common Implementation Issues to Avoid

1. **Direct import of utility functions**: Use inherited methods from Solver class instead of importing from utils.general
2. **Missing type hints**: Always include complete type annotations
3. **Incomplete docstrings**: Follow the established documentation pattern
4. **Hard-coded parameters**: Use kwargs with defaults instead
5. **Memory leaks**: Always use `.copy()` for Member objects

## Example References

Check existing files for reference:
- `greywolf_optimizer.py` - Simple algorithm pattern
- `particleswarm_optimizer.py` - Custom Member class pattern  
- `shuffledfrogleaping_optimizer.py` - Complex algorithm pattern
- `artificialbeecolony_optimizer.py` - Multi-phase algorithm

By following these updated rules, new algorithms will integrate well with the existing library and maintain consistency with your implementation patterns.

*This document was last updated: 2025-08-23*
