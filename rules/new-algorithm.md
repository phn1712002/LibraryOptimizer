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
- `_sort_population(population)`: Sort population
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
