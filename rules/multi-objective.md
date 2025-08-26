# Standards for Multi-Objective Optimization Algorithms

## Overview

This document defines the **rules and best practices** for implementing multi-objective versions of single-objective optimization algorithms in **LibraryOptimizer**.  

---

## Directory Structure

```
src/
├── _core.py                                # Base classes for Single-objective
├── _general.py                             # Common utilities
├── {algorithm_name_optimizer}.py           # Single-objective version
└── multiobjective/
    ├── _core.py                            # Base classes for multi-objective optimization
    └── {algorithm_name_optimizer}.py       # Multi-objective version
```

---

## Naming Conventions

### File Names
- **Single-objective** → `algorithm_name_optimizer.py` (snake_case)
- **Multi-objective** → Same filename, but placed inside `multiobjective/`

### Class Names
- **Single-objective** → `AlgorithmNameOptimizer` (PascalCase)
- **Multi-objective** → `MultiObjectiveAlgorithmNameOptimizer` (PascalCase, prefixed with `MultiObjective`)

### Member Classes
- **Single-objective** → `AlgorithmMember` (extends `Member`)
- **Multi-objective** → `AlgorithmMultiMember` (extends `MultiObjectiveMember`)

---

## Base Classes

### MultiObjectiveMember
```python
class MultiObjectiveMember(Member):
    def __init__(self, position: np.ndarray, fitness: np.ndarray):
        # For compatibility, use first fitness value
        super().__init__(position, 0)
        self.multi_fitness = np.array(fitness)  # Store all fitness values
        self.dominated = False
        self.grid_index = None
        self.grid_sub_index = None
    
    def copy(self):
        new_member = MultiObjectiveMember(self.position.copy(), self.multi_fitness.copy())
        new_member.dominated = self.dominated
        new_member.grid_index = self.grid_index
        new_member.grid_sub_index = self.grid_sub_index
        return new_member
```

### MultiObjectiveSolver

```python
class MultiObjectiveSolver(Solver):
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Multi-objective specific parameters
        self.n_objectives = objective_func(np.random.uniform(self.lb, self.ub, self.dim)).shape[0]
        self.archive_size = kwargs.get('archive_size', 100)
        self.archive = []
        
        # Grid-based selection parameters
        self.alpha = kwargs.get('alpha', 0.1)
        self.n_grid = kwargs.get('n_grid', 7)
        self.beta = kwargs.get('beta', 2)
        self.gamma = kwargs.get('gamma', 2)
        
        self.grid = None
```

---

## Required Methods

### 1. Constructor

```python
def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
             ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
    super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
    
    self.name_solver = "Multi-Objective Algorithm Name Optimizer"
    # Copy algorithm-specific parameters from single-objective version
    self.param1 = kwargs.get('param1', default_value)
```

### 2. Solver Method

```python
def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[AlgorithmMultiMember]]:
    """
    Main optimization routine for multi-objective version.
    
    Returns
    -------
    Tuple[List, List[AlgorithmMultiMember]]
        (history of archive states, final archive of non-dominated solutions)
    """
    history_archive = []
    population = self._init_population(search_agents_no)
    
    # Initialize archive with non-dominated solutions
    self._determine_domination(population)
    non_dominated = self._get_non_dominated_particles(population)
    self.archive.extend(non_dominated)
    
    # Build grid
    costs = self._get_fitness(self.archive)
    if costs.size > 0:
        self.grid = self._create_hypercubes(costs)
        for particle in self.archive:
            particle.grid_index, particle.grid_sub_index = self._get_grid_index(particle)
    
    self._begin_step_solver(max_iter)
    
    for iter in range(max_iter):
        # Algorithm-specific update logic
        
        self._add_to_archive(population)  # Update archive
        history_archive.append([member.copy() for member in self.archive])  # Store history
        self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
    
    self.history_step_solver = history_archive
    self.best_solver = self.archive
    self._end_step_solver()
    
    return history_archive, self.archive
```

---

## Implementation Patterns

### Pattern 1: Direct Extension (Simple)

```python
class MultiObjectiveAlgorithmNameOptimizer(MultiObjectiveSolver):
    """Multi-objective version of AlgorithmNameOptimizer"""
    
    def __init__(self, objective_func: Callable, lb, ub, dim, maximize=True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        self.name_solver = "Multi-Objective Algorithm Name Optimizer"
        self.param1 = kwargs.get('param1', default_value)
    
    def solver(self, search_agents_no, max_iter):
        # Implement using inherited utilities
        pass
```

### Pattern 2: Custom Member (Complex)

```python
class AlgorithmMultiMember(MultiObjectiveMember):
    def __init__(self, position: np.ndarray, fitness: np.ndarray, additional_attr=None):
        super().__init__(position, fitness)
        self.additional_attr = additional_attr
    
    def copy(self):
        new_member = AlgorithmMultiMember(
            self.position.copy(), self.multi_fitness.copy(), self.additional_attr
        )
        new_member.dominated = self.dominated
        new_member.grid_index = self.grid_index
        new_member.grid_sub_index = self.grid_sub_index
        return new_member

class MultiObjectiveAlgorithmNameOptimizer(MultiObjectiveSolver):
    def __init__(self, objective_func: Callable, lb, ub, dim, maximize=True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        self.name_solver = "Multi-Objective Algorithm Name Optimizer"
        self.param1 = kwargs.get('param1', default_value)
    
    def _init_population(self, search_agents_no) -> List[AlgorithmMultiMember]:
        population = []
        for _ in range(search_agents_no):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            fitness = self.objective_func(position)
            population.append(AlgorithmMultiMember(position, fitness))
        return population
    
    def solver(self, search_agents_no: int, max_iter: int):
        # Implementation with custom member class
        pass
```

---

## Key Differences vs Single-Objective

1. **Fitness**:
   * Single-objective → scalar `fitness`
   * Multi-objective → array `multi_fitness`, plus scalar `fitness` for compatibility

2. **Representation**:
   * Single-objective → best solution (`Member`)
   * Multi-objective → archive of non-dominated solutions

3. **Comparison**:
   * Single-objective → `_is_better()`
   * Multi-objective → `_dominates()` (Pareto dominance)

4. **Return Types**:
   * Single-objective → `(history, best_solution)`
   * Multi-objective → `(history_archive, final_archive)`

---

## Available Utilities from MultiObjectiveSolver

### Core Methods
- `_dominates(x, y)`: Check Pareto dominance
- `_determine_domination(population)`: Set domination status
- `_get_non_dominated_particles(population)`: Get non-dominated solutions
- `_get_fitness(population)`: Get fitness matrix
- `_add_to_archive(new_solutions)`: Update archive
- `_select_leader()`: Select leader from archive
- `_select_multiple_leaders(n_leaders)`: Select multiple leaders

### Grid-Based Selection
- `_create_hypercubes(costs)`: Create hypercubes for grid
- `_get_grid_index(particle)`: Get grid index
- `_trim_archive()`: Maintain archive size

### Population Management
- `_init_population(search_agents_no)`: Initialize population
- `_get_random_population(population, size)`: Get random sample
- `_sort_population(population)`: Sort population

### Progress Tracking
- `_begin_step_solver(max_iter)`: Start solver
- `_callbacks(iter, max_iter, best)`: Update progress
- `_end_step_solver()`: End solver
- `plot_pareto_front()`: Visualize results

---

## Auto-Detection

### Registry

```python
_MULTI_OBJECTIVE_MAPPING: Dict[str, str] = {
    "AlgorithmNameOptimizer": "MultiObjectiveAlgorithmNameOptimizer",
}
```

### Logic in `create_solver()`
* Scalar output → single-objective
* Array output → multi-objective

---

## Template

```python
import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember
from .._general import roulette_wheel_selection, normalized_values  # Import utilities as needed

class AlgorithmMultiMember(MultiObjectiveMember):
    def __init__(self, position: np.ndarray, fitness: np.ndarray, additional_attr=None):
        super().__init__(position, fitness)
        self.additional_attr = additional_attr
    
    def copy(self):
        new_member = AlgorithmMultiMember(self.position.copy(), self.multi_fitness.copy(), self.additional_attr)
        new_member.dominated = self.dominated
        new_member.grid_index = self.grid_index
        new_member.grid_sub_index = self.grid_sub_index
        return new_member

class MultiObjectiveAlgorithmNameOptimizer(MultiObjectiveSolver):
    def __init__(self, objective_func: Callable, lb, ub, dim, maximize=True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        self.name_solver = "Multi-Objective Algorithm Name Optimizer"
        # Copy algorithm-specific parameters
        self.param1 = kwargs.get('param1', default_value)
    
    def _init_population(self, search_agents_no) -> List[AlgorithmMultiMember]:
        population = []
        for _ in range(search_agents_no):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            fitness = self.objective_func(position)
            population.append(AlgorithmMultiMember(position, fitness))
        return population
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[AlgorithmMultiMember]]:
        # Algorithm-specific implementation here
        # Use inherited utilities for archive management
        pass
```

---

## Testing

```python
def test_multiobjective_algorithm():
    method = create_solver(
        solver_name="AlgorithmNameOptimizer",
        objective_func=zdt1_function,  # Multi-objective function
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
    
    assert len(final_archive) > 0
    assert len(final_archive[0].multi_fitness) == 2
```

---

## Best Practices

1. **Reuse logic** from the single-objective version whenever possible
2. Always use **Pareto dominance** (`_dominates`) for comparisons
3. Manage archive carefully with built-in utilities
4. Use **grid-based leader selection** for diversity
5. Watch **memory usage** when archive size is large
6. **Import utilities** from `.._general` as needed (roulette_wheel_selection, normalized_values, etc.)

---

## Common Patterns

### Leader Guidance

```python
leader = self._select_leader() or np.random.choice(population)
new_position = current.position + phi * (leader.position - current.position)
```

### Archive-Based Selection

```python
if self.archive:
    parent1, parent2 = np.random.choice(self.archive, 2, replace=False)
    new_position = crossover(parent1.position, parent2.position)
```

### Adaptive Parameters

```python
a = 2 - iter * (2 / max_iter)  # Example (GWO)
```

---

## Error Handling

* Handle **empty archives**
* Validate **bounds** and **objective outputs**
* Ensure grid initialization doesn't fail
* Use try-catch for objective function evaluation

---

## Performance Tips

* Prefer **vectorized operations**
* Avoid excessive copying
* Optimize archive pruning
* Limit archive size for efficiency
* Use inherited utilities instead of reimplementing

---

## Implementation Examples

Check existing files for reference:
- `multiobjective/artificialbeecolony_optimizer.py` - Complex algorithm with custom member
- `multiobjective/greywolf_optimizer.py` - Simple algorithm pattern
- `multiobjective/particleswarm_optimizer.py` - Custom member with additional attributes

*Last updated: 2025-08-26*
