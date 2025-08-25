# Rules for Multi-Objective Optimization Algorithms

## Overview

This document defines the standards and patterns for creating multi-objective versions of single-objective optimization algorithms in LibraryOptimizer. The system now supports automatic generation of multi-objective algorithms when new single-objective algorithms are added, following the unified coding standards of the library.

## Directory Structure

```
src/
├── algorithm_name_optimizer.py          # Single-objective version
└── multiobjective/
    ├── _core.py                         # Multi-objective base classes
    └── algorithm_name_optimizer.py      # Multi-objective version
```

## Naming Conventions

### File Names
- **Single-objective**: `algorithm_name_optimizer.py` (snake_case)
- **Multi-objective**: `algorithm_name_optimizer.py` (same name, different directory)

### Class Names
- **Single-objective**: `AlgorithmNameOptimizer` (PascalCase)
- **Multi-objective**: `MultiObjectiveAlgorithmNameOptimizer` (PascalCase with "MultiObjective" prefix)

### Member Classes
- **Single-objective**: `AlgorithmMember` (extends `Member`)
- **Multi-objective**: `AlgorithmMultiMember` (extends `MultiObjectiveMember`)

## Base Classes

### MultiObjectiveMember
```python
class MultiObjectiveMember(Member):
    def __init__(self, position: np.ndarray, fitness: np.ndarray):
        super().__init__(position, 0)  # Single fitness for compatibility
        self.multi_fitness = np.array(fitness)  # All fitness values
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
                 ub: Union[float, np.ndarray], dim: int, **kwargs):
        super().__init__(objective_func, lb, ub, dim, True)  # maximize=True for compatibility
        
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

## Required Methods for Multi-Objective Algorithms

### 1. Constructor
```python
def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
             ub: Union[float, np.ndarray], dim: int, **kwargs):
    super().__init__(objective_func, lb, ub, dim, **kwargs)
    
    # Set solver name with "Multi-Objective" prefix
    self.name_solver = "Multi-Objective Algorithm Name Optimizer"
    
    # Store algorithm-specific parameters
    self.param1 = kwargs.get('param1', default_value)
    # ... other parameters
```

### 2. Main Solver Method
```python
def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[AlgorithmMultiMember]]:
    """
    Main optimization method for multi-objective version
    
    Returns:
    --------
    Tuple[List, List[AlgorithmMultiMember]]
        History of archive states and the final archive
    """
    # Initialize storage
    history_archive = []
    
    # Initialize population
    population = self._init_population(search_agents_no)
    
    # Initialize archive with non-dominated solutions
    self._determine_domination(population)
    non_dominated = self._get_non_dominated_particles(population)
    self.archive.extend(non_dominated)
    
    # Initialize grid
    costs = self._get_costs(self.archive)
    if costs.size > 0:
        self.grid = self._create_hypercubes(costs)
        for particle in self.archive:
            particle.grid_index, particle.grid_sub_index = self._get_grid_index(particle)
    
    # Start solver
    self._begin_step_solver(max_iter)
    
    # Main optimization loop
    for iter in range(max_iter):
        # Algorithm-specific update logic here
        
        # Update archive with current population
        self._add_to_archive(population)
        
        # Store archive state for history
        history_archive.append([bee.copy() for bee in self.archive])
        
        # Update progress
        self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
    
    # Final processing
    self.history_step_solver = history_archive
    self.best_solver = self.archive
    
    # End solver
    self._end_step_solver()
    
    return history_archive, self.archive
```

## Automatic Generation Patterns

### Pattern 1: Direct Extension (Simple Algorithms)
For algorithms that don't require complex state management:
```python
class MultiObjectiveAlgorithmNameOptimizer(MultiObjectiveSolver):
    """Multi-objective version of AlgorithmNameOptimizer"""
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, **kwargs):
        super().__init__(objective_func, lb, ub, dim, **kwargs)
        self.name_solver = "Multi-Objective Algorithm Name Optimizer"
        
        # Copy parameters from single-objective version
        self.param1 = kwargs.get('param1', default_value)
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[AlgorithmMultiMember]]:
        # Implementation using inherited multi-objective utilities
        # Focus on the algorithm-specific update logic
```

### Pattern 2: Custom Member Class (Complex Algorithms)
For algorithms that need additional state:
```python
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
```

## Key Differences from Single-Objective

### 1. Fitness Handling
- **Single-objective**: `fitness` (scalar)
- **Multi-objective**: `multi_fitness` (array), `fitness` (compatibility scalar)

### 2. Solution Representation
- **Single-objective**: Best solution (`Member`)
- **Multi-objective**: Archive of non-dominated solutions (`List[MultiObjectiveMember]`)

### 3. Comparison
- **Single-objective**: `_is_better()` (scalar comparison)
- **Multi-objective**: `_dominates()` (Pareto dominance)

### 4. Return Types
- **Single-objective**: `Tuple[List[Member], Member]`
- **Multi-objective**: `Tuple[List[List[MultiObjectiveMember]], List[MultiObjectiveMember]]`

## Registry and Auto-Detection

### 1. Registry Mapping
In `src/__init__.py`, add mapping:
```python
_MULTI_OBJECTIVE_MAPPING: Dict[str, str] = {
    "AlgorithmNameOptimizer": "MultiObjectiveAlgorithmNameOptimizer",
    # ... other mappings
}
```

### 2. Auto-Detection Logic
The `create_solver()` function automatically detects objective function type:
- Scalar output → Single-objective version
- Array output → Multi-objective version

## Template for New Multi-Objective Algorithms

```python
import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember
from utils.general import roulette_wheel_selection

class AlgorithmMultiMember(MultiObjectiveMember):
    def __init__(self, position: np.ndarray, fitness: np.ndarray, additional_attr=None):
        super().__init__(position, fitness)
        self.additional_attr = additional_attr # Please replace with desired properties
    
    def copy(self):
        new_member = AlgorithmMultiMember(self.position.copy(), self.multi_fitness.copy(), 
                                        self.additional_attr)
        new_member.dominated = self.dominated
        new_member.grid_index = self.grid_index
        new_member.grid_sub_index = self.grid_sub_index
        return new_member

class MultiObjectiveAlgorithmNameOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Algorithm Name Optimizer
    
    Parameters:
    -----------
    objective_func : Callable
        Multi-objective function that returns array of fitness values
    lb : Union[float, np.ndarray]
        Lower bounds
    ub : Union[float, np.ndarray]
        Upper bounds
    dim : int
        Problem dimension
    **kwargs
        Additional parameters:
        - archive_size: Size of external archive (default: 100)
        - param1: Algorithm-specific parameter
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool=True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        self.name_solver = "Multi-Objective Algorithm Name Optimizer"
        
        # Algorithm-specific parameters
        self.param1 = kwargs.get('param1', default_value)
    
    def _init_population(self, search_agents_no) -> List[AlgorithmMultiMember]:
        population = []
        for _ in range(search_agents_no):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            fitness = self.objective_func(position)
            population.append(AlgorithmMultiMember(position, fitness))
        return population
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[AlgorithmMultiMember]]:
        # Implementation using algorithm-specific logic
        # and inherited multi-objective utilities
        pass
```

## Testing Requirements

### Multi-Objective Test Functions
```python
def test_multiobjective_algorithm():
    method = create_solver(
        solver_name='AlgorithmNameOptimizer',
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
    
    # Should find diverse non-dominated solutions
    assert len(final_archive) > 0
    assert len(final_archive[0].multi_fitness) == 2
```

## Best Practices

1. **Reuse Logic**: Where possible, reuse the core algorithm logic from single-objective version
2. **Pareto Dominance**: Always use `_dominates()` for comparison, not `_is_better()`
3. **Archive Management**: Use inherited archive management utilities
4. **Grid-Based Selection**: Leverage grid-based selection for leader selection
5. **Memory Efficiency**: Be mindful of archive size and memory usage

## Common Implementation Patterns

### Pattern A: Leader Guidance
```python
# Select leader from archive
leader = self._select_leader()
if leader is None:
    leader = np.random.choice(population)

# Use leader to guide search
new_position = current.position + phi * (leader.position - current.position)
```

### Pattern B: Archive-Based Selection
```python
# Use archive members for crossover/mutation
if self.archive:
    parent1 = np.random.choice(self.archive)
    parent2 = np.random.choice(self.archive)
    new_position = crossover(parent1.position, parent2.position)
```

### Pattern C: Adaptive Parameters
```python
# Adapt parameters based on iteration
a = 2 - iter * (2 / max_iter)  # Example from GWO
```

## Error Handling

- Handle empty archive cases gracefully
- Ensure proper bounds checking
- Validate multi-objective function outputs
- Handle grid initialization failures

## Performance Considerations

- Use vectorized operations where possible
- Avoid unnecessary copying of large arrays
- Optimize archive maintenance operations
- Consider archive size limits for memory efficiency

*This document was last updated: 2025-08-25*
