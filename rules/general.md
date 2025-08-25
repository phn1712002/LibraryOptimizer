# Library Optimizer - Coding Standards

## 📋 Overview

This document defines programming standards for the Library Optimizer, including naming conventions, commenting standards, function structure, file organization, and general rules. The system now supports automatic generation of multi-objective versions from single-objective algorithms, following these unified standards.
## 🏷️ Naming Rules

### 1. Class Names
- **PascalCase**: Capitalize the first letter of each word
- **Examples**: `GreyWolfOptimizer`, `ParticleSwarmOptimizer`, `ArtificialBeeColonyOptimizer`

### 2. Function/Method Names
- **snake_case**: Lowercase with underscores separating words
- **Examples**: `create_solver`, `find_solver`, `register_solver`, `_init_population`

### 3. Variable Names
- **snake_case**: Lowercase with underscores separating words
- **Examples**: `objective_func`, `search_agents_no`, `max_iter`, `best_solver`

### 4. Constants
- **UPPER_SNAKE_CASE**: All uppercase with underscores separating words
- **Examples**: `MAX_ITERATIONS`, `DEFAULT_POPULATION_SIZE`

### 5. Private/Protected Variables
- **Start with underscore**: For internal components
- **Examples**: `_SOLVER_REGISTRY`, `_is_better`, `_callbacks`

## 💬 Commenting Rules

### 1. Docstrings
```python
def create_solver(
    solver_name: str, 
    objective_func: Callable, 
    lb: Union[float, np.ndarray], 
    ub: Union[float, np.ndarray], 
    dim: int, 
    maximize: bool = True,
    **kwargs
) -> Solver:
    """
    Create a solver instance based on the specified name.
    
    Args:
        solver_name: Name of the solver (e.g., 'GreyWolfOptimizer')
        objective_func: Objective function to optimize
        lb: Lower bound
        ub: Upper bound
        dim: Problem dimension
        maximize: True for maximization, False for minimization
        **kwargs: Additional parameters for specific solver
        
    Returns:
        Instance of Solver class
        
    Raises:
        ValueError: If solver_name doesn't exist
    """
```

### 2. Inline Comments
```python
# Update a parameter (decreases linearly from 2 to 0)
a = 2 - iter * (2 / max_iter)

# Ensure positions stay within bounds
new_position = np.clip(new_position, self.lb, self.ub)
```

### 3. TODO Comments
```python
# TODO: Implement adaptive parameter tuning
# FIXME: Handle edge case when population is empty
```

## 📝 Function Writing Rules

### 1. Function Signature
```python
def function_name(
    param1: Type, 
    param2: Type = default_value,
    *args,
    **kwargs
) -> ReturnType:
    """Docstring describing the function."""
```

### 2. Type Hints
- **Required**: Use type hints for all public functions
- **Example**: 
```python
def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
```

### 3. Function Length
- **Maximum 50 lines**: Each function should perform a specific task
- **Split large functions**: If function is too long, break into smaller functions

### 4. Return Values
- **Clear**: Always return meaningful values
- **Consistent**: Same return type for same functionality

## 📁 File Organization Rules

### 1. Directory Structure
```
src/
├── __init__.py          # Module exports and registry
├── _core.py              # Base class and common utilities
├── algorithm1_optimizer.py  # Specific algorithm implementation
├── algorithm2_optimizer.py
└── ...

utils/
├── __init__.py
├── general.py           # Common utilities
└── ...

rules/
├── general.md           # General rules (this document)
└── new-algorithm.md     # Template for new algorithms

docs/                    # Documentation
tests/                   # Unit tests
```

### 2. Import Rules
```python
# Standard library imports
import numpy as np
from typing import Callable, Union, Tuple, List

# Third-party imports
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local imports (relative)
from ._core import Solver, Member
# Note: Use inherited utilities from Solver class instead of importing from utils.general
# Only import specific utilities that are not available through inheritance
```

### 3. File Naming
- **snake_case**: For all Python files
- **Descriptive**: File names should describe content
- **Examples**: `greywolf_optimizer.py`, `particleswarm_optimizer.py`

## 🎯 Specific Rules for Optimization Library

### 1. Interface Consistency
All optimizers must inherit from `Solver` and implement:
```python
class CustomOptimizer(Solver):
    def __init__(self, objective_func, lb, ub, dim, maximize=True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize)
        self.name_solver = "Custom Optimizer Name"
        # Store additional parameters
        self.kwargs = kwargs
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        # Implementation here
        return history_step_solver, best_solver
```

### 2. Error Handling
```python
def find_solver(solver_name: str) -> Type[Solver]:
    solver_name_lower = solver_name.strip().lower()
    if solver_name_lower not in _SOLVER_REGISTRY:
        available_solvers = list(_SOLVER_REGISTRY.keys())
        raise ValueError(
            f"Solver '{solver_name}' not found. "
            f"Available solvers: {available_solvers}"
        )
    return _SOLVER_REGISTRY[solver_name_lower]
```

### 3. Performance Considerations
- Use NumPy vectorization instead of Python loops
- Avoid unnecessary copying of large arrays
- Use pre-allocation when possible

### 4. Testing Requirements
Each optimizer must have:
- Unit tests for main functions
- Integration tests with different objective functions
- Performance benchmarks

## 🔧 Code Style and Formatting

### 1. Indentation
- **4 spaces**: No tabs
- **Consistent**: Ensure consistent indentation

### 2. Line Length
- **Maximum 88 characters**: Following PEP 8
- **Wrap appropriately**: Use parentheses for line wrapping

### 3. Whitespace
- **One space** around operators
- **No space** inside parentheses
- **Examples**:
```python
# Good
result = a + b * c
function_call(arg1, arg2)

# Bad
result=a+b*c
function_call( arg1, arg2 )
```

### 4. Import Order
1. Standard library imports
2. Third-party imports  
3. Local application imports
4. Relative imports

## 🚀 Best Practices

### 1. Code Reusability
- Use inheritance from base `Solver` class
- Reuse utilities from inherited methods instead of importing from utils.general
- Avoid code duplication

### 2. Maintainability
- Write self-documenting code
- Use constants for magic numbers
- Keep functions short and focused

### 3. Extensibility
- Design for easy extension
- Use registry pattern for new optimizers
- Support parameters through `**kwargs`

### 4. Documentation
- Provide complete docstrings
- Include examples in docstrings
- Update README with new features

## 📋 Template for New Algorithms

See `rules/new-algorithm.md` for detailed template when adding new algorithms and `rules/auto-generation-multi.md` for automatic multi-objective generation system.

## 🔍 Code Review Checklist

- [ ] Follow naming conventions
- [ ] Complete docstrings
- [ ] Type hints for all public functions
- [ ] No magic numbers
- [ ] Proper error handling
- [ ] Unit tests written
- [ ] Performance considered
- [ ] Code is readable and maintainable

## 📞 Contact

For questions about these rules, contact:
- Author: HoangggNam
- Email: phn1712002@gmai.com

---

*This document was last updated: 2025-08-23*
