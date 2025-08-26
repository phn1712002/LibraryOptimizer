# Automatic Algorithm Detection System

## Overview

This document describes the automatic detection system that chooses between single-objective and multi-objective versions of optimization algorithms based on objective function output, following the unified coding standards of LibraryOptimizer.

## System Architecture

### Automatic Detection Pipeline

```
Objective Function → Type Detection → Solver Selection → Instance Creation
```

### Key Components

1. **Function Analyzer**: Examines objective function output type
2. **Solver Selector**: Chooses appropriate solver version
3. **Registry Mapper**: Maps single-objective to multi-objective solvers
4. **Instance Creator**: Creates solver instances with proper parameters

## Automatic Detection Process

### Step 1: Single-Objective Algorithm Creation
Create your algorithm following the standard pattern:

```python
class YourAlgorithmOptimizer(Solver):
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        self.name_solver = "Your Algorithm Optimizer"
        # Algorithm-specific parameters
        self.param1 = kwargs.get('param1', default_value)
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        # Your algorithm implementation
        pass
```

### Step 2: Manual Multi-Objective Implementation
Manually create the multi-objective version following the multi-objective standards:

```python
# File: src/multiobjective/your_algorithm_optimizer.py
class MultiObjectiveYourAlgorithmOptimizer(MultiObjectiveSolver):
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        self.name_solver = "Multi-Objective Your Algorithm Optimizer"
        # Copy all algorithm-specific parameters
        self.param1 = kwargs.get('param1', default_value)
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        # Multi-objective implementation using inherited utilities
        pass
```

### Step 3: Manual Registry Updates
Manually update the registry mappings:

```python
# In src/__init__.py
_MULTI_OBJECTIVE_MAPPING: Dict[str, str] = {
    "YourAlgorithmOptimizer": "MultiObjectiveYourAlgorithmOptimizer",
    # ... other mappings
}

_SOLVER_REGISTRY: Dict[str, Type[Solver]] = {
    "YourAlgorithmOptimizer": YourAlgorithmOptimizer,
    "MultiObjectiveYourAlgorithmOptimizer": MultiObjectiveYourAlgorithmOptimizer,
    # ... other solvers
}
```

## Automatic Features

### 1. Parameter Inheritance
All algorithm-specific parameters are automatically passed to the appropriate version:

```python
# Parameters work for both single and multi-objective
method = create_solver(
    'YourAlgorithmOptimizer',
    objective_func,
    lb, ub, dim,
    param1=0.5,        # Algorithm-specific
    archive_size=100   # Multi-objective specific
)
```

### 2. Objective Function Detection
The system automatically detects function type:

```python
# Returns scalar → uses single-objective
def sphere(x): return np.sum(x**2)

# Returns array → uses multi-objective  
def zdt1(x): return [x[0], 1 - np.sqrt(x[0])]

# Automatic selection:
method = create_solver('YourAlgorithmOptimizer', objective_func, lb, ub, dim)
```

### 3. Utility Integration
Multi-objective versions automatically inherit:

- Archive management utilities
- Grid-based selection
- Pareto dominance checking
- Progress tracking
- Visualization tools

## Implementation Guide

### For New Algorithms

1. **Create Single-Objective Version**: Follow single-objective rules
2. **Create Multi-Objective Version**: Manually implement following multi-objective standards
3. **Update Registry**: Add mappings in `src/__init__.py`
4. **Test**: Verify both versions work correctly

### Parameter Handling
All algorithm-specific parameters should be passed through `**kwargs` and extracted in the constructor:

```python
def __init__(self, objective_func, lb, ub, dim, maximize=True, **kwargs):
    super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
    self.param1 = kwargs.get('param1', default_value)
    self.param2 = kwargs.get('param2', default_value)
```

## Error Handling and Fallbacks

### Graceful Error Handling
```python
def create_solver(solver_name, objective_func, lb, ub, dim, **kwargs):
    try:
        # Auto-detection logic
        test_point = np.random.uniform(lb, ub, dim)
        result = objective_func(test_point)
        
        if hasattr(result, '__len__') and len(result) > 1:
            # Multi-objective function detected
            return multi_objective_version(objective_func, lb, ub, dim, maximize, **kwargs)
        else:
            # Single-objective function detected
            return single_objective_version(objective_func, lb, ub, dim, maximize, **kwargs)
            
    except Exception as e:
        print(f"Auto-detection failed: {e}")
        print("Falling back to single-objective version")
        return single_objective_version(objective_func, lb, ub, dim, maximize, **kwargs)
```

## Best Practices

### 1. Code Consistency
- Follow existing naming conventions
- Use consistent parameter naming
- Maintain same code style as library

### 2. Documentation
- Comprehensive docstrings for classes and methods
- Include examples in documentation
- Document all parameters and returns

### 3. Error Handling
- Provide meaningful error messages
- Include fallback mechanisms
- Validate all inputs

## Integration with Existing System

### Registry Management
The system uses manual registry updates:
- Import statements must be added manually
- Registry mappings must be updated manually
- Multi-objective mapping must be created manually

### Backward Compatibility
- All existing single-objective algorithms remain unchanged
- Multi-objective versions are opt-in
- Fallback to single-objective if multi-objective not available

## Usage Examples

### Example 1: Basic Usage
```python
# The system automatically chooses the right version
method = create_solver(
    'YourAlgorithmOptimizer',
    objective_func,  # Auto-detected as single or multi-objective
    lb, ub, dim, maximize
)
```

### Example 2: Explicit Multi-Objective
```python
# Force multi-objective version
method = create_solver(
    'MultiObjectiveYourAlgorithmOptimizer',
    multi_obj_function,
    lb, ub, dim, maximize
    archive_size=100,
)
```

### Example 3: Custom Parameters
```python
# Pass algorithm-specific parameters
method = create_solver(
    'YourAlgorithmOptimizer',
    objective_func,
    lb, ub, dim, maximize
    param1=0.5,
    param2=0.3,
    archive_size=150  # Multi-objective specific
)
```

## Troubleshooting

### Common Issues

1. **Objective Function Detection Failure**
   - Ensure function returns consistent output types
   - Handle edge cases in function implementation

2. **Parameter Inheritance Issues**
   - Check that all parameters are passed through kwargs
   - Validate parameter types and ranges

3. **Registry Mapping Errors**
   - Verify both single and multi-objective versions are imported
   - Check mapping entries in `_MULTI_OBJECTIVE_MAPPING`

### Debug Mode
```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)

# Create solver with debug info
method = create_solver('YourAlgorithmOptimizer', objective_func, lb, ub, dim, maximize)
```

## Current Implementation Status

The automatic generation system currently provides:
- ✅ Automatic objective function type detection
- ✅ Automatic solver version selection
- ✅ Parameter inheritance through kwargs
- ✅ Graceful fallback to single-objective
- ✅ Multi-objective utility integration

Manual steps required:
- ✗ Multi-objective algorithm implementation
- ✗ Registry import statements
- ✗ Multi-objective mapping entries
- ✗ Custom member class implementation (if needed)

## Future Enhancements

### Planned Features
- Automated multi-objective template generation
- Registry auto-update scripts
- Parameter validation system
- Advanced visualization tools
- Parallel execution support

### Extension Points
- Custom dominance criteria
- Alternative archive strategies
- Specialized grid implementations
- Hybrid algorithm support

*This document was last updated: 2025-08-26*
