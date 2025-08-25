# Automatic Algorithm Generation System

## Overview

This document describes the automatic generation system that transforms new single-objective optimization algorithms into their multi-objective counterparts, following the unified coding standards of LibraryOptimizer.

## System Architecture

### Automatic Generation Pipeline

```
Single-Objective Algorithm → Analysis → Multi-Objective Generation → Registry Update
```

### Key Components

1. **Algorithm Analyzer**: Examines single-objective algorithm structure
2. **Template Generator**: Creates multi-objective version templates
3. **Registry Manager**: Updates solver registry automatically
4. **Code Validator**: Ensures generated code follows library standards

## Automatic Generation Process

### Step 1: Single-Objective Algorithm Creation
Create your algorithm following the standard pattern:

```python
class YourAlgorithmOptimizer(Solver):
    def __init__(self, objective_func, lb, ub, dim, maximize=True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        self.name_solver = "Your Algorithm Optimizer"
        # Algorithm-specific parameters
        self.param1 = kwargs.get('param1', default_value)
    
    def solver(self, search_agents_no, max_iter):
        # Your algorithm implementation
        pass
```

### Step 2: Automatic Multi-Objective Generation
The system automatically generates:

```python
# File: src/multiobjective/your_algorithm_optimizer.py
class MultiObjectiveYourAlgorithmOptimizer(MultiObjectiveSolver):
    def __init__(self, objective_func, lb, ub, dim, maximize=True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        self.name_solver = "Multi-Objective Your Algorithm Optimizer"
        # Copy all algorithm-specific parameters
        self.param1 = kwargs.get('param1', default_value)
    
    def solver(self, search_agents_no, max_iter):
        # Multi-objective implementation using inherited utilities
        pass
```

### Step 3: Automatic Registry Updates
The system automatically updates:

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
All algorithm-specific parameters are automatically copied to the multi-objective version:

```python
# Single-objective parameters
self.population_size = kwargs.get('population_size', 50)
self.crossover_rate = kwargs.get('crossover_rate', 0.8)
self.mutation_rate = kwargs.get('mutation_rate', 0.1)

# Automatically available in multi-objective version
```

### 2. Objective Function Detection
The system automatically detects function type:

```python
# Returns scalar → uses single-objective
def sphere(x): return np.sum(x**2)

# Returns array → uses multi-objective  
def zdt1(x): return [x[0], 1 - np.sqrt(x[0])]

# Automatic selection:
method = create_solver('YourAlgorithm', objective_func, lb, ub, dim)
```

### 3. Utility Integration
Multi-objective versions automatically inherit:

- Archive management utilities
- Grid-based selection
- Pareto dominance checking
- Progress tracking
- Visualization tools

## Generation Templates

### Template 1: Simple Algorithm (No Custom Member)
```python
class MultiObjectiveAlgorithmNameOptimizer(MultiObjectiveSolver):
    def __init__(self, objective_func, lb, ub, dim, maximize=True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        self.name_solver = "Multi-Objective Algorithm Name Optimizer"
        # Copy parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def solver(self, search_agents_no, max_iter):
        # Focus on algorithm-specific update logic
        # Use inherited multi-objective utilities
        pass
```

### Template 2: Complex Algorithm (With Custom Member)
```python
class AlgorithmMultiMember(MultiObjectiveMember):
    def __init__(self, position, fitness, additional_attr=None):
        super().__init__(position, fitness)
        self.additional_attr = additional_attr # Please replace with desired properties
    
    def copy(self):
        return AlgorithmMultiMember(self.position.copy(), 
                                  self.multi_fitness.copy(),
                                  self.additional_attr)

class MultiObjectiveAlgorithmNameOptimizer(MultiObjectiveSolver):
    def __init__(self, objective_func, lb, ub, dim, maximize, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        self.name_solver = "Multi-Objective Algorithm Name Optimizer"
        # Copy parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def _init_population(self, search_agents_no):
        population = []
        for _ in range(search_agents_no):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            fitness = self.objective_func(position)
            population.append(AlgorithmMultiMember(position, fitness))
        return population
    
    def solver(self, search_agents_no, max_iter):
        # Implementation with custom member class
        pass
```

## Implementation Patterns

### Pattern A: Leader-Based Algorithms
```python
def solver(self, search_agents_no, max_iter):
    # Initialize population and archive
    population = self._init_population(search_agents_no)
    self._add_to_archive(population)
    
    for iter in range(max_iter):
        for i, agent in enumerate(population):
            # Select leader from archive
            leader = self._select_leader()
            
            # Algorithm-specific update using leader
            new_position = self._update_position(agent, leader, iter, max_iter)
            
            # Update agent
            population[i].position = new_position
            population[i].multi_fitness = self.objective_func(new_position)
        
        # Update archive
        self._add_to_archive(population)
        
        # Track progress
        self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
```

### Pattern B: Population-Based Algorithms
```python
def solver(self, search_agents_no, max_iter):
    # Initialize population and archive
    population = self._init_population(search_agents_no)
    self._add_to_archive(population)
    
    for iter in range(max_iter):
        # Algorithm-specific population update
        new_population = self._update_population(population, iter, max_iter)
        
        # Evaluate new population
        for agent in new_population:
            agent.multi_fitness = self.objective_func(agent.position)
        
        # Update archive
        self._add_to_archive(new_population)
        population = new_population
        
        # Track progress
        self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
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
            # Multi-objective detected
            return multi_objective_version(objective_func, lb, ub, dim, **kwargs)
        else:
            # Single-objective detected
            return single_objective_version(objective_func, lb, ub, dim, **kwargs)
            
    except Exception as e:
        print(f"Auto-detection failed: {e}")
        print("Falling back to single-objective version")
        return single_objective_version(objective_func, lb, ub, dim, **kwargs)
```

## Performance Optimization

### Memory Efficiency
```python
def _trim_archive(self):
    """Efficient archive trimming using grid-based selection"""
    while len(self.archive) > self.archive_size:
        # Grid-based removal for diversity preservation
        self._remove_least_diverse_solution()
```

### Vectorized Operations
```python
def _update_positions(self, population, leaders):
    """Vectorized position update"""
    # Use NumPy vectorization instead of loops
    positions = np.array([agent.position for agent in population])
    leader_positions = np.array([leader.position for leader in leaders])
    
    # Vectorized update
    new_positions = positions + self.step_size * (leader_positions - positions)
    
    # Update agents
    for i, agent in enumerate(population):
        agent.position = new_positions[i]
```


## Best Practices for Automatic Generation

### 1. Code Consistency
- Follow existing naming conventions
- Use consistent parameter naming
- Maintain same code style as library

### 2. Documentation
- Auto-generate comprehensive docstrings
- Include examples in documentation
- Document all parameters and returns

### 3. Error Handling
- Provide meaningful error messages
- Include fallback mechanisms
- Validate all inputs

## Integration with Existing System

### Registry Auto-Update
The system automatically handles:
- Import statement generation
- Registry mapping updates
- Multi-objective mapping creation

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

3. **Performance Problems**
   - Use vectorized operations
   - Optimize archive maintenance
   - Consider archive size limits

### Debug Mode
```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)

# Create solver with debug info
method = create_solver('YourAlgorithmOptimizer', objective_func, lb, ub, dim, maximize)
```

## Future Enhancements

### Planned Features
- Automatic hyperparameter tuning
- Adaptive archive size management
- Advanced visualization tools
- Parallel execution support
- Cloud deployment integration

### Extension Points
- Custom dominance criteria
- Alternative archive strategies
- Specialized grid implementations
- Hybrid algorithm support

*This document was last updated: 2025-08-25*
