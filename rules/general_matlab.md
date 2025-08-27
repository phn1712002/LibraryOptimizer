# 📚 GUIDE TO ADDING NEW OPTIMIZATION ALGORITHMS IN MATLAB

## 📋 Overview

This document provides instructions on how to implement and add a new optimization algorithm into the MATLAB library.  
The current project folder structure is:

```

matlab/
├── core/           # Base classes (read here first – foundation for new algorithms)
├── src/            # Specific algorithms (all new algorithms go here)
├── strucs/         # Data structures (place derived Member/MultiObjectiveMember classes here if needed)
├── utils/          # Utilities
└── main.m          # Main test script

````

## 🏗️ Base Class Structure

### 1. Core Classes

- **`Solver.m`**: Base class for single-objective algorithms  
- **`MultiObjectiveSolver.m`**: Base class for multi-objective algorithms  
- **`Member.m`**: Represents an individual in the population (single-objective)  
- **`MultiObjectiveMember.m`**: Represents an individual in the population (multi-objective)  

### 2. Inheritance Examples

```matlab
% Single-objective
classdef AlgorithmName < Solver
    % Implementation
end

% Multi-objective
classdef AlgorithmNameMO < MultiObjectiveSolver
    % Implementation
end
````

## 📝 Steps to Create a New Algorithm

### Step 1: Create a new file in `src/`

```matlab
% File: matlab/src/algorithm_name.m
classdef AlgorithmName < Solver
    %{
    AlgorithmName - Short description of the algorithm
    
    Parameters:
    ------------
    objective_func : function handle
        Objective function to optimize
    lb : float or array
        Lower bound of the search space
    ub : float or array
        Upper bound of the search space
    dim : int
        Number of problem dimensions
    maximize : bool
        Optimization direction (true: maximize, false: minimize)
    varargin : cell array
        Additional algorithm parameters
    %}
    
    methods
        function obj = AlgorithmName(objective_func, lb, ub, dim, maximize, varargin)
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set algorithm name
            obj.name_solver = "Algorithm Name";
            
            % Parse algorithm-specific parameters
            p = inputParser;
            addParameter(p, 'param_1', default_value, @data_type);
            addParameter(p, 'param_2', default_value, @data_type);
            parse(p, varargin{:});
            
            obj.param_1 = p.Results.param_1;
            obj.param_2 = p.Results.param_2;
        end
    end
end
```

### Step 2: Implement the `solver` method

```matlab
function [history, best] = solver(obj, search_agents_no, max_iter)
    %{
    solver - Main optimization procedure
    
    Inputs:
        search_agents_no : int
            Population size
        max_iter : int
            Maximum number of iterations
            
    Returns:
        history : cell array
            Best solutions found across iterations
        best : Member
            Best final solution
    %}
    
    % Initialize population
    population = obj.init_population(search_agents_no);
    
    % Start algorithm
    obj.begin_step_solver(max_iter);
    
    % Main loop
    for iter = 1:max_iter
        % Update control parameters (example)
        a = 2 - iter * (2 / max_iter);
        
        % Update each individual
        for i = 1:length(population)
            % Example update logic
            % new_position = ...
            
            % Enforce bounds
            new_position = max(min(new_position, obj.ub), obj.lb);
            
            % Update position & fitness
            population{i}.position = new_position;
            population{i}.fitness = obj.objective_func(new_position);
        end
        
        % Select best individual
        fitness_values = obj.get_fitness(population);
        if obj.maximize
            [best_fitness, best_idx] = max(fitness_values);
        else
            [best_fitness, best_idx] = min(fitness_values);
        end
        
        % Update history
        obj.history_step_solver{end+1} = population{best_idx}.copy();
        
        % Progress callback
        obj.callbacks(iter, max_iter, population{best_idx});
    end
    
    % End algorithm
    obj.end_step_solver();
    
    % Return results
    history = obj.history_step_solver;
    best = obj.best_solver;
end
```

### Step 3: Multi-objective Solver

```matlab
classdef AlgorithmNameMO < MultiObjectiveSolver
    methods
        function obj = AlgorithmNameMO(objective_func, lb, ub, dim, maximize, varargin)
            obj@MultiObjectiveSolver(objective_func, lb, ub, dim, maximize, varargin{:});
            obj.name_solver = "Multi-Objective Algorithm Name";
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            % Multi-objective logic similar to MultiObjectiveWhaleOptimizer
            % Use parent methods:
            % - obj.determine_domination()
            % - obj.get_non_dominated_particles()
            % - obj.add_to_archive()
            % - obj.select_leader()
        end
    end
end
```

## 🎯 Naming Rules

### 1. Class Names

* **PascalCase**: `GreyWolfOptimizer`, `ParticleSwarmOptimizer`

### 2. Method & Variable Names

* **snake\_case**: `objective_func`, `search_agents_no`, `init_population`

### 3. Constants

* **UPPER\_SNAKE\_CASE**: `MAX_ITERATIONS`, `DEFAULT_POPULATION_SIZE`

## 📊 Base Class Methods

### From `Solver`:

* `init_population()`: Initialize random population
* `get_positions()`: Extract positions from population
* `get_fitness()`: Extract fitness values
* `is_better()`: Compare two individuals
* `begin_step_solver()`: Display starting info
* `end_step_solver()`: Display final results
* `plot_history_step_solver()`: Plot convergence curve

### From `MultiObjectiveSolver`:

* `dominates()`: Check domination
* `determine_domination()`: Determine domination in population
* `get_non_dominated_particles()`: Extract non-dominated solutions
* `create_hypercubes()`: Build grid for archive
* `select_leader()`: Select a leader solution
* `add_to_archive()`: Add solution to archive
* `plot_pareto_front()`: Plot Pareto front

## 🔧 Template Example

```matlab
classdef TemplateOptimizer < Solver
    properties
        param_1
        param_2
    end
    
    methods
        function obj = TemplateOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            obj.name_solver = "Template Optimizer";
            
            % Parse custom parameters
            p = inputParser;
            addParameter(p, 'param_1', 0.5, @isnumeric);
            addParameter(p, 'param_2', 1.0, @isnumeric);
            parse(p, varargin{:});
            
            obj.param_1 = p.Results.param_1;
            obj.param_2 = p.Results.param_2;
        end
        
        function [history, best] = solver(obj, search_agents_no, max_iter)
            population = obj.init_population(search_agents_no);
            obj.begin_step_solver(max_iter);
            
            for iter = 1:max_iter
                % Update logic
                for i = 1:length(population)
                    % ...
                end
                
                % Track best solution
                fitness_values = obj.get_fitness(population);
                if obj.maximize
                    [best_fitness, best_idx] = max(fitness_values);
                else
                    [best_fitness, best_idx] = min(fitness_values);
                end
                obj.history_step_solver{end+1} = population{best_idx}.copy();
                obj.best_solver = population{best_idx};
                
                obj.callbacks(iter, max_iter, population{best_idx});
            end
            
            obj.end_step_solver();
            history = obj.history_step_solver;
            best = obj.best_solver;
        end
    end
end
```

## 📋 Completion Checklist

* [ ] Create file under `src/` with correct name
* [ ] Inherit from the correct base class (`Solver` or `MultiObjectiveSolver`)
* [ ] Implement the `solver` method
* [ ] Set `obj.name_solver` in the constructor
* [ ] Parse algorithm parameters with `inputParser`
* [ ] Ensure positions stay within bounds (`lb`, `ub`)
* [ ] Correctly update history and best solution
* [ ] Verify progress display and final results

## 🆘 Support

See existing examples in `src/`:

* `MultiObjectiveWhaleOptimizer.m` – Multi-objective example
* Other algorithms will be added later

Check base classes in `core/` for available utility methods.
