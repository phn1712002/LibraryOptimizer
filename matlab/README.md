# 🚀 MATLAB Optimizer Library

A metaheuristic optimization library in **MATLAB**, featuring both **single-objective** and **multi-objective** algorithms with a unified and extensible interface.  

## ✨ Features

- **Metaheuristic Algorithms**: Particle Swarm Optimization (PSO), Grey Wolf Optimizer (GWO), Whale Optimization Algorithm (WOA), Artificial Bee Colony (ABC), and many more.  
- **Multi-Objective Support**: Multi-objective versions directly extended from single-objective algorithms.  
- **Unified Structure**: All algorithms follow the same base classes (`Solver`, `MultiObjectiveSolver`).  
- **Visualization**: Built-in plotting of convergence curves and Pareto fronts.  
- **Extensible**: Add new algorithms easily by following a standard template.  

---

## 📂 Folder Structure

```bash
matlab/
├── core/           # Base classes (Solver, MultiObjectiveSolver, Member, etc.)
├── examples/       # Contains sample scripts
├── measurements/   # Contains evaluation functions
├── problems/       # Contains objective functions, problem functions
├── src/            # Implementations of optimization algorithms
├── strucs/         # Data structures for individuals (Particle, MultiMember, etc.)
├── utils/          # Utility functions
└── add_lib.m       # Script to add subfolders automatically
└── main.m          # Example script to run tests
````

---

## ⚡ Usage
### 1. View the algorithms available in the library (Console):
```matlab
clear; close all; clc;
add_lib(pwd);                   % Add all folders to MATLAB's working path
SolverFactory().show_solvers(); % This line of code displays all the algorithms available in the library
```
---

### 2. Single-Objective Optimization

```matlab
% Add current directory to path
clear; close all; clc;
add_lib(pwd);

% Parameters
dim = 2; 
lb = -5 * ones(1, dim);    
ub = 5 * ones(1, dim);      
search_agents_no = 50;
max_iter = 100;
maximize = false;
objective_func = @(x) sphere_function(x);

% Create solver using factory
all_solver = SolverFactory();
all_solver.show_solvers();
method = all_solver.create_solver('TeachingLearningBasedOptimizer', ...
    objective_func, lb, ub, dim, maximize);

% Run optimization
[history, best] = method.solver(search_agents_no, max_iter);
```
---

### 3. Multi-Objective Optimization

```matlab
% Add current directory to path
clear; close all; clc;
add_lib(pwd);

% Parameters
dim = 2;
lb = zeros(1, dim);
ub = ones(1, dim);
search_agents_no = 100;
max_iter = 100;
maximize = false;
objective_func = @(x) zdt1_function(x);

% Create solver for multi-objective optimization
all_solver = SolverFactory();
all_solver.show_solvers();
method = all_solver.create_solver('MultiObjectiveShuffledFrogLeapingOptimizer', ...
    objective_func, lb, ub, dim, maximize);

% Run optimization
[history, archive] = method.solver(search_agents_no, max_iter);
```
---

## 🛠️ Adding a New Algorithm

1. Create a new file under `src/` (e.g., `GreyWolfOptimizer.m`).
2. Inherit from `Solver` (single-objective) or `MultiObjectiveSolver` (multi-objective).
3. Implement the `solver(search_agents_no, max_iter)` method.
4. Set `obj.name_solver` in the constructor.
5. Ensure proper updates of `history` and `best` solutions.



## 📊 Key Base Classes

* `Solver.m`: Base class for single-objective optimization
* `MultiObjectiveSolver.m`: Base class for multi-objective optimization
* `Member.m`: Represents an individual (single-objective)
* `MultiObjectiveMember.m`: Represents an individual (multi-objective)

## 📋 Available Algorithms

* Grey Wolf Optimizer (GWO) / Multi-Objective GWO
* Whale Optimization Algorithm (WOA) / Multi-Objective WOA
* Particle Swarm Optimization (PSO) / Multi-Objective PSO
* Artificial Bee Colony (ABC) / Multi-Objective ABC
* Genetic Algorithm (GA) / Multi-Objective GA
* And more to be added...


## 📑 Documentation

* See `example_1d.m`, `example_2d.m`, .... for usage examples.
* Reference implementations:

  * `src/ParticleSwarmOptimizer.m` (single-objective)
  * `src/MultiObjectiveParticleSwarmOptimizer.m` (multi-objective)



## 👨‍💻 Author

* Author: phn1712002
* Email: [phn1712002@gmail.com](mailto:phn1712002@gmail.com)
* GitHub: [phn1712002](https://github.com/phn1712002)

## Acknowledgments

This library builds upon research in metaheuristic optimization and implements algorithms from various scientific publications.