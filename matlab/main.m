% Main script with auto-detection of objective function type
clear; close all; clc;

% Add current directory to path
add_lib(pwd);

% Create instance of your fitness function
fitness_obj = my_fitness();

% Set parameters based on your fitness function properties
dim = fitness_obj.n_dim;          % Number of dimensions (3)
lb = fitness_obj.lb;             % Lower bounds [22, 100, 0.3]
ub = fitness_obj.ub;             % Upper bounds [26, 140, 0.5]
maximize = fitness_obj.maximize; % Minimization (false)

% Define objective function handle
objective_func = @(x) fitness_obj.calculation(x);

% Optimization parameters
search_agents_no = 50;  % Population size
max_iter = 100;         % Maximum iterations

% Use SolverFactory with auto-detection - it will detect multi-objective function
% and automatically use the multi-objective version of ParticleSwarmOptimizer
method = SolverFactory.create_solver('ParticleSwarmOptimizer', ...
    objective_func, lb, ub, dim, maximize, ...
    'w', 0.7, 'c1', 1.5, 'c2', 1.5);

% Run optimization
[history, archive] = method.solver(search_agents_no, max_iter);
