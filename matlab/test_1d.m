% Add current directory to path
clear; close all; clc;
add_lib(pwd);

% Parameter
dim = 10; 
lb = -5 * ones(1, dim);    
ub = 5 * ones(1, dim);     
search_agents_no = 50;
max_iter = 100;
maximize = false;
objective_func = @(x) sphere_function(x);
varargin = [];

% Create solver for Sphere using the factory function
all_solver = SolverFactory();
all_solver.show_solvers();
method = all_solver.create_solver('BatOptimizer', objective_func, lb, ub, dim, maximize, varargin);
[history, archive] = method.solver(search_agents_no, max_iter);