% Add current directory to path
clear; close all; clc;
add_lib(pwd);

% Parameter
dim = 2;
lb = zeros(1, dim);
ub = ones(1, dim);
search_agents_no = 100;
max_iter = 100;
maximize = false;
objective_func = @(x) zdt1_function(x);
varargin = [];

% Create solver for ZDT1 using the factory function
all_solver = SolverFactory();
all_solver.show_solvers();
method = all_solver.create_solver('MultiObjectiveTeachingLearningBasedOptimizer', objective_func, lb, ub, dim, maximize, varargin);

% Run optimization for ZDT1
[history, archive] = method.solver(search_agents_no, max_iter);
%plot_history_multi_animation(history)