% Main script to test Multi-Objective Grey Wolf Optimizer
clear; close all; clc;

% Add current directory to path
addpath(pwd);

dim = 2;
lb = zeros(1, dim);
ub = ones(1, dim);
search_agents_no = 100;
max_iter = 100;
maximize = false;

% Create MOGWO solver for ZDT1
mogwo = MultiObjectiveWhaleOptimizer(@zdt1_function, lb, ub, dim, maximize);

% Run optimization for ZDT1
[history, archive] = mogwo.solver(search_agents_no, max_iter);

% Test multi-objective function - ZDT1 benchmark problem
function fitness = zdt1_function(x)
    % ZDT1 benchmark problem (minimization)
    % x should be a vector of length 30 (standard ZDT1 dimension)
    n = length(x);
    
    % Objective 1: f1(x) = x1
    f1 = x(1);
    
    % Objective 2: f2(x) = g(x) * (1 - sqrt(x1/g(x)))
    g = 1 + 9 * sum(x(2:end)) / (n - 1);
    f2 = g * (1 - sqrt(x(1) / g));
    
    fitness = [f1, f2];
end
