classdef GravitationalSearchOptimizer < Solver
    %{
    Gravitational Search Algorithm (GSA) Optimizer
    
    Reference: Rashedi, Esmat, Hossein Nezamabadi-Pour, and Saeid Saryazdi. "GSA: a gravitational search algorithm." 
               Information sciences 179.13 (2009): 2232-2248.
    
    GSA is a population-based optimization algorithm inspired by the law of gravity and mass interactions.
    Each solution is considered as a mass, and their interactions are governed by gravitational forces.
    
    Parameters:
    -----------
    objective_func : function handle
        Objective function to optimize
    lb : float or array
        Lower bounds for variables
    ub : float or array
        Upper bounds for variables
    dim : int
        Problem dimension
    maximize : bool
        Whether to maximize (true) or minimize (false) objective
    varargin : cell array
        Additional algorithm parameters:
        - elitist_check: Whether to use elitist strategy (default: true)
        - r_power: Power parameter for distance calculation (default: 1)
        - g0: Initial gravitational constant (default: 100)
        - alpha: Decay parameter for gravitational constant (default: 20)
    %}
    
    properties
        elitist_check  % Whether to use elitist strategy
        r_power        % Power parameter for distance calculation
        g0             % Initial gravitational constant
        alpha          % Decay parameter for gravitational constant
        
        % Internal state variables
        velocities
    end
    
    methods
        function obj = GravitationalSearchOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            GravitationalSearchOptimizer constructor - Initialize the GSA solver
            
            Inputs:
                objective_func : function handle
                    Objective function to optimize
                lb : float or array
                    Lower bounds of search space
                ub : float or array
                    Upper bounds of search space
                dim : int
                    Number of dimensions in the problem
                maximize : bool
                    Whether to maximize (true) or minimize (false) objective
                varargin : cell array
                    Additional GSA parameters:
                    - elitist_check: Whether to use elitist strategy (default: true)
