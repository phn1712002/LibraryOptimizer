classdef GreyWolfOptimizer < Solver
    %{
    Grey Wolf Optimizer (GWO) Algorithm.
    
    GWO is a nature-inspired metaheuristic optimization algorithm that mimics
    the leadership hierarchy and hunting behavior of grey wolves in nature.
    The algorithm considers the social hierarchy of wolves and simulates three
    main steps of hunting: searching for prey, encircling prey, and attacking prey.
    
    The social hierarchy consists of:
    - Alpha (α): Best solution
    - Beta (β): Second best solution  
    - Delta (δ): Third best solution
    - Omega (ω): Other candidate solutions
    
    References:
        Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer.
        Advances in engineering software, 69, 46-61.
    %}
    
    methods
        function obj = GreyWolfOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            GreyWolfOptimizer constructor - Initialize the GWO solver
            
            Inputs:
                objective_func : function handle
                    Objective function to optimize
                lb : float or array
                    Lower bounds of search space
                ub : float or array
                    Upper bounds of search space
                dim : int
