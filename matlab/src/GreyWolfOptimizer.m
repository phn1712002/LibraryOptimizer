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
                    Number of dimensions in the problem
                maximize : bool
                    Whether to maximize (true) or minimize (false) objective
                varargin : cell array
                    Additional solver parameters
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set solver name
            obj.name_solver = "Grey Wolf Optimizer";
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for GWO algorithm
            
            The algorithm simulates the hunting behavior of grey wolves through
            three main phases controlled by coefficient vectors:
            1. Searching for prey: Exploration phase with random search agents
            2. Encircling prey: Exploitation phase moving towards alpha, beta, delta
            3. Attacking prey: Convergence phase as coefficients approach zero
            
            Inputs:
                search_agents_no : int
                    Number of grey wolves in the pack
                max_iter : int
                    Maximum number of iterations for optimization
                    
            Returns:
                history_step_solver : cell array
                    History of best solutions at each iteration
                best_solver : Member
                    Best solution (alpha wolf) found overall
            %}
            
            % Initialize the population of search agents
            population = obj.init_population(search_agents_no);

            % Initialize storage variables
            history_step_solver = {};
            best_solver = obj.best_solver;
            
            % Call the begin function
            obj.begin_step_solver(max_iter);

            % Main optimization loop
            for iter = 1:max_iter
                % Update alpha, beta, delta based on current population
                [sorted_population, idx] = obj.sort_population(population);
                alpha = sorted_population{1}.copy();
                beta = sorted_population{2}.copy();
                delta = sorted_population{3}.copy();

                % Update a parameter (decreases linearly from 2 to 0)
                a = 2 - iter * (2 / max_iter);
                
                % Update all search agents
                for i = 1:length(population)
                    new_position = zeros(1, obj.dim);
                    
                    for j = 1:obj.dim
                        % Update position using alpha, beta, and delta wolves
                        r1 = rand();
                        r2 = rand();
                        
                        A1 = 2 * a * r1 - a;
                        C1 = 2 * r2;
                        
                        D_alpha = abs(C1 * alpha.position(j) - population{i}.position(j));
                        X1 = alpha.position(j) - A1 * D_alpha;
                        
                        r1 = rand();
                        r2 = rand();
                        
                        A2 = 2 * a * r1 - a;
                        C2 = 2 * r2;
                        
                        D_beta = abs(C2 * beta.position(j) - population{i}.position(j));
                        X2 = beta.position(j) - A2 * D_beta;
                        
                        r1 = rand();
                        r2 = rand();
                        
                        A3 = 2 * a * r1 - a;
                        C3 = 2 * r2;
                        
                        D_delta = abs(C3 * delta.position(j) - population{i}.position(j));
                        X3 = delta.position(j) - A3 * D_delta;
                        
                        % Update position component
                        new_position(j) = (X1 + X2 + X3) / 3;
                    end
                    
                    % Ensure positions stay within bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Update member position and fitness
                    population{i}.position = new_position;
                    population{i}.fitness = obj.objective_func(new_position);
                    
                    % Update best immediately if better solution found
                    if obj.is_better(population{i}, best_solver)
                        best_solver = population{i}.copy();
                    end
                end
                
                % Store the best solution at this iteration
                history_step_solver{end+1} = best_solver.copy();
                
                % Call the callbacks 
                obj.callbacks(iter, max_iter, best_solver); 
            end
            
            % Final evaluation of all positions to find the best solution
            obj.history_step_solver = history_step_solver;
            obj.best_solver = best_solver;
            
            % Call the end function
            obj.end_step_solver();
        end
        
        function [sorted_population, sorted_indices] = sort_population(obj, population)
            %{
            sort_population - Sort population based on fitness
            
            Inputs:
                population : cell array
                    Population to sort
                    
            Returns:
                sorted_population : cell array
                    Sorted population
                sorted_indices : array
                    Indices of sorted order
            %}
            
            % Extract fitness values from population
            fitness_values = zeros(1, length(population));
            for i = 1:length(population)
                fitness_values(i) = population{i}.fitness;
            end
            
            % Sort indices based on optimization direction
            if obj.maximize
                % Sort in descending order for maximization
                [~, sorted_indices] = sort(fitness_values, 'descend');
            else
                % Sort in ascending order for minimization
                [~, sorted_indices] = sort(fitness_values, 'ascend');
            end
            
            % Sort population based on sorted indices
            sorted_population = cell(1, length(population));
            for i = 1:length(population)
                sorted_population{i} = population{sorted_indices(i)};
            end
        end
    end
end
