classdef SeagullOptimizer < Solver
    %{
    Seagull Optimization Algorithm (SOA).
    
    SOA is a nature-inspired metaheuristic optimization algorithm that mimics
    the migration and attacking behavior of seagulls in nature. The algorithm
    simulates the migration and attacking behaviors of seagulls, which include:
    
    1. Migration (exploration): Seagulls move towards the best position
    2. Attacking (exploitation): Seagulls attack prey using spiral movements
    
    The algorithm uses a control parameter Fc that decreases linearly to balance
    exploration and exploitation phases.
    
    References:
        Dhiman, G., & Kumar, V. (2019). Seagull optimization algorithm: 
        Theory and its applications for large-scale industrial engineering problems. 
        Knowledge-Based Systems, 165, 169-196.
    %}
    
    methods
        function obj = SeagullOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            SeagullOptimizer constructor - Initialize the SOA solver
            
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
                    Additional algorithm parameters
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set algorithm name
            obj.name_solver = "Seagull Optimization Algorithm";
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for SOA algorithm
            
            Inputs:
                search_agents_no : int
                    Number of search agents (seagulls)
                max_iter : int
                    Maximum number of iterations
                    
            Returns:
                history_step_solver : cell array
                    History of best solutions at each iteration
                best_solver : Member
                    Best solution found overall
            %}
            
            % Initialize the population of search agents
            population = obj.init_population(search_agents_no);

            % Initialize storage variables
            history_step_solver = {};
            
            % Initialize best solution
            sorted_population = obj.sort_population(population);
            best_solver = sorted_population{1}.copy();
            
            % Call the begin function
            obj.begin_step_solver(max_iter);

            % Main optimization loop
            for iter = 1:max_iter
                % Evaluate all search agents and update best solution
                for i = 1:search_agents_no
                    % Ensure positions stay within bounds
                    population{i}.position = max(min(population{i}.position, obj.ub), obj.lb);
                    
                    % Update fitness
                    population{i}.fitness = obj.objective_func(population{i}.position);
                    
                    % Update best solution if better solution found
                    if obj.is_better(population{i}, best_solver)
                        best_solver = population{i}.copy();
                    end
                end
                
                % Update control parameter Fc (decreases linearly from 2 to 0)
                Fc = 2 - iter * (2 / max_iter);
                
                % Update all search agents
                for i = 1:search_agents_no
                    new_position = zeros(1, obj.dim);
                    
                    for j = 1:obj.dim
                        % Generate random numbers
                        r1 = rand();
                        r2 = rand();
                        
                        % Calculate A1 and C1 parameters
                        A1 = 2 * Fc * r1 - Fc;
                        C1 = 2 * r2;
                        
                        % Calculate ll parameter
                        ll = (Fc - 1) * rand() + 1;
                        
                        % Calculate D_alphs (direction towards best solution)
                        D_alphs = Fc * population{i}.position(j) + ...
                                 A1 * (best_solver.position(j) - population{i}.position(j));
                        
                        % Update position using spiral movement (attacking behavior)
                        X1 = D_alphs * exp(ll) * cos(ll * 2 * pi) + best_solver.position(j);
                        new_position(j) = X1;
                    end
                    
                    % Ensure positions stay within bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Update member position
                    population{i}.position = new_position;
                end
                
                % Store the best solution at this iteration
                history_step_solver{end+1} = best_solver.copy();
                
                % Call the callbacks 
                obj.callbacks(iter, max_iter, best_solver); 
            end
            
            % Final evaluation of all positions to find the best solution
            for i = 1:search_agents_no
                % Ensure positions stay within bounds
                population{i}.position = max(min(population{i}.position, obj.ub), obj.lb);
                
                % Update fitness
                population{i}.fitness = obj.objective_func(population{i}.position);
                
                % Update best solution if better solution found
                if obj.is_better(population{i}, best_solver)
                    best_solver = population{i}.copy();
                end
            end
            
            % Final evaluation and storage
            obj.history_step_solver = history_step_solver;
            obj.best_solver = best_solver;
            
            % Call the end function
            obj.end_step_solver();
        end
        
        function sorted_population = sort_population(obj, population)
            %{
            sort_population - Sort population based on fitness
            
            Inputs:
                population : cell array
                    Population to sort
                    
            Returns:
                sorted_population : cell array
                    Sorted population (best first)
            %}
            
            % Extract fitness values
            fitness_values = obj.get_fitness(population);
            
            % Sort based on optimization direction
            if obj.maximize
                [~, sorted_indices] = sort(fitness_values, 'descend');
            else
                [~, sorted_indices] = sort(fitness_values, 'ascend');
            end
            
            % Sort population
            sorted_population = cell(1, length(population));
            for i = 1:length(population)
                sorted_population{i} = population{sorted_indices(i)};
            end
        end
    end
end
