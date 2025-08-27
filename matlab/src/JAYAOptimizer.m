classdef JAYAOptimizer < Solver
    %{
    JAYA (To Win) Optimizer
    
    A simple and powerful optimization algorithm that always tries to get closer 
    to success (best solution) and away from failure (worst solution).
    
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
        Additional algorithm parameters
    %}
    
    methods
        function obj = JAYAOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            JAYAOptimizer constructor - Initialize the JAYA solver
            
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
                    Additional JAYA parameters
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set solver name
            obj.name_solver = "JAYA Optimizer";
            
            % JAYA algorithm doesn't have specific parameters beyond the base ones
            % The algorithm uses random coefficients in the update equation
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for JAYA algorithm
            
            Inputs:
                search_agents_no : int
                    Number of search agents (population size)
                max_iter : int
                    Maximum number of iterations
                    
            Returns:
                history_step_solver : cell array
                    History of best solutions at each iteration
                best_solver : Member
                    Best solution found overall
            %}
            
            % Initialize storage variables
            history_step_solver = {};
            
            % Initialize the population
            population = obj.init_population(search_agents_no);
            
            % Initialize best solution
            sorted_population = obj.sort_population(population);
            best_solver = sorted_population{1}.copy();
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Find best and worst solutions in current population
                sorted_population = obj.sort_population(population);
                best_member = sorted_population{1};
                worst_member = sorted_population{end};
                
                % Update each search agent
                for i = 1:search_agents_no
                    % Generate new candidate solution using JAYA update equation
                    % xnew = x + rand*(best - |x|) - rand*(worst - |x|)
                    % Note: The MATLAB code uses abs(x(i,j)) but this seems unusual
                    % We'll implement the exact formula from the MATLAB code
                    
                    % Create new position using JAYA formula
                    new_position = zeros(1, obj.dim);
                    for j = 1:obj.dim
                        rand1 = rand();
                        rand2 = rand();
                        new_position(j) = (...
                            population{i}.position(j) + ...
                            rand1 * (best_member.position(j) - abs(population{i}.position(j))) - ...
                            rand2 * (worst_member.position(j) - abs(population{i}.position(j)))...
                        );
                    end
                    
                    % Apply bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Evaluate new fitness
                    new_fitness = obj.objective_func(new_position);
                    
                    % Greedy selection: accept if better
                    if obj.is_better(Member(new_position, new_fitness), population{i})
                        population{i}.position = new_position;
                        population{i}.fitness = new_fitness;
                    end
                end
                
                % Update best solution
                sorted_population = obj.sort_population(population);
                current_best = sorted_population{1};
                if obj.is_better(current_best, best_solver)
                    best_solver = current_best.copy();
                end
                
                % Store the best solution at this iteration
                history_step_solver{end+1} = best_solver.copy();
                
                % Update progress
                obj.callbacks(iter, max_iter, best_solver);
            end
            
            % Final evaluation and storage
            obj.history_step_solver = history_step_solver;
            obj.best_solver = best_solver;
            
            % End solver
            obj.end_step_solver();
        end
        
        function [sorted_population, sorted_indices] = sort_population(obj, population)
            %{
            sort_population - Sort the population based on fitness
            
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
