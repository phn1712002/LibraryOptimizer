classdef HarmonySearchOptimizer < Solver
    %{
    Harmony Search Algorithm.
    
    Harmony Search is a music-inspired metaheuristic optimization algorithm that
    mimics the process of musicians improvising harmonies to find the perfect state
    of harmony. The algorithm maintains a harmony memory (HM) of candidate solutions
    and generates new harmonies through three operations: memory consideration,
    pitch adjustment, and random selection.
    
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
        Optimization direction (true: maximize, false: minimize)
    varargin : cell array
        Additional algorithm parameters:
        - hmcr: Harmony Memory Considering Rate (default: 0.95)
        - par: Pitch Adjustment Rate (default: 0.3)
        - bw: Bandwidth (default: 0.2)
    
    References:
        Geem, Z. W., Kim, J. H., & Loganathan, G. V. (2001). 
        A new heuristic optimization algorithm: harmony search. 
        Simulation, 76(2), 60-68.
    %}
    
    properties
        hmcr
        par
        bw
        harmony_memory
        harmony_fitness
    end
    
    methods
        function obj = HarmonySearchOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            HarmonySearchOptimizer constructor - Initialize the Harmony Search solver
            
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
            obj.name_solver = "Harmony Search Optimizer";
            
            % Set algorithm-specific parameters with defaults
            obj.hmcr = obj.get_kw('hmcr', 0.95);  % Harmony Memory Considering Rate
            obj.par = obj.get_kw('par', 0.3);  % Pitch Adjustment Rate
            obj.bw = obj.get_kw('bw', 0.2);  % Bandwidth
            
            % Initialize harmony memory
            obj.harmony_memory = [];
            obj.harmony_fitness = [];
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for Harmony Search Algorithm
            
            Parameters:
            -----------
            search_agents_no : int
                Number of search agents (population size)
            max_iter : int
                Maximum number of iterations
                
            Returns:
            --------
            history_step_solver : cell array
                History of best solutions at each iteration
            best_solver : Member
                Best solution found overall
            %}
            
            % Initialize storage variables
            history_step_solver = {};
            
            % Initialize harmony memory
            obj.harmony_memory = zeros(search_agents_no, obj.dim);
            obj.harmony_fitness = zeros(search_agents_no, 1);
            
            % Initialize harmony memory with random solutions
            for i = 1:search_agents_no
                obj.harmony_memory(i, :) = rand(1, obj.dim) .* (obj.ub - obj.lb) + obj.lb;
                obj.harmony_fitness(i) = obj.objective_func(obj.harmony_memory(i, :));
            end
            
            % Find initial best and worst solutions
            if obj.maximize
                [~, best_idx] = max(obj.harmony_fitness);
                [~, worst_idx] = min(obj.harmony_fitness);
            else
                [~, best_idx] = min(obj.harmony_fitness);
                [~, worst_idx] = max(obj.harmony_fitness);
            end
            
            best_solver = Member(obj.harmony_memory(best_idx, :), obj.harmony_fitness(best_idx));
            worst_fitness = obj.harmony_fitness(worst_idx);
            worst_idx_current = worst_idx;
            
            % Call the begin function
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Create a new harmony
                new_harmony = zeros(1, obj.dim);
                
                for j = 1:obj.dim
                    if rand() < obj.hmcr
                        % Memory consideration: select from harmony memory
                        harmony_idx = randi(search_agents_no);
                        new_harmony(j) = obj.harmony_memory(harmony_idx, j);
                        
                        % Pitch adjustment
                        if rand() < obj.par
                            new_harmony(j) = new_harmony(j) + obj.bw * (2 * rand() - 1);
                        end
                    else
                        % Random selection
                        new_harmony(j) = rand() * (obj.ub(j) - obj.lb(j)) + obj.lb(j);
                    end
                end
                
                % Ensure positions stay within bounds
                new_harmony = max(min(new_harmony, obj.ub), obj.lb);
                
                % Evaluate new harmony
                new_fitness = obj.objective_func(new_harmony);
                
                % Update harmony memory if new harmony is better than worst
                if obj.is_better_fitness(new_fitness, worst_fitness)
                    obj.harmony_memory(worst_idx_current, :) = new_harmony;
                    obj.harmony_fitness(worst_idx_current) = new_fitness;
                    
                    % Update best and worst
                    if obj.maximize
                        [~, best_idx] = max(obj.harmony_fitness);
                        [~, worst_idx_current] = min(obj.harmony_fitness);
                    else
                        [~, best_idx] = min(obj.harmony_fitness);
                        [~, worst_idx_current] = max(obj.harmony_fitness);
                    end
                    
                    current_best = Member(obj.harmony_memory(best_idx, :), obj.harmony_fitness(best_idx));
                    worst_fitness = obj.harmony_fitness(worst_idx_current);
                    
                    % Update best solution if improved
                    if obj.is_better(current_best, best_solver)
                        best_solver = current_best.copy();
                    end
                end
                
                % Store the best solution at this iteration
                history_step_solver{end+1} = best_solver.copy();
                
                % Call the callbacks
                obj.callbacks(iter, max_iter, best_solver);
            end
            
            % Final processing
            obj.history_step_solver = history_step_solver;
            obj.best_solver = best_solver;
            
            % Call the end function
            obj.end_step_solver();
        end
        
        function result = is_better_fitness(obj, fitness_1, fitness_2)
            %{
            is_better_fitness - Compare two fitness values based on optimization direction
            
            Inputs:
                fitness_1 : float
                    First fitness value
                fitness_2 : float
                    Second fitness value
                    
            Returns:
                result : bool
                    true if fitness_1 is better than fitness_2 according to optimization direction
            %}
            
            if obj.maximize
                result = fitness_1 > fitness_2;
            else
                result = fitness_1 < fitness_2;
            end
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
