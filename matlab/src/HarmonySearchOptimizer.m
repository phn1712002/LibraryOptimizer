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
        Optimization direction
    **kwargs
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
            HarmonySearchOptimizer constructor - Initialize the HS solver
            
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
            
            % Set default HS parameters
            obj.hmcr = obj.get_kw('hmcr', 0.95);  % Harmony Memory Considering Rate
            obj.par = obj.get_kw('par', 0.3);     % Pitch Adjustment Rate
            obj.bw = obj.get_kw('bw', 0.2);       % Bandwidth
            
            % Initialize harmony memory
            obj.harmony_memory = [];
            obj.harmony_fitness = [];
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for Harmony Search algorithm
            
            Inputs:
                search_agents_no : int
                    Number of search agents (population size)
                max_iter : int
                    Maximum number of iterations
                    
            Returns:
                Tuple containing:
                    - history_step_solver: Cell array of best solutions at each iteration
                    - best_solver: Best solution found overall
            %}
            
            % Initialize storage variables
            history_step_solver = {};
            
            % Initialize harmony memory
            obj.harmony_memory = zeros(search_agents_no, obj.dim);
            obj.harmony_fitness = zeros(1, search_agents_no);
            
            % Initialize harmony memory with random solutions
            for i = 1:search_agents_no
                position = rand(1, obj.dim) .* (obj.ub - obj.lb) + obj.lb;
                fitness = obj.objective_func(position);
                obj.harmony_memory(i, :) = position;
                obj.harmony_fitness(i) = fitness;
            end
            
            % Find initial best and worst solutions
            if obj.maximize
                [best_fitness, best_idx] = max(obj.harmony_fitness);
                [worst_fitness, worst_idx] = min(obj.harmony_fitness);
            else
                [best_fitness, best_idx] = min(obj.harmony_fitness);
                [worst_fitness, worst_idx] = max(obj.harmony_fitness);
            end
            
            best_solver = Member(obj.harmony_memory(best_idx, :), best_fitness);
            worst_fitness_current = worst_fitness;
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
                if obj.is_better(new_fitness, worst_fitness_current)
                    obj.harmony_memory(worst_idx_current, :) = new_harmony;
                    obj.harmony_fitness(worst_idx_current) = new_fitness;
                    
                    % Update best and worst
                    if obj.maximize
                        [best_fitness, best_idx] = max(obj.harmony_fitness);
                        [worst_fitness_current, worst_idx_current] = min(obj.harmony_fitness);
                    else
                        [best_fitness, best_idx] = min(obj.harmony_fitness);
                        [worst_fitness_current, worst_idx_current] = max(obj.harmony_fitness);
                    end
                    
                    current_best = Member(obj.harmony_memory(best_idx, :), best_fitness);
                    
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
    end
end
