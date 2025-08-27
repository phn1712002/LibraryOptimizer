classdef MultiObjectiveHarmonySearchOptimizer < MultiObjectiveSolver
    %{
    Multi-Objective Harmony Search (MOHS) Optimization Algorithm.
    
    MOHS extends the Harmony Search algorithm for multi-objective optimization problems.
    It maintains an archive of non-dominated solutions and uses Pareto dominance
    for solution comparison and archive management.
    
    Parameters:
    -----------
    objective_func : function handle
        Multi-objective function to optimize (returns array of objectives)
    lb : float or array
        Lower bounds for variables
    ub : float or array
        Upper bounds for variables  
    dim : int
        Problem dimension
    maximize : bool or array
        Optimization direction for each objective (true: maximize, false: minimize)
    **kwargs
        Additional algorithm parameters including:
        - hmcr: Harmony Memory Considering Rate (default: 0.95)
        - par: Pitch Adjustment Rate (default: 0.3)
        - bw: Bandwidth (default: 0.2)
        - archive_size: Maximum size of Pareto archive (default: 100)
    %}
    
    properties
        hmcr
        par
        bw
        archive_size
        harmony_memory
        harmony_fitness
    end
    
    methods
        function obj = MultiObjectiveHarmonySearchOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveHarmonySearchOptimizer constructor
            
            Inputs:
                objective_func : function handle
                    Multi-objective function to optimize
                lb : float or array
                    Lower bounds of search space
                ub : float or array
                    Upper bounds of search space
                dim : int
                    Number of dimensions in the problem
                maximize : bool or array
                    Optimization direction for each objective
                varargin : cell array
                    Additional solver parameters
            %}
            
            % Call parent constructor
            obj@MultiObjectiveSolver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set solver name
            obj.name_solver = "Multi-Objective Harmony Search Optimizer";
            
            % Set default MOHS parameters
            obj.hmcr = obj.get_kw('hmcr', 0.95);
            obj.par = obj.get_kw('par', 0.3);
            obj.bw = obj.get_kw('bw', 0.2);
            obj.archive_size = obj.get_kw('archive_size', 100);
            
            % Initialize harmony memory
            obj.harmony_memory = [];
            obj.harmony_fitness = [];
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for MOHS algorithm
            
            Inputs:
                search_agents_no : int
                    Number of search agents (harmony memory size)
                max_iter : int
                    Maximum number of iterations
                    
            Returns:
                Tuple containing:
                    - history_archive: Cell array of archive states at each iteration
                    - archive: Final Pareto archive
            %}
            
            % Initialize storage variables
            history_archive = {};
            
            % Initialize harmony memory
            obj.harmony_memory = zeros(search_agents_no, obj.dim);
            obj.harmony_fitness = zeros(search_agents_no, length(obj.maximize));
            
            % Initialize harmony memory with random solutions
            for i = 1:search_agents_no
                position = rand(1, obj.dim) .* (obj.ub - obj.lb) + obj.lb;
                fitness = obj.objective_func(position);
                obj.harmony_memory(i, :) = position;
                obj.harmony_fitness(i, :) = fitness;
            end
            
            % Initialize archive with non-dominated solutions
            archive = obj.initialize_archive_from_matrix(obj.harmony_memory, obj.harmony_fitness, obj.archive_size);
            
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
                new_member = MultiObjectiveMember(new_harmony, new_fitness);
                
                % Add to archive if non-dominated
                archive = obj.add_to_archive(new_member, archive, obj.archive_size);
                
                % Update harmony memory using crowding distance-based replacement
                if ~isempty(archive)
                    % Find the most crowded solution in harmony memory
                    crowding_distances = obj.calculate_crowding_distance_matrix(obj.harmony_fitness);
                    [~, most_crowded_idx] = min(crowding_distances);
                    
                    % Replace if new solution dominates or is non-dominated
                    old_member = MultiObjectiveMember(obj.harmony_memory(most_crowded_idx, :), ...
                                                     obj.harmony_fitness(most_crowded_idx, :));
                    
                    if obj.dominates(new_member, old_member) || ...
                       (~obj.dominates(old_member, new_member) && rand() < 0.5)
                        obj.harmony_memory(most_crowded_idx, :) = new_harmony;
                        obj.harmony_fitness(most_crowded_idx, :) = new_fitness;
                    end
                end
                
                % Store archive history
                history_archive{end+1} = archive;
                
                % Call the callbacks
                obj.callbacks(iter, max_iter, archive);
            end
            
            % Final processing
            obj.history_archive = history_archive;
            obj.archive = archive;
            
            % Call the end function
            obj.end_step_solver();
        end
    end
end
