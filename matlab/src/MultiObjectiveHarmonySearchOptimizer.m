classdef MultiObjectiveHarmonySearchOptimizer < MultiObjectiveSolver
    %{
    Multi-Objective Harmony Search Algorithm.
    
    This algorithm extends the standard Harmony Search for multi-objective optimization
    using archive management and grid-based selection for maintaining diversity.
    
    Parameters:
    -----------
    objective_func : function handle
        Objective function that returns an array of fitness values
    lb : float or array
        Lower bounds for variables
    ub : float or array
        Upper bounds for variables
    dim : int
        Problem dimension
    maximize : bool
        Optimization direction (true for maximize, false for minimize)
    varargin : cell array
        Additional algorithm parameters:
        - hmcr: Harmony Memory Considering Rate (default: 0.95)
        - par: Pitch Adjustment Rate (default: 0.3)
        - bw: Bandwidth (default: 0.2)
        - archive_size: Size of the external archive (default: 100)
        - alpha: Grid inflation parameter (default: 0.1)
        - n_grid: Number of grids per dimension (default: 7)
        - beta: Leader selection pressure (default: 2)
        - gamma: Archive removal pressure (default: 2)
    %}
    
    properties
        hmcr
        par
        bw
        harmony_memory
        harmony_fitness
    end
    
    methods
        function obj = MultiObjectiveHarmonySearchOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveHarmonySearchOptimizer constructor - Initialize the MOHS solver
            
            Inputs:
                objective_func : function handle
                    Objective function to optimize (returns array for multiple objectives)
                lb : float or array
                    Lower bounds of search space
                ub : float or array
                    Upper bounds of search space
                dim : int
                    Number of dimensions in the problem
                maximize : bool
                    Whether to maximize (true) or minimize (false) objectives
                varargin : cell array
                    Additional solver parameters
            %}
            
            % Call parent constructor
            obj@MultiObjectiveSolver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set solver name
            obj.name_solver = "Multi-Objective Harmony Search Optimizer";
            
            % Algorithm-specific parameters with defaults
            obj.hmcr = obj.get_kw('hmcr', 0.95);  % Harmony Memory Considering Rate
            obj.par = obj.get_kw('par', 0.3);  % Pitch Adjustment Rate
            obj.bw = obj.get_kw('bw', 0.2);  % Bandwidth
            
            % Initialize harmony memory
            obj.harmony_memory = [];
            obj.harmony_fitness = [];
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective Harmony Search
            
            Parameters:
            -----------
            search_agents_no : int
                Number of search agents (population size)
            max_iter : int
                Maximum number of iterations
                
            Returns:
                history_archive : cell array
                    History of archive states
                archive : cell array
                    Final archive of non-dominated solutions
            %}
            
            % Initialize storage
            history_archive = {};
            
            % Initialize harmony memory
            obj.harmony_memory = zeros(search_agents_no, obj.dim);
            obj.harmony_fitness = cell(search_agents_no, 1);
            
            % Initialize harmony memory with random solutions
            for i = 1:search_agents_no
                position = rand(1, obj.dim) .* (obj.ub - obj.lb) + obj.lb;
                fitness = obj.objective_func(position);
                obj.harmony_memory(i, :) = position;
                obj.harmony_fitness{i} = fitness(:)';
            end
            
            % Convert harmony memory to MultiObjectiveMember objects
            population = repmat(MultiObjectiveMember, 1, search_agents_no);
            for i = 1:search_agents_no
                population(i) = MultiObjectiveMember(obj.harmony_memory(i, :), obj.harmony_fitness{i});
            end
            
            % Initialize archive with non-dominated solutions
            obj.determine_domination(population);
            non_dominated = obj.get_non_dominated_particles(population);
            obj.archive = [obj.archive, non_dominated];
            
            % Initialize grid for archive
            costs = obj.get_fitness(obj.archive);
            if ~isempty(costs)
                obj.grid = obj.create_hypercubes(costs);
                for k = 1:numel(obj.archive)
                    [gi, gs] = obj.get_grid_index(obj.archive(k));
                    obj.archive(k).grid_index = gi;
                    obj.archive(k).grid_sub_index = gs;
                end
            end
            
            % Start solver
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
                
                % Create new member
                new_member = MultiObjectiveMember(new_harmony, new_fitness(:)');
                
                % Update harmony memory if new harmony is non-dominated
                % For multi-objective, we use archive-based replacement
                if ~isempty(obj.archive)
                    % Create temporary population including new member
                    temp_population = [population, new_member];
                    
                    % Determine domination
                    obj.determine_domination(temp_population);
                    
                    % Get non-dominated solutions
                    non_dominated_temp = obj.get_non_dominated_particles(temp_population);
                    
                    % If new member is non-dominated, add to archive
                    if any(arrayfun(@(x) isequal(x.position, new_member.position) && isequal(x.multi_fitness, new_member.multi_fitness), non_dominated_temp))
                        % Add to archive and trim if necessary
                        obj.archive = [obj.archive, new_member.copy()];
                        obj = obj.trim_archive();
                        
                        % Update harmony memory: replace a random harmony with new one
                        replace_idx = randi(search_agents_no);
                        obj.harmony_memory(replace_idx, :) = new_harmony;
                        obj.harmony_fitness{replace_idx} = new_fitness(:)';
                        population(replace_idx) = new_member;
                    end
                end
                
                % Update archive with current population
                obj = obj.add_to_archive(population);
                
                % Store archive state for history
                archive_copy = cell(1, length(obj.archive));
                for idx = 1:length(obj.archive)
                    archive_copy{idx} = obj.archive(idx).copy();
                end
                history_archive{end+1} = archive_copy;
                
                % Update progress
                if ~isempty(obj.archive)
                    best_member = obj.archive(1);
                else
                    best_member = [];
                end
                obj.callbacks(iter, max_iter, best_member);
            end
            
            % Final processing
            obj.history_step_solver = history_archive;
            obj.best_solver = obj.archive;
            
            % End solver
            obj.end_step_solver();
            
            archive = obj.archive;
        end
    end
end
