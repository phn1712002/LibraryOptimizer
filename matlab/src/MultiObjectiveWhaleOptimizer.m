classdef MultiObjectiveWhaleOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveWhaleOptimizer - Multi-Objective Whale Optimization Algorithm
    
    This algorithm extends the standard WOA for multi-objective optimization
    using archive management and grid-based selection for leader selection.
    
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
        Additional parameters:
    %}
    
    methods
        function obj = MultiObjectiveWhaleOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveWhaleOptimizer constructor - Initialize the MOWOA solver
            
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
            obj.name_solver = "Multi-Objective Whale Optimizer";
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective WOA
            
            Inputs:
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
            
            % Initialize population
            population = obj.init_population(search_agents_no);
            
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
                % Update a parameters (decreases linearly)
                a = 2 - iter * (2 / max_iter);
                a2 = -1 + iter * ((-1) / max_iter);
                
                % Update all search agents
                for i = 1:length(population)
                    whale = population(i);
                    new_position = zeros(1, obj.dim);
                    
                    % Select leader from archive using grid-based selection
                    leader = obj.select_leader();
                    
                    % If no leader in archive, use random whale from population
                    if isempty(leader)
                        random_idx = randi(length(population));
                        leader = population(random_idx);
                    end
                    
                    % Update position for each dimension
                    for j = 1:obj.dim
                        r1 = rand();
                        r2 = rand();
                        
                        A = 2 * a * r1 - a;  % Eq. (2.3) in the paper
                        C = 2 * r2;          % Eq. (2.4) in the paper
                        
                        b = 1;               % parameters in Eq. (2.5)
                        l = (a2 - 1) * rand() + 1;  % parameters in Eq. (2.5)
                        
                        p = rand();  % p in Eq. (2.6)
                        
                        if p < 0.5
                            if abs(A) >= 1
                                % Search for prey (exploration phase)
                                % Select random leader from archive for exploration
                                if ~isempty(obj.archive)
                                    random_idx = randi(length(obj.archive));
                                    rand_leader = obj.archive(random_idx);
                                    D_X_rand = abs(C * rand_leader.position(j) - whale.position(j));  % Eq. (2.7)
                                    new_position(j) = rand_leader.position(j) - A * D_X_rand;  % Eq. (2.8)
                                else
                                    % If archive is empty, use random whale from population
                                    random_idx = randi(length(population));
                                    rand_whale = population(random_idx);
                                    D_X_rand = abs(C * rand_whale.position(j) - whale.position(j));
                                    new_position(j) = rand_whale.position(j) - A * D_X_rand;
                                end
                            else
                                % Encircling prey (exploitation phase)
                                D_leader = abs(C * leader.position(j) - whale.position(j));  % Eq. (2.1)
                                new_position(j) = leader.position(j) - A * D_leader;  % Eq. (2.2)
                            end
                        else
                            % Bubble-net attacking method (spiral updating position)
                            distance_to_leader = abs(leader.position(j) - whale.position(j));
                            % Eq. (2.5) - spiral movement
                            new_position(j) = distance_to_leader * exp(b * l) * cos(l * 2 * pi) + leader.position(j);
                        end
                    end
                    
                    % Ensure positions stay within bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Update whale position and fitness
                    population(i).position = new_position;
                    population(i).multi_fitness = obj.objective_func(new_position);
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
