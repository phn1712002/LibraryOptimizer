classdef MultiObjectiveGreyWolfOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveGreyWolfOptimizer - Multi-Objective Grey Wolf Optimization Algorithm
    
    This algorithm extends the standard GWO for multi-objective optimization
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
        - archive_size: Size of the external archive (default: 100)
        - alpha: Grid inflation parameter (default: 0.1)
        - n_grid: Number of grids per dimension (default: 7)
        - beta: Leader selection pressure (default: 2)
        - gamma: Archive removal pressure (default: 2)
    %}
    
    methods
        function obj = MultiObjectiveGreyWolfOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveGreyWolfOptimizer constructor - Initialize the MOGWO solver
            
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
            obj.name_solver = "Multi-Objective Grey Wolf Optimizer";
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective GWO
            
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
                % Update a parameter (decreases linearly from 2 to 0)
                a = 2 - iter * (2 / max_iter);
                
                % Update all search agents
                for i = 1:length(population)
                    wolf = population(i);
                    new_position = zeros(1, obj.dim);
                    
                    % Select alpha, beta, and delta wolves from archive using grid-based selection
                    leaders = obj.select_multiple_leaders(3);
                    
                    % If we don't have enough leaders, use random wolves from population
                    if numel(leaders) < 3
                        % Get additional random wolves from population to make 3 leaders
                        available_wolves = population;
                        leader_positions = obj.getPositions(leaders);
                        if ~isempty(leader_positions)
                            available_wolves = available_wolves(~obj.rows_member_of(obj.getPositions(available_wolves), leader_positions));
                        end
                        
                        needed = 3 - numel(leaders);
                        if ~isempty(available_wolves)
                            additional_indices = randperm(numel(available_wolves), min(needed, numel(available_wolves)));
                            additional_leaders = available_wolves(additional_indices);
                            leaders = [leaders, additional_leaders];
                        end
                    end
                    
                    % Ensure we have exactly 3 leaders
                    if numel(leaders) > 3
                        leaders = leaders(1:3);
                    end
                    
                    % Update position using alpha, beta, and delta wolves
                    for j = 1:obj.dim
                        % Update position using each leader
                        total_contribution = 0;
                        for leader_idx = 1:numel(leaders)
                            leader = leaders(leader_idx);
                            r1 = rand();
                            r2 = rand();
                            
                            A = 2 * a * r1 - a;
                            C = 2 * r2;
                            
                            D = abs(C * leader.position(j) - wolf.position(j));
                            X = leader.position(j) - A * D;
                            
                            total_contribution = total_contribution + X;
                        end
                        % Average the contributions from all leaders
                        new_position(j) = total_contribution / numel(leaders);
                    end
                    
                    % Ensure positions stay within bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Update wolf position and fitness
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
