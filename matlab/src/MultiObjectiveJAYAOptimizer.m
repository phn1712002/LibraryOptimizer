classdef MultiObjectiveJAYAOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveJAYAOptimizer - Multi-Objective JAYA (To Win) Optimizer
    
    This algorithm extends the standard JAYA for multi-objective optimization
    using archive management and grid-based selection.
    
    Parameters:
    -----------
    objective_func : function handle
        Objective function that returns a list of fitness values
    lb : float or array
        Lower bounds for variables
    ub : float or array
        Upper bounds for variables
    dim : int
        Problem dimension
    maximize : bool
        Optimization direction (true for maximize, false for minimize)
    varargin : cell array
        Additional parameters
    %}
    
    methods
        function obj = MultiObjectiveJAYAOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveJAYAOptimizer constructor - Initialize the MOJAYA solver
            
            Inputs:
                objective_func : function handle
                    Multi-objective function to optimize (returns array for multiple objectives)
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
            obj.name_solver = "Multi-Objective JAYA Optimizer";
        end
        
        function leaders = select_multiple_leaders(obj, n_leaders)
            %{
            select_multiple_leaders - Select multiple diverse leaders from archive
            
            Inputs:
                n_leaders : int
                    Number of leaders to select
                    
            Returns:
                leaders : object array
                    Array of diverse leaders
            %}
            
            if isempty(obj.archive) || n_leaders <= 0
                leaders = repmat(MultiObjectiveMember(0, 0), 1, 0);
                return;
            end
            
            % Simple approach: select the first n_leaders from archive
            n_to_select = min(n_leaders, length(obj.archive));
            leaders = repmat(MultiObjectiveMember(0, 0), 1, n_to_select);
            
            for i = 1:n_to_select
                leaders(i) = obj.archive(i).copy();
            end
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective JAYA
            
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
                % Find best and worst solutions in current population
                obj.determine_domination(population);
                non_dominated_pop = obj.get_non_dominated_particles(population);
                
                % If we have non-dominated solutions, use them as leaders
                if ~isempty(non_dominated_pop)
                    % Select best and worst from non-dominated using grid-based selection
                    leaders = obj.select_multiple_leaders(2);  % Get 2 leaders
                    
                    if length(leaders) >= 2
                        best_member = leaders(1);
                        worst_member = leaders(2);
                    else
                        % Fallback: use random selection if not enough leaders
                        if length(non_dominated_pop) >= 2
                            indices = randperm(length(non_dominated_pop), 2);
                            best_member = non_dominated_pop(indices(1));
                            worst_member = non_dominated_pop(indices(2));
                        else
                            % If only one non-dominated, use it as best and random as worst
                            best_member = non_dominated_pop(1);
                            dominated_pop = population(~ismember(population, non_dominated_pop));
                            if ~isempty(dominated_pop)
                                worst_idx = randi(length(dominated_pop));
                                worst_member = dominated_pop(worst_idx);
                            else
                                worst_member = population(randi(length(population)));
                            end
                        end
                    end
                else
                    % If no non-dominated solutions, use random selection
                    indices = randperm(length(population), 2);
                    best_member = population(indices(1));
                    worst_member = population(indices(2));
                end
                
                % Update each search agent
                for i = 1:search_agents_no
                    % Create new position using JAYA formula
                    new_position = zeros(1, obj.dim);
                    for j = 1:obj.dim
                        rand1 = rand();
                        rand2 = rand();
                        new_position(j) = (...
                            population(i).position(j) + ...
                            rand1 * (best_member.position(j) - abs(population(i).position(j))) - ...
                            rand2 * (worst_member.position(j) - abs(population(i).position(j)))...
                        );
                    end
                    
                    % Apply bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Evaluate new fitness
                    new_fitness = obj.objective_func(new_position);
                    
                    % Create new member
                    new_member = MultiObjectiveMember(new_position, new_fitness);
                    
                    % Check if new solution dominates current solution
                    if obj.dominates(new_member, population(i))
                        population(i).position = new_position;
                        population(i).multi_fitness = new_fitness;
                        population(i).dominated = false;  % Reset domination status
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
                    best_archive_member = obj.archive(1);
                else
                    best_archive_member = [];
                end
                obj.callbacks(iter, max_iter, best_archive_member);
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
