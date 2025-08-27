classdef MultiObjectiveSeagullOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveSeagullOptimizer - Multi-Objective Seagull Optimization Algorithm (SOA).
    
    Multi-objective version of the Seagull Optimization Algorithm that mimics
    the migration and attacking behavior of seagulls for multi-objective optimization.
    
    References:
        Dhiman, G., & Kumar, V. (2019). Seagull optimization algorithm: 
        Theory and its applications for large-scale industrial engineering problems. 
        Knowledge-Based Systems, 165, 169-196.
    
    Parameters:
    -----------
    objective_func : function handle
        Multi-objective function that returns array of fitness values
    lb : float or array
        Lower bounds for variables
    ub : float or array
        Upper bounds for variables
    dim : int
        Problem dimension
    maximize : bool
        Optimization direction (true for maximize, false for minimize)
    varargin : cell array
        Additional algorithm parameters
    %}
    
    methods
        function obj = MultiObjectiveSeagullOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveSeagullOptimizer constructor - Initialize the MOSOA solver
            
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
            obj.name_solver = "Multi-Objective Seagull Optimization Algorithm";
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Execute the Multi-Objective Seagull Optimization Algorithm.
            
            The algorithm simulates the migration and attacking behavior of seagulls:
            1. Migration phase: Seagulls move towards the best position (exploration)
            2. Attacking phase: Seagulls attack prey using spiral movements (exploitation)
            
            Inputs:
                search_agents_no : int
                    Number of seagulls in the population
                max_iter : int
                    Maximum number of iterations for optimization
                    
            Returns:
                history_archive : cell array
                    History of archive states
                archive : cell array
                    Final archive of non-dominated solutions
            %}
            
            % Initialize the population of search agents
            population = obj.init_population(search_agents_no);

            % Initialize archive with non-dominated solutions
            obj.determine_domination(population);
            non_dominated = obj.get_non_dominated_particles(population);
            obj.archive = [obj.archive, non_dominated];
            
            % Build grid
            costs = obj.get_fitness(obj.archive);
            if ~isempty(costs)
                obj.grid = obj.create_hypercubes(costs);
                for k = 1:numel(obj.archive)
                    [gi, gs] = obj.get_grid_index(obj.archive(k));
                    obj.archive(k).grid_index = gi;
                    obj.archive(k).grid_sub_index = gs;
                end
            end
            
            % Initialize storage variables
            history_archive = {};
            
            % Call the begin function
            obj.begin_step_solver(max_iter);

            % Main optimization loop
            for iter = 1:max_iter
                % Update control parameter Fc (decreases linearly from 2 to 0)
                Fc = 2 - iter * (2 / max_iter);
                
                % Update all search agents
                for i = 1:search_agents_no
                    member = population(i);
                    new_position = zeros(1, obj.dim);
                    
                    % Select leader from archive
                    leader = obj.select_leader();
                    if isempty(leader)
                        % If no leader available, use random member
                        random_idx = randi(length(population));
                        leader = population(random_idx);
                    end
                    
                    for j = 1:obj.dim
                        % Generate random numbers
                        r1 = rand();
                        r2 = rand();
                        
                        % Calculate A1 and C1 parameters
                        A1 = 2 * Fc * r1 - Fc;
                        C1 = 2 * r2;
                        
                        % Calculate ll parameter
                        ll = (Fc - 1) * rand() + 1;
                        
                        % Calculate D_alphs (direction towards leader)
                        D_alphs = Fc * member.position(j) + A1 * (leader.position(j) - member.position(j));
                        
                        % Update position using spiral movement (attacking behavior)
                        X1 = D_alphs * exp(ll) * cos(ll * 2 * pi) + leader.position(j);
                        new_position(j) = X1;
                    end
                    
                    % Ensure positions stay within bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Update member position and fitness
                    member.position = new_position;
                    member.multi_fitness = obj.objective_func(new_position);
                end
                
                % Add non-dominated solutions to archive
                obj = obj.add_to_archive(population);
                
                % Store archive history
                archive_copy = cell(1, length(obj.archive));
                for idx = 1:length(obj.archive)
                    archive_copy{idx} = obj.archive(idx).copy();
                end
                history_archive{end+1} = archive_copy;
                
                % Call the callbacks 
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
            
            % Call the end function
            obj.end_step_solver();
            
            archive = obj.archive;
        end
    end
end
