classdef MultiObjectiveArtificialBeeColonyOptimizer < MultiObjectiveSolver
    %{
    Multi-Objective Artificial Bee Colony (MOABC) Optimization Algorithm.
    
    MOABC extends the ABC algorithm for multi-objective optimization problems.
    It maintains an archive of non-dominated solutions and uses Pareto dominance
    for solution comparison and selection.
    
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
        - n_onlooker: Number of onlooker bees (default: search_agents_no)
        - abandonment_limit: Maximum trials before abandonment (default: dim * 5)
        - acceleration_coef: Acceleration coefficient for search (default: 1.0)
        - archive_size: Maximum size of Pareto archive (default: 100)
    %}
    
    properties
        n_onlooker
        abandonment_limit
        acceleration_coef
        archive_size
    end
    
    methods
        function obj = MultiObjectiveArtificialBeeColonyOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveArtificialBeeColonyOptimizer constructor
            
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
            obj.name_solver = "Multi-Objective Artificial Bee Colony Optimizer";
            
            % Set default MOABC parameters
            obj.n_onlooker = obj.get_kw('n_onlooker', []);
            obj.abandonment_limit = obj.get_kw('abandonment_limit', []);
            obj.acceleration_coef = obj.get_kw('acceleration_coef', 1.0);
            obj.archive_size = obj.get_kw('archive_size', 100);
        end
        
        function population = init_population(obj, search_agents_no)
            %{
            init_population - Initialize the bee colony population
            
            Inputs:
                search_agents_no : int
                    Number of bees to initialize
                    
            Returns:
                population : array
                    Array of initialized MultiObjectiveMember objects
            %}
            
            population = MultiObjectiveMember.empty(search_agents_no, 0);
            for i = 1:search_agents_no
                position = rand(1, obj.dim) .* (obj.ub - obj.lb) + obj.lb;
                fitness = obj.objective_func(position);
                population(i) = MultiObjectiveMember(position, fitness);
            end
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for MOABC algorithm
            
            Inputs:
                search_agents_no : int
                    Number of employed bees
                max_iter : int
                    Maximum number of iterations
                    
            Returns:
                Tuple containing:
                    - history_archive: Cell array of archive states at each iteration
                    - archive: Final Pareto archive
            %}
            
            % Initialize storage variables
            history_archive = {};
            
            % Initialize parameters
            if isempty(obj.n_onlooker)
                obj.n_onlooker = search_agents_no;
            end
            
            if isempty(obj.abandonment_limit)
                obj.abandonment_limit = round(0.6 * obj.dim * search_agents_no);
            end
            
            % Initialize population and archive
            population = obj.init_population(search_agents_no);
            archive = obj.initialize_archive(population, obj.archive_size);
            
            % Call the begin function
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Phase 1: Employed Bees
                for i = 1:search_agents_no
                    % Choose a random neighbor different from current bee
                    neighbors = setdiff(1:search_agents_no, i);
                    k = neighbors(randi(length(neighbors)));
                    
                    % Define acceleration coefficient
                    phi = obj.acceleration_coef * (2 * rand(1, obj.dim) - 1);
                    
                    % Generate new candidate solution
                    new_position = population(i).position + phi .* (population(i).position - population(k).position);
                    
                    % Apply bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Evaluate new fitness
                    new_fitness = obj.objective_func(new_position);
                    new_bee = MultiObjectiveMember(new_position, new_fitness);
                    
                    % Comparison using Pareto dominance
                    if obj.dominates(new_bee, population(i))
                        population(i).position = new_position;
                        population(i).fitness = new_fitness;
                        population(i).trial = 0;
                    else
                        population(i).trial = population(i).trial + 1;
                    end
                    
                    % Add to archive if non-dominated
                    archive = obj.add_to_archive(new_bee, archive, obj.archive_size);
                end
                
                % Calculate selection probabilities based on crowding distance
                crowding_distances = obj.calculate_crowding_distance(archive);
                probabilities = crowding_distances / sum(crowding_distances);
                
                % Phase 2: Onlooker Bees
                for m = 1:obj.n_onlooker
                    % Select source site using roulette wheel selection
                    i = obj.roulette_wheel_selection(probabilities);
                    
                    % Choose a random neighbor
                    neighbors = setdiff(1:search_agents_no, i);
                    k = neighbors(randi(length(neighbors)));
                    
                    % Define acceleration coefficient
                    phi = obj.acceleration_coef * (2 * rand(1, obj.dim) - 1);
                    
                    % Generate new candidate solution
                    new_position = population(i).position + phi .* (population(i).position - population(k).position);
                    
                    % Apply bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Evaluate new fitness
                    new_fitness = obj.objective_func(new_position);
                    new_bee = MultiObjectiveMember(new_position, new_fitness);
                    
                    % Comparison using Pareto dominance
                    if obj.dominates(new_bee, population(i))
                        population(i).position = new_position;
                        population(i).fitness = new_fitness;
                        population(i).trial = 0;
                    else
                        population(i).trial = population(i).trial + 1;
                    end
                    
                    % Add to archive if non-dominated
                    archive = obj.add_to_archive(new_bee, archive, obj.archive_size);
                end
                
                % Phase 3: Scout Bees
                for i = 1:search_agents_no
                    if population(i).trial >= obj.abandonment_limit
                        % Abandon and replace with random solution
                        position = rand(1, obj.dim) .* (obj.ub - obj.lb) + obj.lb;
                        fitness = obj.objective_func(position);
                        population(i) = MultiObjectiveMember(position, fitness);
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
        
        function selected_idx = roulette_wheel_selection(~, probabilities)
            %{
            roulette_wheel_selection - Perform roulette wheel selection
            
            Inputs:
                probabilities : array
                    Selection probabilities
                    
            Returns:
                selected_idx : int
                    Index of selected individual
            %}
            
            r = rand();
            cumulative_sum = cumsum(probabilities);
            selected_idx = find(r <= cumulative_sum, 1);
        end
    end
end
