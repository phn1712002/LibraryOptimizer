classdef MultiObjectiveArtificialBeeColonyOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveArtificialBeeColonyOptimizer - Multi-Objective Artificial Bee Colony Optimizer
    
    This algorithm extends the standard ABC for multi-objective optimization
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
        Additional parameters:
        - abandonment_limit: Trial limit for scout bees (default: calculated as 0.6 * dim * population_size)
        - n_onlooker: Number of onlooker bees (default: same as population size)
        - acceleration_coef: Acceleration coefficient (default: 1.0)
    %}
    
    properties
        acceleration_coef
        n_onlooker
        abandonment_limit
    end
    
    methods
        function obj = MultiObjectiveArtificialBeeColonyOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveArtificialBeeColonyOptimizer constructor - Initialize the MOABC solver
            
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
            obj.name_solver = "Multi-Objective Artificial Bee Colony Optimizer";
            
            % Set algorithm parameters with defaults
            obj.acceleration_coef = obj.get_kw('acceleration_coef', 1.0);
            obj.n_onlooker = obj.get_kw('n_onlooker', []);
            obj.abandonment_limit = obj.get_kw('abandonment_limit', []);
        end
        
        function population = init_population(obj, N)
            %{
            init_population - Initialize multi-objective bee population
            
            Inputs:
                N : int
                    Number of bees to initialize
                    
            Returns:
                population : object array
                    Array of BeeMultiMember objects
            %}
            pop_example = BeeMultiMember(0, 0, 0);
            population = repmat(pop_example, 1, N);
            for i = 1:N
                pos = obj.lb + (obj.ub - obj.lb).*rand(1, obj.dim);
                fit = obj.objective_func(pos); fit = fit(:).';
                population(i) = BeeMultiMember(pos, fit, 0);
            end
        end
        
        function normalized_costs = get_normalized_costs(obj, population)
            %{
            get_normalized_costs - Get normalized cost matrix from population for aggregation methods
            
            Inputs:
                population : object array
                    Population to normalize
                    
            Returns:
                normalized_costs : array
                    Normalized cost matrix
            %}
            
            costs = obj.get_fitness(population);
            normalized_costs = zeros(size(costs));
            
            for obj_idx = 1:obj.n_objectives
                obj_values = costs(:, obj_idx);
                min_val = min(obj_values);
                max_val = max(obj_values);
                
                if max_val ~= min_val
                    normalized_costs(:, obj_idx) = (obj_values - min_val) / (max_val - min_val);
                else
                    normalized_costs(:, obj_idx) = ones(size(obj_values));
                end
            end
        end
        
        function selected_idx = roulette_wheel_selection(~, probabilities)
            %{
            roulette_wheel_selection - Perform roulette wheel selection (fitness proportionate selection).
            
            Inputs:
                probabilities : array
                    Array of selection probabilities
                    
            Returns:
                selected_idx : int
                    Selected index based on roulette wheel
            %}
            
            cumulative = cumsum(probabilities);
            r = rand();
            selected_idx = find(cumulative >= r, 1);
            if isempty(selected_idx)
                selected_idx = length(probabilities);
            end
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective ABC
            
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
            
            % Set default number of onlooker bees
            if isempty(obj.n_onlooker)
                obj.n_onlooker = search_agents_no;
            end
            
            % Set default abandonment limit
            if isempty(obj.abandonment_limit)
                % Default: 60% of variable dimension * population size (as in MATLAB code)
                obj.abandonment_limit = round(0.6 * obj.dim * search_agents_no);
            end
            
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
                % Phase 1: Employed Bees
                bee_population = repmat(BeeMultiMember(0, 0, 0), 1, search_agents_no);
                for i = 1:search_agents_no
                    bee_population(i) = population(i).copy();
                end
                
                for i = 1:search_agents_no
                    % Choose a random neighbor different from current bee
                    neighbors = setdiff(1:search_agents_no, i);
                    k = neighbors(randi(length(neighbors)));
                    
                    % Define acceleration coefficient
                    phi = obj.acceleration_coef * (rand(1, obj.dim) * 2 - 1);
                    
                    % Generate new candidate solution using neighbor guidance
                    new_position = population(i).position + phi .* (population(i).position - population(k).position);
                    
                    % Apply bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Evaluate new fitness
                    new_fitness = obj.objective_func(new_position);
                    new_bee = BeeMultiMember(new_position, new_fitness);
                    
                    % Check if new solution dominates current solution
                    if obj.dominates(new_bee, population(i))
                        bee_population(i) = new_bee;
                        bee_population(i).trial = 0;
                    else
                        bee_population(i).trial = population(i).trial + 1;
                    end
                end
                
                % Update population
                population = bee_population;
                
                % Phase 2: Onlooker Bees
                % Calculate fitness values for selection probabilities
                % For multi-objective, we use a simple aggregation approach for selection
                % Sum of normalized objectives (assuming minimization for all objectives)
                normalized_costs = obj.get_normalized_costs(population);
                fitness_values = 1.0 ./ (sum(normalized_costs, 2) + 1e-10);
                
                % Normalize to get probabilities
                probabilities = fitness_values / sum(fitness_values);
                
                for onlooker = 1:obj.n_onlooker
                    % Select source site using roulette wheel selection
                    i = obj.roulette_wheel_selection(probabilities);
                    
                    % Choose a random neighbor different from current bee
                    neighbors = setdiff(1:search_agents_no, i);
                    k = neighbors(randi(length(neighbors)));
                    
                    % Define acceleration coefficient
                    phi = obj.acceleration_coef * (rand(1, obj.dim) * 2 - 1);
                    
                    % Generate new candidate solution using neighbor guidance
                    new_position = population(i).position + phi .* (population(i).position - population(k).position);
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Evaluate new fitness
                    new_fitness = obj.objective_func(new_position);
                    new_bee = BeeMultiMember(new_position, new_fitness);
                    
                    % Greedy selection
                    if obj.dominates(new_bee, population(i))
                        population(i) = new_bee;
                        population(i).trial = 0;
                    else
                        population(i).trial = population(i).trial + 1;
                    end
                end
                
                % Phase 3: Scout Bees
                for i = 1:search_agents_no
                    if population(i).trial >= obj.abandonment_limit
                        % Replace abandoned solution
                        position = obj.lb + (obj.ub - obj.lb) .* rand(1, obj.dim);
                        fitness = obj.objective_func(position);
                        population(i) = BeeMultiMember(position, fitness);
                        population(i).trial = 0;
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
