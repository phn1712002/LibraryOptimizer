classdef ArtificialBeeColonyOptimizer < Solver
    %{
    Artificial Bee Colony (ABC) Optimization Algorithm.
    
    ABC is a nature-inspired optimization algorithm based on the foraging behavior
    of honey bees. The algorithm classifies bees into three groups:
    1. Employed bees: Exploit food sources and share information
    2. Onlooker bees: Choose food sources based on information from employed bees
    3. Scout bees: Explore new food sources when current ones are exhausted
    
    References:
        Karaboga, D. (2005). An idea based on honey bee swarm for numerical optimization.
        Technical Report-TR06, Erciyes University, Engineering Faculty, Computer Engineering Department.
    %}
    
    properties
        n_onlooker
        abandonment_limit
        acceleration_coef
    end
    
    methods
        function obj = ArtificialBeeColonyOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            ArtificialBeeColonyOptimizer constructor - Initialize the ABC solver
            
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
                    Additional solver parameters including:
                    - n_onlooker: Number of onlooker bees (default: search_agents_no)
                    - abandonment_limit: Maximum trials before abandonment (default: dim * 5)
                    - acceleration_coef: Acceleration coefficient for search (default: 1.0)
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set solver name
            obj.name_solver = "Artificial Bee Colony Optimizer";
            
            % Set default ABC parameters
            obj.n_onlooker = obj.get_kw('n_onlooker', []);
            obj.abandonment_limit = obj.get_kw('abandonment_limit', []);
            obj.acceleration_coef = obj.get_kw('acceleration_coef', 1.0);
        end
        
        function population = init_population(obj, search_agents_no)
            %{
            init_population - Initialize the bee colony population
            
            Inputs:
                search_agents_no : int
                    Number of bees (employed bees) to initialize
                    
            Returns:
                population : cell array
                    Cell array of initialized Bee objects with random positions
            %}
            
            population = cell(1, search_agents_no);
            for i = 1:search_agents_no
                position = rand(1, obj.dim) .* (obj.ub - obj.lb) + obj.lb;
                fitness = obj.objective_func(position);
                population{i} = Bee(position, fitness, 0);
            end
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for ABC algorithm
            
            The algorithm consists of three main phases repeated for each iteration:
            1. Employed Bee Phase: Each employed bee searches for new solutions
            2. Onlooker Bee Phase: Onlooker bees probabilistically choose solutions
            3. Scout Bee Phase: Abandoned solutions are replaced with new random solutions
            
            Inputs:
                search_agents_no : int
                    Number of employed bees (and initial food sources)
                max_iter : int
                    Maximum number of iterations for the optimization process
                    
            Returns:
                Tuple containing:
                    - history_step_solver: Cell array of best solutions at each iteration
                    - best_solver: Best solution found overall
            %}
            
            % Initialize storage variables
            history_step_solver = {};
            
            % Initialize parameters
            if isempty(obj.n_onlooker)
                obj.n_onlooker = search_agents_no;  % Default: same as population size
            end
            
            if isempty(obj.abandonment_limit)
                % Default: 60% of variable dimension * population size
                obj.abandonment_limit = round(0.6 * obj.dim * search_agents_no);
            end
            
            % Initialize the population of bees
            population = obj.init_population(search_agents_no);
            
            % Initialize best solution
            [sorted_population, ~] = obj.sort_population(population);
            best_solver = sorted_population{1}.copy();
            
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
                    new_position = population{i}.position + phi .* (population{i}.position - population{k}.position);
                    
                    % Apply bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Evaluate new fitness
                    new_fitness = obj.objective_func(new_position);
                    
                    % Create new bee for comparison
                    new_bee = Bee(new_position, new_fitness);
                    
                    % Comparison (greedy selection)
                    if obj.is_better(new_bee, population{i})
                        population{i}.position = new_position;
                        population{i}.fitness = new_fitness;
                        population{i}.trial = 0;  % Reset trial counter
                    else
                        population{i}.trial = population{i}.trial + 1;  % Increase trial counter
                    end
                end
                
                % Calculate fitness values and selection probabilities
                fitness_values = zeros(1, search_agents_no);
                for i = 1:search_agents_no
                    fitness_values(i) = population{i}.fitness;
                end
                
                % Convert cost to fitness (for minimization problems, we need to invert)
                if ~obj.maximize
                    % For minimization, lower cost is better, so we use negative cost for fitness calculation
                    max_cost = max(fitness_values);
                    fitness_values = max_cost - fitness_values + 1e-10;  % Add small value to avoid division by zero
                end
                
                % Normalize to get probabilities
                probabilities = fitness_values / sum(fitness_values);
                
                % Phase 2: Onlooker Bees
                for m = 1:obj.n_onlooker
                    % Select source site using roulette wheel selection
                    i = obj.roulette_wheel_selection(probabilities);
                    
                    % Choose a random neighbor different from current bee
                    neighbors = setdiff(1:search_agents_no, i);
                    k = neighbors(randi(length(neighbors)));
                    
                    % Define acceleration coefficient
                    phi = obj.acceleration_coef * (2 * rand(1, obj.dim) - 1);
                    
                    % Generate new candidate solution
                    new_position = population{i}.position + phi .* (population{i}.position - population{k}.position);
                    
                    % Apply bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Evaluate new fitness
                    new_fitness = obj.objective_func(new_position);
                    
                    % Create new bee for comparison
                    new_bee = Bee(new_position, new_fitness);
                    
                    % Comparison (greedy selection)
                    if obj.is_better(new_bee, population{i})
                        population{i}.position = new_position;
                        population{i}.fitness = new_fitness;
                        population{i}.trial = 0;  % Reset trial counter
                    else
                        population{i}.trial = population{i}.trial + 1;  % Increase trial counter
                    end
                end
                
                % Phase 3: Scout Bees
                for i = 1:search_agents_no
                    if population{i}.trial >= obj.abandonment_limit
                        % Abandon and replace with random solution
                        position = rand(1, obj.dim) .* (obj.ub - obj.lb) + obj.lb;
                        fitness = obj.objective_func(position);
                        population{i} = Bee(position, fitness, 0);  % Reset trial counter
                    end
                end
                
                % Update best solution
                [sorted_population, ~] = obj.sort_population(population);
                current_best = sorted_population{1};
                if obj.is_better(current_best, best_solver)
                    best_solver = current_best.copy();
                end
                
                % Store the best solution at this iteration
                history_step_solver{end+1} = best_solver.copy();
                
                % Call the callbacks
                obj.callbacks(iter, max_iter, best_solver);
            end
            
            % Final evaluation and storage
            obj.history_step_solver = history_step_solver;
            obj.best_solver = best_solver;
            
            % Call the end function
            obj.end_step_solver();
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
        
        function selected_idx = roulette_wheel_selection(~, probabilities)
            %{
            roulette_wheel_selection - Perform roulette wheel selection
            
            Inputs:
                probabilities : array
                    Selection probabilities for each individual
                    
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
