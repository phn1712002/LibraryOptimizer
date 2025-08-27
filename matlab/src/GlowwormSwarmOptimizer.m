classdef GlowwormSwarmOptimizer < Solver
    %{
    Glowworm Swarm Optimization (GSO) Algorithm.
    
    GSO is a nature-inspired metaheuristic optimization algorithm that mimics
    the behavior of glowworms (fireflies) that use bioluminescence to attract
    mates and prey. Each glowworm carries a luminescent quantity called luciferin,
    which they use to communicate with other glowworms in their neighborhood.
    
    Key features:
    - Luciferin-based communication
    - Dynamic neighborhood topology
    - Adaptive decision range
    - Probabilistic movement towards brighter neighbors
    
    References:
        Krishnanand, K. N., & Ghose, D. (2009). Glowworm swarm optimization: 
        a new method for optimizing multi-modal functions. International Journal 
        of Computational Intelligence Studies, 1(1), 93-119.
    %}
    
    properties
        L0     % Initial luciferin value
        r0     % Initial decision range
        rho    % Luciferin decay constant
        gamma  % Luciferin enhancement constant
        beta   % Decision range update constant
        s      % Step size for movement
        rs     % Maximum sensing range
        nt     % Desired number of neighbors
        
        % Internal state variables
        luciferin
        decision_range
    end
    
    methods
        function obj = GlowwormSwarmOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            GlowwormSwarmOptimizer constructor - Initialize the GSO solver
            
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
                    Additional GSO parameters:
                    - L0: Initial luciferin value (default: 5.0)
                    - r0: Initial decision range (default: 3.0)
                    - rho: Luciferin decay constant (default: 0.4)
                    - gamma: Luciferin enhancement constant (default: 0.6)
                    - beta: Decision range update constant (default: 0.08)
                    - s: Step size for movement (default: 0.6)
                    - rs: Maximum sensing range (default: 10.0)
                    - nt: Desired number of neighbors (default: 10)
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set solver name
            obj.name_solver = "Glowworm Swarm Optimizer";
            
            % Algorithm-specific parameters with defaults
            % kwargs (Name-Value pairs)
            if ~isempty(varargin)
                obj.kwargs = obj.nv2struct(varargin{:});
            end
            obj.L0 = obj.get_kw('L0', 5.0);     % Initial luciferin
            obj.r0 = obj.get_kw('r0', 3.0);     % Initial decision range
            obj.rho = obj.get_kw('rho', 0.4);   % Luciferin decay constant
            obj.gamma = obj.get_kw('gamma', 0.6); % Luciferin enhancement constant
            obj.beta = obj.get_kw('beta', 0.08); % Decision range update constant
            obj.s = obj.get_kw('s', 0.6);       % Step size for movement
            obj.rs = obj.get_kw('rs', 10.0);    % Maximum sensing range
            obj.nt = obj.get_kw('nt', 10);      % Desired number of neighbors
        end
        
        function distance = euclidean_distance(~, pos1, pos2)
            %{
            euclidean_distance - Calculate Euclidean distance between two positions
            
            Inputs:
                pos1 : array
                    First position vector
                pos2 : array
                    Second position vector
                    
            Returns:
                distance : float
                    Euclidean distance between the two positions
            %}
            
            distance = sqrt(sum((pos1 - pos2) .^ 2));
        end
        
        function neighbors = get_neighbors(obj, current_idx, population)
            %{
            get_neighbors - Get indices of neighbors within decision range that have higher luciferin
            
            Inputs:
                current_idx : int
                    Index of current glowworm
                population : cell array
                    List of all glowworms
                    
            Returns:
                neighbors : array
                    Indices of valid neighbors
            %}
            
            current_pos = population{current_idx}.position;
            current_luciferin = obj.luciferin(current_idx);
            
            neighbors = [];
            for j = 1:length(population)
                if j == current_idx
                    continue;
                end
                
                distance = obj.euclidean_distance(current_pos, population{j}.position);
                if distance < obj.decision_range(current_idx) && obj.luciferin(j) > current_luciferin
                    neighbors = [neighbors, j];
                end
            end
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for GSO algorithm
            
            The algorithm consists of three main phases:
            1. Luciferin update: Each glowworm updates its luciferin based on fitness
            2. Movement: Each glowworm probabilistically moves towards brighter neighbors
            3. Decision range update: Each glowworm adjusts its sensing range
            
            Inputs:
                search_agents_no : int
                    Number of glowworms in the swarm
                max_iter : int
                    Maximum number of iterations for optimization
                    
            Returns:
                history_step_solver : cell array
                    History of best solutions at each iteration
                best_solver : Member
                    Best solution found overall
            %}
            
            % Initialize the population of glowworms
            population = obj.init_population(search_agents_no);
            
            % Initialize luciferin and decision range
            obj.luciferin = ones(1, search_agents_no) * obj.L0;
            obj.decision_range = ones(1, search_agents_no) * obj.r0;
            
            % Initialize storage variables
            history_step_solver = {};
            
            % Initialize best solution
            sorted_population = obj.sort_population(population);
            best_solver = sorted_population{1}.copy();
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Update luciferin for all glowworms
                fitness_values = obj.get_fitness(population);
                
                % Convert fitness to luciferin (handle maximization/minimization)
                if obj.maximize
                    % For maximization: higher fitness = higher luciferin
                    luciferin_update = obj.gamma * fitness_values;
                else
                    % For minimization: lower fitness = higher luciferin
                    % We need to invert the fitness for minimization
                    max_fitness = max(fitness_values);
                    luciferin_update = obj.gamma * (max_fitness - fitness_values + 1e-10);
                end
                
                obj.luciferin = (1 - obj.rho) * obj.luciferin + luciferin_update;
                
                % Find the best glowworm (highest luciferin)
                [~, best_idx] = max(obj.luciferin);
                current_best = population{best_idx}.copy();
                
                % Update best solution if better
                if obj.is_better(current_best, best_solver)
                    best_solver = current_best.copy();
                end
                
                % Move each glowworm
                for i = 1:search_agents_no
                    % Get neighbors within decision range that have higher luciferin
                    neighbors = obj.get_neighbors(i, population);
                    
                    if isempty(neighbors)
                        % No neighbors found, stay in current position
                        continue;
                    end
                    
                    % Calculate probabilities for movement towards each neighbor
                    neighbor_luciferin = obj.luciferin(neighbors);
                    current_luciferin = obj.luciferin(i);
                    
                    % Probability proportional to luciferin difference
                    probabilities = (neighbor_luciferin - current_luciferin) / sum(neighbor_luciferin - current_luciferin);
                    
                    % Select a neighbor using roulette wheel selection
                    selected_neighbor_idx = obj.roulette_wheel_selection(probabilities);
                    selected_neighbor = neighbors(selected_neighbor_idx);
                    
                    % Move towards the selected neighbor
                    current_pos = population{i}.position;
                    neighbor_pos = population{selected_neighbor}.position;
                    
                    % Calculate direction vector
                    direction = neighbor_pos - current_pos;
                    distance = obj.euclidean_distance(current_pos, neighbor_pos);
                    
                    if distance > 0
                        % Normalize direction and move with step size s
                        direction_normalized = direction / distance;
                        new_position = current_pos + obj.s * direction_normalized;
                    else
                        % If distance is zero, add small random perturbation
                        new_position = current_pos + obj.s * (rand(1, obj.dim) * 0.2 - 0.1);
                    end
                    
                    % Ensure positions stay within bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Update glowworm position and fitness
                    population{i}.position = new_position;
                    population{i}.fitness = obj.objective_func(new_position);
                    
                    % Update decision range based on number of neighbors
                    neighbor_count = length(neighbors);
                    obj.decision_range(i) = min(obj.rs, max(0, obj.decision_range(i) + obj.beta * (obj.nt - neighbor_count)));
                end
                
                % Store the best solution at this iteration
                history_step_solver{end+1} = best_solver.copy();
                
                % Update progress
                obj.callbacks(iter, max_iter, best_solver);
            end
            
            % Final evaluation and storage
            obj.history_step_solver = history_step_solver;
            obj.best_solver = best_solver;
            
            % End solver
            obj.end_step_solver();
        end
        
        function selected_idx = roulette_wheel_selection(~, probabilities)
            %{
            roulette_wheel_selection - Perform roulette wheel selection (fitness proportionate selection)
            
            Inputs:
                probabilities : array
                    Array of selection probabilities
                    
            Returns:
                selected_idx : int
                    Index of selected element
            %}
            
            cumulative_prob = cumsum(probabilities);
            r = rand();
            selected_idx = find(cumulative_prob >= r, 1);
            if isempty(selected_idx)
                selected_idx = 1;
            end
        end
        
        function [sorted_population, sorted_indices] = sort_population(obj, population)
            %{
            sort_population - Sort the population based on fitness
            
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
    end
end
