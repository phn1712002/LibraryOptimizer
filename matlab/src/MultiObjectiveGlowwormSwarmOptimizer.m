classdef MultiObjectiveGlowwormSwarmOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveGlowwormSwarmOptimizer - Multi-Objective Glowworm Swarm Optimization (GSO) Algorithm.
    
    This is the multi-objective version of the GSO algorithm that extends
    the single-objective implementation to handle multiple objectives using
    Pareto dominance and archive management.
    
    Key features:
    - Pareto dominance-based selection
    - Archive management for non-dominated solutions
    - Grid-based diversity maintenance
    - Luciferin-based communication adapted for multi-objective optimization
    
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
        Additional algorithm parameters including:
        - L0: Initial luciferin value (default: 5.0)
        - r0: Initial decision range (default: 3.0)
        - rho: Luciferin decay constant (default: 0.4)
        - gamma: Luciferin enhancement constant (default: 0.6)
        - beta: Decision range update constant (default: 0.08)
        - s: Step size for movement (default: 0.6)
        - rs: Maximum sensing range (default: 10.0)
        - nt: Desired number of neighbors (default: 10)
    %}
    
    properties
        L0
        r0
        rho
        gamma
        beta
        s
        rs
        nt
        luciferin
        decision_range
    end
    
    methods
        function obj = MultiObjectiveGlowwormSwarmOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveGlowwormSwarmOptimizer constructor - Initialize the MOGSO solver
            
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
            obj.name_solver = "Multi-Objective Glowworm Swarm Optimizer";
            
            % Set algorithm parameters with defaults
            obj.L0 = obj.get_kw('L0', 5.0);  % Initial luciferin
            obj.r0 = obj.get_kw('r0', 3.0);  % Initial decision range
            obj.rho = obj.get_kw('rho', 0.4);  % Luciferin decay constant
            obj.gamma = obj.get_kw('gamma', 0.6);  % Luciferin enhancement constant
            obj.beta = obj.get_kw('beta', 0.08);  % Decision range update constant
            obj.s = obj.get_kw('s', 0.6);  % Step size for movement
            obj.rs = obj.get_kw('rs', 10.0);  % Maximum sensing range
            obj.nt = obj.get_kw('nt', 10);  % Desired number of neighbors
            
            % Initialize internal state variables
            obj.luciferin = [];
            obj.decision_range = [];
        end
        
        function population = init_population(obj, N)
            %{
            init_population - Initialize multi-objective glowworm population
            
            Inputs:
                N : int
                    Number of glowworms to initialize
                    
            Returns:
                population : object array
                    Array of GlowwormMultiMember objects
            %}
            pop_example = GlowwormMultiMember(0, 0, 0, 0);
            population = repmat(pop_example, 1, N);
            for i = 1:N
                pos = obj.lb + (obj.ub - obj.lb).*rand(1, obj.dim);
                fit = obj.objective_func(pos); fit = fit(:).';
                population(i) = GlowwormMultiMember(pos, fit, obj.L0, obj.r0);
            end
        end
        
        function distance = euclidean_distance(obj, pos1, pos2)
            %{
            euclidean_distance - Calculate Euclidean distance between two positions.
            
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
            get_neighbors - Get indices of neighbors within decision range that dominate the current glowworm.
            
            Inputs:
                current_idx : int
                    Index of current glowworm
                population : object array
                    Array of all glowworms
                    
            Returns:
                neighbors : array
                    Indices of valid neighbors that dominate the current glowworm
            %}
            
            current_member = population(current_idx);
            current_pos = current_member.position;
            current_luciferin = obj.luciferin(current_idx);
            
            neighbors = [];
            for j = 1:length(population)
                if j == current_idx
                    continue;
                end
                
                neighbor_member = population(j);
                distance = obj.euclidean_distance(current_pos, neighbor_member.position);
                
                % Check if neighbor is within decision range and dominates current glowworm
                if (distance < obj.decision_range(current_idx) && ...
                    obj.dominates(neighbor_member, current_member) && ...
                    obj.luciferin(j) > current_luciferin)
                    neighbors = [neighbors, j];
                end
            end
        end
        
        function luciferin_value = calculate_luciferin(obj, member)
            %{
            calculate_luciferin - Calculate luciferin value for a member based on its multi-objective fitness.
            
            For multi-objective optimization, luciferin is calculated based on
            the member's position in the Pareto front and its diversity contribution.
            
            Inputs:
                member : GlowwormMultiMember
                    The member to calculate luciferin for
                    
            Returns:
                luciferin_value : float
                    Luciferin value
            %}
            
            % Simple approach: use the average of normalized fitness values
            if obj.maximize
                % For maximization: higher fitness values are better
                luciferin_value = mean(member.multi_fitness);
            else
                % For minimization: lower fitness values are better
                % We invert the values for consistency
                if ~isempty(obj.archive)
                    max_vals = zeros(1, obj.n_objectives);
                    for k = 1:length(obj.archive)
                        max_vals = max(max_vals, obj.archive(k).multi_fitness);
                    end
                    normalized = 1.0 ./ (member.multi_fitness + 1e-10);
                    luciferin_value = mean(normalized);
                else
                    normalized = 1.0 ./ (member.multi_fitness + 1e-10);
                    luciferin_value = mean(normalized);
                end
            end
        end
        
        function selected_idx = roulette_wheel_selection(obj, probabilities)
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
            solver - Execute the Multi-Objective Glowworm Swarm Optimization Algorithm.
            
            The algorithm extends the single-objective GSO with:
            - Pareto dominance for neighbor selection
            - Archive management for non-dominated solutions
            - Grid-based diversity maintenance
            
            Inputs:
                search_agents_no : int
                    Number of glowworms in the swarm
                max_iter : int
                    Maximum number of iterations for optimization
                    
            Returns:
                history_archive : cell array
                    History of archive states
                archive : cell array
                    Final archive of non-dominated solutions
            %}
            
            % Initialize the population of glowworms
            population = obj.init_population(search_agents_no);
            
            % Initialize luciferin and decision range arrays
            obj.luciferin = ones(1, search_agents_no) * obj.L0;
            obj.decision_range = ones(1, search_agents_no) * obj.r0;
            
            % Initialize archive with non-dominated solutions from initial population
            obj.determine_domination(population);
            non_dominated = obj.get_non_dominated_particles(population);
            obj.archive = [obj.archive, non_dominated];
            
            % Build initial grid for archive
            costs = obj.get_fitness(obj.archive);
            if ~isempty(costs)
                obj.grid = obj.create_hypercubes(costs);
                for k = 1:numel(obj.archive)
                    [gi, gs] = obj.get_grid_index(obj.archive(k));
                    obj.archive(k).grid_index = gi;
                    obj.archive(k).grid_sub_index = gs;
                end
            end
            
            % Initialize history storage
            history_archive = {};
            
            % Start solver execution
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Update luciferin for all glowworms based on their fitness
                for i = 1:search_agents_no
                    member = population(i);
                    obj.luciferin(i) = (1 - obj.rho) * obj.luciferin(i) + obj.gamma * obj.calculate_luciferin(member);
                end
                
                % Move each glowworm
                for i = 1:search_agents_no
                    % Get neighbors that dominate the current glowworm
                    neighbors = obj.get_neighbors(i, population);
                    
                    if isempty(neighbors)
                        % No dominating neighbors found, stay in current position
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
                    current_pos = population(i).position;
                    neighbor_pos = population(selected_neighbor).position;
                    
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
                    population(i).position = new_position;
                    population(i).multi_fitness = obj.objective_func(new_position);
                    
                    % Update decision range based on number of neighbors
                    neighbor_count = length(neighbors);
                    obj.decision_range(i) = min(obj.rs, max(0, obj.decision_range(i) + obj.beta * (obj.nt - neighbor_count)));
                end
                
                % Update archive with current population
                obj = obj.add_to_archive(population);
                
                % Store current archive state in history
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
            
            % Final storage
            obj.history_step_solver = history_archive;
            obj.best_solver = obj.archive;
            
            % End solver execution
            obj.end_step_solver();
            
            archive = obj.archive;
        end
    end
end
