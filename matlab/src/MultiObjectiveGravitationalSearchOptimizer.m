classdef MultiObjectiveGravitationalSearchOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveGravitationalSearchOptimizer - Multi-Objective Gravitational Search Algorithm (GSA) Optimizer
    
    Reference: Rashedi, Esmat, Hossein Nezamabadi-Pour, and Saeid Saryazdi. "GSA: a gravitational search algorithm." 
               Information sciences 179.13 (2009): 2232-2248.
    
    Multi-objective version of GSA that maintains an archive of non-dominated solutions
    and uses grid-based selection for leader guidance.
    
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
        Additional algorithm parameters:
        - elitist_check: Whether to use elitist strategy (default: true)
        - r_power: Power parameter for distance calculation (default: 1)
        - g0: Initial gravitational constant (default: 100)
        - alpha: Decay parameter for gravitational constant (default: 20)
    %}
    
    properties
        elitist_check
        r_power
        g0
        alpha
        velocities
    end
    
    methods
        function obj = MultiObjectiveGravitationalSearchOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveGravitationalSearchOptimizer constructor - Initialize the MOGSA solver
            
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
            obj.name_solver = "Multi-Objective Gravitational Search Optimizer";
            
            % Set GSA-specific parameters with defaults
            obj.elitist_check = obj.get_kw('elitist_check', true);
            obj.r_power = obj.get_kw('r_power', 1);
            obj.g0 = obj.get_kw('g0', 100);
            obj.alpha = obj.get_kw('alpha', 20);
            
            % Initialize velocity storage
            obj.velocities = [];
        end
        
        function population = init_population(obj, N)
            %{
            init_population - Initialize multi-objective gravitational search population
            
            Inputs:
                N : int
                    Number of particles to initialize
                    
            Returns:
                population : object array
                    Array of GravitationalSearchMultiMember objects
            %}
            pop_example = GravitationalSearchMultiMember(0, 0, 0);
            population = repmat(pop_example, 1, N);
            for i = 1:N
                pos = obj.lb + (obj.ub - obj.lb).*rand(1, obj.dim);
                fit = obj.objective_func(pos); fit = fit(:).';
                population(i) = GravitationalSearchMultiMember(pos, fit);
            end
        end
        
        function masses = mass_calculation(obj, fitness_matrix)
            %{
            mass_calculation - Calculate masses for all agents based on their multi-objective fitness.
            
            Inputs:
                fitness_matrix : array
                    Matrix of fitness values for all agents (pop_size x n_objectives)
                    
            Returns:
                masses : array
                    Normalized mass values for all agents
            %}
            
            pop_size = size(fitness_matrix, 1);
            
            % For multi-objective, calculate a composite fitness
            % Use the sum of normalized objectives as a simple approach
            normalized_fitness = zeros(pop_size, 1);
            
            for obj_idx = 1:obj.n_objectives
                obj_values = fitness_matrix(:, obj_idx);
                min_val = min(obj_values);
                max_val = max(obj_values);
                
                if max_val ~= min_val
                    normalized_obj = (obj_values - min_val) / (max_val - min_val);
                else
                    normalized_obj = ones(pop_size, 1);
                end
                
                normalized_fitness = normalized_fitness + normalized_obj;
            end
            
            % Calculate masses based on composite fitness (lower is better for minimization)
            best_val = min(normalized_fitness);
            worst_val = max(normalized_fitness);
            
            if best_val == worst_val
                masses = ones(pop_size, 1) / pop_size;
                return;
            end
            
            masses = (normalized_fitness - worst_val) / (best_val - worst_val);
            mass_sum = sum(masses);
            
            if mass_sum > 0
                masses = masses / mass_sum;
            end
        end
        
        function g = gravitational_constant(obj, iteration, max_iter)
            %{
            gravitational_constant - Calculate gravitational constant for current iteration.
            
            Inputs:
                iteration : int
                    Current iteration number
                max_iter : int
                    Maximum number of iterations
                    
            Returns:
                g : float
                    Gravitational constant value
            %}
            
            gimd = exp(-obj.alpha * double(iteration) / max_iter);
            g = obj.g0 * gimd;
        end
        
        function accelerations = gravitational_field(obj, population, masses, iteration, max_iter, g)
            %{
            gravitational_field - Calculate gravitational forces and accelerations for all agents.
            
            Inputs:
                population : object array
                    Current population
                masses : array
                    Mass values for all agents
                iteration : int
                    Current iteration number
                max_iter : int
                    Maximum number of iterations
                g : float
                    Gravitational constant
                    
            Returns:
                accelerations : array
                    Acceleration matrix for all agents
            %}
            
            pop_size = length(population);
            positions = zeros(pop_size, obj.dim);
            for i = 1:pop_size
                positions(i, :) = population(i).position;
            end
            
            % Determine kbest (number of best agents to consider)
            final_percent = 2;  % Minimum percentage of best agents
            if obj.elitist_check
                kbest_percent = final_percent + (1 - iteration / max_iter) * (100 - final_percent);
                kbest = round(pop_size * kbest_percent / 100);
            else
                kbest = pop_size;
            end
            
            kbest = max(1, min(kbest, pop_size));  % Ensure kbest is within valid range
            
            % Sort agents by their composite fitness (for leader selection)
            composite_fitness = zeros(pop_size, 1);
            for i = 1:pop_size
                composite_fitness(i) = sum(population(i).multi_fitness);
            end
            [~, sorted_indices] = sort(composite_fitness);  % Lower is better for minimization
            
            % Initialize force matrix
            forces = zeros(pop_size, obj.dim);
            
            for i = 1:pop_size
                for j = 1:kbest
                    agent_idx = sorted_indices(j);
                    if agent_idx == i
                        continue;  % Skip self-interaction
                    end
                    
                    % Calculate Euclidean distance
                    distance = norm(positions(i, :) - positions(agent_idx, :));
                    
                    % Avoid division by zero
                    if distance < 1e-10
                        distance = 1e-10;
                    end
                    
                    % Calculate force components
                    for d = 1:obj.dim
                        rand_val = rand();
                        force_component = rand_val * masses(agent_idx) * ...
                                        (positions(agent_idx, d) - positions(i, d)) / ...
                                        (distance ^ obj.r_power + eps);
                        forces(i, d) = forces(i, d) + force_component;
                    end
                end
            end
            
            % Calculate accelerations
            accelerations = forces * g;
        end
        
        function [new_population, velocities] = update_positions(obj, population, accelerations)
            %{
            update_positions - Update positions and velocities of all agents.
            
            Inputs:
                population : object array
                    Current population
                accelerations : array
                    Acceleration matrix
                    
            Returns:
                new_population : object array
                    Updated population
                velocities : array
                    Updated velocities
            %}
            
            pop_size = length(population);
            positions = zeros(pop_size, obj.dim);
            for i = 1:pop_size
                positions(i, :) = population(i).position;
            end
            
            % Initialize velocities if not already done
            if isempty(obj.velocities)
                obj.velocities = zeros(pop_size, obj.dim);
            end
            
            % Update velocities and positions
            for i = 1:pop_size
                for d = 1:obj.dim
                    rand_val = rand();
                    obj.velocities(i, d) = rand_val * obj.velocities(i, d) + accelerations(i, d);
                    positions(i, d) = positions(i, d) + obj.velocities(i, d);
                end
            end
            
            % Ensure positions stay within bounds
            positions = max(min(positions, obj.ub), obj.lb);
            
            % Update population with new positions and recalculate fitness
            new_population = repmat(GravitationalSearchMultiMember(0, 0, 0), 1, pop_size);
            for i = 1:pop_size
                new_position = positions(i, :);
                fitness = obj.objective_func(new_position);
                new_population(i) = GravitationalSearchMultiMember(...
                    new_position, fitness, obj.velocities(i, :) ...
                );
            end
            
            velocities = obj.velocities;
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for Multi-Objective Gravitational Search Algorithm.
            
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
            
            % Initialize velocities
            obj.velocities = zeros(search_agents_no, obj.dim);
            
            % Initialize archive with non-dominated solutions
            obj.determine_domination(population);
            non_dominated = obj.get_non_dominated_particles(population);
            obj.archive = [obj.archive, non_dominated];
            
            % Initialize grid for archive management
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
                % Extract fitness values for mass calculation
                fitness_matrix = zeros(search_agents_no, obj.n_objectives);
                for i = 1:search_agents_no
                    fitness_matrix(i, :) = population(i).multi_fitness;
                end
                
                % Calculate masses
                masses = obj.mass_calculation(fitness_matrix);
                
                % Calculate gravitational constant
                g = obj.gravitational_constant(iter, max_iter);
                
                % Calculate gravitational field and accelerations
                accelerations = obj.gravitational_field(population, masses, iter, max_iter, g);
                
                % Update positions
                [population, obj.velocities] = obj.update_positions(population, accelerations);
                
                % Update archive with current population
                obj = obj.add_to_archive(population);
                
                % Trim archive if it exceeds maximum size
                if length(obj.archive) > obj.archive_size
                    obj = obj.trim_archive();
                end
                
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
