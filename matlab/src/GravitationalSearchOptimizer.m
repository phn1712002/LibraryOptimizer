classdef GravitationalSearchOptimizer < Solver
    %{
    Gravitational Search Algorithm (GSA) Optimizer
    
    Reference: Rashedi, Esmat, Hossein Nezamabadi-Pour, and Saeid Saryazdi. "GSA: a gravitational search algorithm." 
               Information sciences 179.13 (2009): 2232-2248.
    
    GSA is a population-based optimization algorithm inspired by the law of gravity and mass interactions.
    Each solution is considered as a mass, and their interactions are governed by gravitational forces.
    
    Parameters:
    -----------
    objective_func : function handle
        Objective function to optimize
    lb : float or array
        Lower bounds for variables
    ub : float or array
        Upper bounds for variables
    dim : int
        Problem dimension
    maximize : bool
        Whether to maximize (true) or minimize (false) objective
    varargin : cell array
        Additional algorithm parameters:
        - elitist_check: Whether to use elitist strategy (default: true)
        - r_power: Power parameter for distance calculation (default: 1)
        - g0: Initial gravitational constant (default: 100)
        - alpha: Decay parameter for gravitational constant (default: 20)
    %}
    
    properties
        elitist_check  % Whether to use elitist strategy
        r_power        % Power parameter for distance calculation
        g0             % Initial gravitational constant
        alpha          % Decay parameter for gravitational constant
        
        % Internal state variables
        velocities
    end
    
    methods
        function obj = GravitationalSearchOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            GravitationalSearchOptimizer constructor - Initialize the GSA solver
            
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
                    Additional GSA parameters:
                    - elitist_check: Whether to use elitist strategy (default: true)
                    - r_power: Power parameter for distance calculation (default: 1)
                    - g0: Initial gravitational constant (default: 100)
                    - alpha: Decay parameter for gravitational constant (default: 20)
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set solver name
            obj.name_solver = "Gravitational Search Optimizer";
            
            % Algorithm-specific parameters with defaults
            % kwargs (Name-Value pairs)
            if ~isempty(varargin)
                obj.kwargs = obj.nv2struct(varargin{:});
            end
            obj.elitist_check = obj.get_kw('elitist_check', true);  % Elitist strategy
            obj.r_power = obj.get_kw('r_power', 1);                 % Power parameter
            obj.g0 = obj.get_kw('g0', 100);                         % Initial gravitational constant
            obj.alpha = obj.get_kw('alpha', 20);                    % Decay parameter
        end
        
        function masses = mass_calculation(obj, fitness_values)
            %{
            mass_calculation - Calculate masses for all agents based on their fitness
            
            Inputs:
                fitness_values : array
                    Array of fitness values for all agents
                    
            Returns:
                masses : array
                    Normalized mass values for all agents
            %}
            
            pop_size = length(fitness_values);
            
            if obj.maximize
                % For maximization, higher fitness is better
                best_val = max(fitness_values);
                worst_val = min(fitness_values);
            else
                % For minimization, lower fitness is better
                best_val = min(fitness_values);
                worst_val = max(fitness_values);
            end
            
            if best_val == worst_val
                % All agents have same fitness
                masses = ones(1, pop_size) / pop_size;
                return;
            end
            
            % Calculate raw masses
            masses = zeros(1, pop_size);
            for i = 1:pop_size
                masses(i) = (fitness_values(i) - worst_val) / (best_val - worst_val);
            end
            
            % Normalize masses
            mass_sum = sum(masses);
            if mass_sum > 0
                masses = masses / mass_sum;
            end
        end
        
        function g = gravitational_constant(obj, iteration, max_iter)
            %{
            gravitational_constant - Calculate gravitational constant for current iteration
            
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
            gravitational_field - Calculate gravitational forces and accelerations for all agents
            
            Inputs:
                population : cell array
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
            positions = obj.get_positions(population);
            
            % Determine kbest (number of best agents to consider)
            final_percent = 2;  % Minimum percentage of best agents
            if obj.elitist_check
                kbest_percent = final_percent + (1 - iteration / max_iter) * (100 - final_percent);
                kbest = round(pop_size * kbest_percent / 100);
            else
                kbest = pop_size;
            end
            
            kbest = max(1, min(kbest, pop_size));  % Ensure kbest is within valid range
            
            % Sort agents by fitness (best first for maximization, worst first for minimization)
            fitness_values = obj.get_fitness(population);
            if obj.maximize
                [~, sorted_indices] = sort(fitness_values, 'descend');
            else
                [~, sorted_indices] = sort(fitness_values, 'ascend');
            end
            
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
            update_positions - Update positions and velocities of all agents
            
            Inputs:
                population : cell array
                    Current population
                accelerations : array
                    Acceleration matrix
                    
            Returns:
                new_population : cell array
                    Updated population
                velocities : array
                    Updated velocities
            %}
            
            pop_size = length(population);
            positions = obj.get_positions(population);
            
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
            new_population = cell(1, pop_size);
            for i = 1:pop_size
                new_position = positions(i, :);
                fitness = obj.objective_func(new_position);
                new_population{i} = Member(new_position, fitness);
            end
            
            velocities = obj.velocities;
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for Gravitational Search Algorithm
            
            Inputs:
                search_agents_no : int
                    Number of search agents (population size)
                max_iter : int
                    Maximum number of iterations
                    
            Returns:
                history_step_solver : cell array
                    History of best solutions at each iteration
                best_solver : Member
                    Best solution found overall
            %}
            
            % Initialize storage variables
            history_step_solver = {};
            
            % Initialize population
            population = obj.init_population(search_agents_no);
            
            % Initialize best solution
            sorted_population = obj.sort_population(population);
            best_solver = sorted_population{1}.copy();
            
            % Initialize velocities
            obj.velocities = zeros(search_agents_no, obj.dim);
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iteration = 1:max_iter
                % Extract fitness values for mass calculation
                fitness_values = obj.get_fitness(population);
                
                % Calculate masses
                masses = obj.mass_calculation(fitness_values);
                
                % Calculate gravitational constant
                g = obj.gravitational_constant(iteration, max_iter);
                
                % Calculate gravitational field and accelerations
                accelerations = obj.gravitational_field(population, masses, iteration, max_iter, g);
                
                % Update positions
                [population, obj.velocities] = obj.update_positions(population, accelerations);
                
                % Update best solution
                sorted_population = obj.sort_population(population);
                current_best = sorted_population{1};
                if obj.is_better(current_best, best_solver)
                    best_solver = current_best.copy();
                end
                
                % Save history
                history_step_solver{end+1} = best_solver.copy();
                
                % Update progress
                obj.callbacks(iteration, max_iter, best_solver);
            end
            
            % Final evaluation and storage
            obj.history_step_solver = history_step_solver;
            obj.best_solver = best_solver;
            
            % End solver
            obj.end_step_solver();
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
