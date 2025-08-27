classdef ArtificialEcosystemOptimizer < Solver
    %{
    Artificial Ecosystem-based Optimization (AEO) algorithm.
    
    A nature-inspired meta-heuristic algorithm based on energy flow in ecosystems.
    
    Parameters:
    -----------
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
        Additional algorithm parameters:
        - production_weight: Production phase weight (default: 1.0)
        - consumption_weight: Consumption phase weight (default: 1.0)
        - decomposition_weight: Decomposition phase weight (default: 1.0)
    %}
    
    properties
        production_weight    % Production phase weight
        consumption_weight   % Consumption phase weight
        decomposition_weight % Decomposition phase weight
    end
    
    methods
        function obj = ArtificialEcosystemOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            ArtificialEcosystemOptimizer constructor - Initialize the AEO solver
            
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
                    Additional AEO parameters:
                    - production_weight: Production phase weight (default: 1.0)
                    - consumption_weight: Consumption phase weight (default: 1.0)
                    - decomposition_weight: Decomposition phase weight (default: 1.0)
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set algorithm name
            obj.name_solver = "Artificial Ecosystem Optimizer";
            
            % kwargs (Name-Value pairs)
            if ~isempty(varargin)
                obj.kwargs = obj.nv2struct(varargin{:});
            end
            obj.production_weight = obj.get_kw('production_weight', 1.0);
            obj.consumption_weight = obj.get_kw('consumption_weight', 1.0);
            obj.decomposition_weight = obj.get_kw('decomposition_weight', 1.0);
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for AEO algorithm
            
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
            
            % Initialize the population
            population = obj.init_population(search_agents_no);
            
            % Sort population and get best solution
            [sorted_population, ~] = obj.sort_population(population);
            best_solver = sorted_population{1}.copy();
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Production phase: Create new organism based on best and random position
                new_population = obj.production_phase(population, iter, max_iter);
                
                % Consumption phase: Update organisms based on consumption behavior
                new_population = obj.consumption_phase(new_population, population);
                
                % Decomposition phase: Update organisms based on decomposition behavior
                new_population = obj.decomposition_phase(new_population, best_solver);
                
                % Evaluate new population and update
                population = obj.evaluate_and_update(population, new_population);
                
                % Update best solution
                [sorted_population, ~] = obj.sort_population(population);
                current_best = sorted_population{1};
                if obj.is_better(current_best, best_solver)
                    best_solver = current_best.copy();
                end
                
                % Store the best solution at this iteration
                history_step_solver{end+1} = best_solver.copy();
                
                % Call callback
                obj.callbacks(iter, max_iter, best_solver);
            end
            
            % End solver
            obj.history_step_solver = history_step_solver;
            obj.best_solver = best_solver;
            obj.end_step_solver();
        end
        
        function new_population = production_phase(obj, population, iter, max_iter)
            %{
            production_phase - Create new organism based on best and random position
            
            Inputs:
                population : cell array
                    Current population
                iter : int
                    Current iteration
                max_iter : int
                    Maximum number of iterations
                    
            Returns:
                new_population : cell array
                    New population after production phase
            %}
            
            new_population = {};
            
            % Get sorted population
            [sorted_population, ~] = obj.sort_population(population);
            
            % Create random position in search space
            random_position = obj.lb + (obj.ub - obj.lb) .* rand(1, obj.dim);
            
            % Calculate production weight (decreases linearly)
            r1 = rand();
            a = (1 - iter / max_iter) * r1;
            
            % Create first organism: combination of best and random position
            best_position = sorted_population{end}.position;  % Worst becomes producer
            new_position = (1 - a) * best_position + a * random_position;
            new_position = max(min(new_position, obj.ub), obj.lb);
            new_fitness = obj.objective_func(new_position);
            new_population{1} = Member(new_position, new_fitness);
        end
        
        function new_population = consumption_phase(obj, new_population, old_population)
            %{
            consumption_phase - Update organisms based on consumption behavior
            
            Inputs:
                new_population : cell array
                    Population from production phase
                old_population : cell array
                    Original population
                    
            Returns:
                new_population : cell array
                    Population after consumption phase
            %}
            
            % Get sorted population
            [sorted_old_population, ~] = obj.sort_population(old_population);
            
            % Handle second organism (special case)
            if length(old_population) >= 2
                % Generate consumption factor C using Levy flight
                u = randn(1, obj.dim);
                v = randn(1, obj.dim);
                C = 0.5 * u ./ abs(v);
                
                % Second organism consumes from producer (first organism)
                new_position = old_population{2}.position + C .* (...
                    old_population{2}.position - new_population{1}.position...
                );
                
                % Apply bounds
                new_position = max(min(new_position, obj.ub), obj.lb);
                new_fitness = obj.objective_func(new_position);
                new_population{2} = Member(new_position, new_fitness);
            end
            
            % For remaining organisms (starting from third one)
            for i = 3:length(old_population)
                % Generate consumption factor C using Levy flight
                u = randn(1, obj.dim);
                v = randn(1, obj.dim);
                C = 0.5 * u ./ abs(v);
                
                r = rand();
                
                if r < 1/3
                    % Consume from producer (first organism)
                    new_position = old_population{i}.position + C .* (...
                        old_population{i}.position - new_population{1}.position...
                    );
                elseif r < 2/3
                    % Consume from random consumer (between 1 and i-1)
                    random_idx = randi([2, i-1]);
                    new_position = old_population{i}.position + C .* (...
                        old_population{i}.position - old_population{random_idx}.position...
                    );
                else
                    % Consume from both producer and random consumer
                    r2 = rand();
                    random_idx = randi([2, i-1]);
                    new_position = old_population{i}.position + C .* (...
                        r2 * (old_population{i}.position - new_population{1}.position) + ...
                        (1 - r2) * (old_population{i}.position - old_population{random_idx}.position)...
                    );
                end
                
                % Apply bounds
                new_position = max(min(new_position, obj.ub), obj.lb);
                new_fitness = obj.objective_func(new_position);
                new_population{i} = Member(new_position, new_fitness);
            end
        end
        
        function new_population = decomposition_phase(obj, population, best_solver)
            %{
            decomposition_phase - Update organisms based on decomposition behavior
            
            Inputs:
                population : cell array
                    Current population
                best_solver : Member
                    Best solution found so far
                    
            Returns:
                new_population : cell array
                    Population after decomposition phase
            %}
            
            new_population = {};
            
            % Find the best organism in current population
            [sorted_population, ~] = obj.sort_population(population);
            best_current = sorted_population{1};
            
            for i = 1:length(population)
                % Generate decomposition factors
                r3 = rand();
                % Random dimension selection (0 or 1)
                dim_choice = randi([0, 1]);
                weight_factor = 3 * randn();
                
                % Calculate new position using decomposition equation
                random_multiplier = randi([1, 2]);  % This gives 1 or 2
                new_position = best_solver.position + weight_factor * (...
                    (r3 * random_multiplier - 1) * best_solver.position - ...
                    (2 * r3 - 1) * population{i}.position...
                );
                
                % Apply bounds
                new_position = max(min(new_position, obj.ub), obj.lb);
                new_fitness = obj.objective_func(new_position);
                new_population{i} = Member(new_position, new_fitness);
            end
        end
        
        function updated_population = evaluate_and_update(obj, old_population, new_population)
            %{
            evaluate_and_update - Evaluate new population and update if better solutions are found
            
            Inputs:
                old_population : cell array
                    Original population
                new_population : cell array
                    New population after all phases
                    
            Returns:
                updated_population : cell array
                    Updated population
            %}
            
            updated_population = {};
            
            for i = 1:length(old_population)
                if i <= length(new_population) && obj.is_better(new_population{i}, old_population{i})
                    updated_population{i} = new_population{i}.copy();
                else
                    updated_population{i} = old_population{i}.copy();
                end
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
