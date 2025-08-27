classdef AntColonyOptimizer < Solver
    % Ant Colony Optimization for Continuous Domains (ACOR).
    %
    % ACOR is a population-based metaheuristic algorithm inspired by the foraging
    % behavior of ants. It uses a solution archive and Gaussian sampling to
    % explore the search space.
    %
    % References:
    %     Socha, K., & Dorigo, M. (2008). Ant colony optimization for continuous domains.
    %     European Journal of Operational Research, 185(3), 1155-1173.
    %
    % Parameters:
    % -----------
    % objective_func : function handle
    %     Objective function to optimize
    % lb : float or array
    %     Lower bounds of search space
    % ub : float or array
    %     Upper bounds of search space
    % dim : int
    %     Number of dimensions in the problem
    % maximize : bool
    %     Whether to maximize (true) or minimize (false) objective
    % varargin : cell array
    %     Additional algorithm parameters including:
    %     - q: Intensification factor (selection pressure), default 0.5
    %     - zeta: Deviation-distance ratio, default 1.0
    
    properties
        q          % Intensification factor (selection pressure)
        zeta       % Deviation-distance ratio
    end
    
    methods
        function obj = AntColonyOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            AntColonyOptimizer constructor - Initialize the ACOR solver
            
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
                    Additional ACOR parameters:
                    - q: Intensification factor (default: 0.5)
                    - zeta: Deviation-distance ratio (default: 1.0)
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set algorithm name
            obj.name_solver = "Ant Colony Optimizer for Continuous Domains";
            
            % kwargs (Name-Value pairs)
            if ~isempty(varargin)
                obj.kwargs = obj.nv2struct(varargin{:});
            end
            obj.q = obj.get_kw('q', 0.5);      % Intensification factor
            obj.zeta = obj.get_kw('zeta', 1.0); % Deviation-distance ratio
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for ACOR algorithm
            
            Inputs:
                search_agents_no : int
                    Number of search agents (population/archive size)
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
            
            % Initialize population (archive)
            population = obj.init_population(search_agents_no);
            
            % Sort initial population
            [sorted_population, ~] = obj.sort_population(population);
            best_solver = sorted_population{1}.copy();
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Calculate solution weights (Gaussian kernel weights)
            w = obj.calculate_weights(search_agents_no);
            
            % Calculate selection probabilities
            p = w / sum(w);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Calculate means (positions of all solutions in archive)
                means = zeros(search_agents_no, obj.dim);
                for i = 1:search_agents_no
                    means(i, :) = population{i}.position;
                end
                
                % Calculate standard deviations for each solution
                sigma = obj.calculate_standard_deviations(means);
                
                % Create new population by sampling from Gaussian distributions
                new_population = obj.sample_new_population(means, sigma, p, search_agents_no);
                
                % Merge archive and new population
                merged_population = [population, new_population];
                
                % Sort merged population and keep only the best solutions
                [sorted_merged, ~] = obj.sort_population(merged_population);
                population = sorted_merged(1:search_agents_no);
                
                % Update best solution
                current_best = population{1};
                if obj.is_better(current_best, best_solver)
                    best_solver = current_best.copy();
                end
                
                % Save history
                history_step_solver{end+1} = best_solver.copy();
                
                % Call callback
                obj.callbacks(iter, max_iter, best_solver);
            end
            
            % End solver
            obj.history_step_solver = history_step_solver;
            obj.best_solver = best_solver;
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
        
        function w = calculate_weights(obj, n_pop)
            %{
            calculate_weights - Calculate Gaussian kernel weights for solution selection
            
            Inputs:
                n_pop : int
                    Population size
                    
            Returns:
                w : array
                    Array of weights for each solution
            %}
            
            w = (1 / (sqrt(2 * pi) * obj.q * n_pop)) * ...
                exp(-0.5 * (((1:n_pop) / (obj.q * n_pop)) .^ 2));
        end
        
        function sigma = calculate_standard_deviations(obj, means)
            %{
            calculate_standard_deviations - Calculate standard deviations for Gaussian sampling
            
            Inputs:
                means : array
                    Array of solution positions (means)
                    
            Returns:
                sigma : array
                    Array of standard deviations for each solution
            %}
            
            n_pop = size(means, 1);
            sigma = zeros(size(means));
            
            for l = 1:n_pop
                % Calculate average distance to other solutions
                D = sum(abs(means(l, :) - means), 1);
                sigma(l, :) = obj.zeta * D / (n_pop - 1);
            end
        end
        
        function new_population = sample_new_population(obj, means, sigma, probabilities, n_sample)
            %{
            sample_new_population - Sample new solutions using Gaussian distributions
            
            Inputs:
                means : array
                    Array of solution positions (means)
                sigma : array
                    Array of standard deviations
                probabilities : array
                    Selection probabilities for each solution
                n_sample : int
                    Number of samples to generate
                    
            Returns:
                new_population : cell array
                    Cell array of newly sampled solutions
            %}
            
            new_population = cell(1, n_sample);
            
            for sample_idx = 1:n_sample
                % Initialize new position
                new_position = zeros(1, obj.dim);
                
                % Construct solution component by component
                for i = 1:obj.dim
                    % Select Gaussian kernel using roulette wheel selection
                    l = obj.roulette_wheel_selection(probabilities);
                    
                    % Generate Gaussian random variable
                    new_position(i) = means(l, i) + sigma(l, i) * randn();
                end
                
                % Ensure positions stay within bounds
                new_position = max(min(new_position, obj.ub), obj.lb);
                
                % Evaluate fitness
                new_fitness = obj.objective_func(new_position);
                
                % Create new member
                new_population{sample_idx} = Member(new_position, new_fitness);
            end
        end
        
        function selected_idx = roulette_wheel_selection(~, probabilities)
            %{
            roulette_wheel_selection - Perform roulette wheel selection
            
            Inputs:
                probabilities : array
                    Selection probabilities for each solution
                    
            Returns:
                selected_idx : int
                    Index of the selected solution
            %}
            
            r = rand();
            cumulative_sum = cumsum(probabilities);
            selected_idx = find(r <= cumulative_sum, 1);
        end
    end
end
