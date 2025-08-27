classdef BiogeographyBasedOptimizer < Solver
    %{
    Biogeography-Based Optimization (BBO) algorithm.
    
    BBO is a population-based optimization algorithm inspired by the migration
    of species between habitats in biogeography. It models how species migrate
    between islands based on habitat suitability index (HSI).
    
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
        Additional algorithm parameters including:
        - keep_rate: Rate of habitats to keep (default: 0.2)
        - alpha: Migration coefficient (default: 0.9)
        - p_mutation: Mutation probability (default: 0.1)
        - sigma: Mutation step size (default: 2% of variable range)
    %}
    
    properties
        keep_rate    % Rate of habitats to keep
        alpha        % Migration coefficient
        p_mutation   % Mutation probability
        sigma        % Mutation step size
    end
    
    methods
        function obj = BiogeographyBasedOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            BiogeographyBasedOptimizer constructor - Initialize the BBO solver
            
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
                    Additional BBO parameters:
                    - keep_rate: Rate of habitats to keep (default: 0.2)
                    - alpha: Migration coefficient (default: 0.9)
                    - p_mutation: Mutation probability (default: 0.1)
                    - sigma: Mutation step size (default: 2% of variable range)
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set algorithm name
            obj.name_solver = "Biogeography Based Optimizer";
            
            % kwargs (Name-Value pairs)
            if ~isempty(varargin)
                obj.kwargs = obj.nv2struct(varargin{:});
            end
            obj.keep_rate = obj.get_kw('keep_rate', 0.2);
            obj.alpha = obj.get_kw('alpha', 0.9);
            obj.p_mutation = obj.get_kw('p_mutation', 0.1);
            
            % Calculate sigma as 2% of variable range if not provided
            var_range = obj.ub - obj.lb;
            obj.sigma = obj.get_kw('sigma', 0.02 * var_range);
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for BBO algorithm
            
            Inputs:
                search_agents_no : int
                    Number of habitats (population size)
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
            
            % Calculate derived parameters
            n_keep = round(obj.keep_rate * search_agents_no);  % Number of habitats to keep
            n_new = search_agents_no - n_keep;                 % Number of new habitats
            
            % Initialize migration rates (emigration and immigration)
            mu = linspace(1, 0, search_agents_no);      % Emigration rates (decreasing)
            lambda_rates = 1 - mu;                      % Immigration rates (increasing)
            
            % Initialize population
            population = obj.init_population(search_agents_no);
            
            % Sort initial population and get best solution
            [sorted_population, ~] = obj.sort_population(population);
            best_solver = sorted_population{1}.copy();
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Create new population for this iteration
                new_population = cell(1, search_agents_no);
                for i = 1:search_agents_no
                    new_population{i} = population{i}.copy();
                end
                
                % Update each habitat
                for i = 1:search_agents_no
                    % Migration phase for each dimension
                    for k = 1:obj.dim
                        % Immigration: if random number <= immigration rate
                        if rand() <= lambda_rates(i)
                            % Calculate emigration probabilities (excluding current habitat)
                            ep = mu;
                            ep(i) = 0;  % Set current habitat probability to 0
                            ep_sum = sum(ep);
                            
                            if ep_sum > 0
                                ep = ep / ep_sum;  % Normalize probabilities
                                
                                % Select source habitat using roulette wheel selection
                                j = obj.roulette_wheel_selection(ep);
                                
                                % Perform migration
                                new_population{i}.position(k) = (...
                                    population{i}.position(k) + ...
                                    obj.alpha * (population{j}.position(k) - population{i}.position(k))...
                                );
                            end
                        end
                        
                        % Mutation: if random number <= mutation probability
                        if rand() <= obj.p_mutation
                            new_population{i}.position(k) = new_population{i}.position(k) + ...
                                obj.sigma(k) * randn();
                        end
                    end
                    
                    % Apply bounds constraints
                    new_population{i}.position = max(min(new_population{i}.position, obj.ub), obj.lb);
                    
                    % Evaluate new fitness
                    new_population{i}.fitness = obj.objective_func(new_population{i}.position);
                end
                
                % Sort new population
                [sorted_new_population, ~] = obj.sort_population(new_population);
                
                % Select next iteration population: keep best + new solutions
                next_population = [sorted_population(1:n_keep), sorted_new_population(1:n_new)];
                
                % Sort the combined population
                [sorted_next_population, ~] = obj.sort_population(next_population);
                population = sorted_next_population;
                
                % Update best solution
                current_best = population{1};
                if obj.is_better(current_best, best_solver)
                    best_solver = current_best.copy();
                end
                
                % Save history
                history_step_solver{end+1} = best_solver.copy();
                
                % Call callback for progress tracking
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
