classdef FireflyOptimizer < Solver
    %{
    Firefly Algorithm Optimizer
    
    Implementation based on the MATLAB Firefly Algorithm by Xin-She Yang.
    Fireflies are attracted to each other based on their brightness (fitness),
    with attractiveness decreasing with distance.
    
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
        Optimization direction (true: maximize, false: minimize)
    varargin : cell array
        Additional algorithm parameters:
        - alpha: Randomness parameter (default: 0.5)
        - betamin: Minimum attractiveness (default: 0.2)
        - gamma: Absorption coefficient (default: 1.0)
        - alpha_reduction: Whether to reduce alpha over iterations (default: true)
        - alpha_delta: Alpha reduction factor (default: 0.97)
    %}
    properties
        alpha          
        betamin      
        gamma         
        alpha_reduction         
        alpha_delta
        alpha_initial
    end

    methods
        function obj = FireflyOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            FireflyOptimizer constructor - Initialize the Firefly Algorithm solver
            
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
                    Additional solver parameters
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set solver name
            obj.name_solver = "Firefly Optimizer";
            
            % Set default Firefly Algorithm parameters
            obj.alpha = obj.get_kw('alpha', 0.5);  % Randomness parameter
            obj.betamin = obj.get_kw('betamin', 0.2);  % Minimum attractiveness
            obj.gamma = obj.get_kw('gamma', 1.0);  % Absorption coefficient
            obj.alpha_reduction = obj.get_kw('alpha_reduction', true);  % Reduce alpha over time
            obj.alpha_delta = obj.get_kw('alpha_delta', 0.97);  % Alpha reduction factor
            
            % Store initial alpha for reference
            obj.alpha_initial = obj.alpha;
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for Firefly Algorithm
            
            Parameters:
            -----------
            search_agents_no : int
                Number of fireflies in the population
            max_iter : int
                Maximum number of iterations
                
            Returns:
            --------
            history_step_solver : cell array
                History of best solutions at each iteration
            best_solver : Member
                Best solution found overall
            %}
            
            % Initialize storage variables
            history_step_solver = {};
            
            % Initialize the population of fireflies
            population = obj.init_population(search_agents_no);
            
            % Initialize best solution
            [sorted_population, ~] = obj.sort_population(population);
            best_solver = sorted_population{1}.copy();
            
            % Call the begin function
            obj.begin_step_solver(max_iter);
            
            % Calculate scale for random movement
            scale = abs(obj.ub - obj.lb);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Evaluate all fireflies
                for i = 1:search_agents_no
                    population{i}.fitness = obj.objective_func(population{i}.position);
                end
                
                % Sort fireflies by brightness (fitness)
                [sorted_population, ~] = obj.sort_population(population);
                
                % Update best solution
                current_best = sorted_population{1};
                if obj.is_better(current_best, best_solver)
                    best_solver = current_best.copy();
                end
                
                % Move all fireflies towards brighter ones
                for i = 1:search_agents_no
                    for j = 1:search_agents_no
                        % Firefly i moves towards firefly j if j is brighter
                        if obj.is_better(population{j}, population{i})
                            % Calculate distance between fireflies
                            r = sqrt(sum((population{i}.position - population{j}.position).^2));
                            
                            % Calculate attractiveness
                            beta = obj.calculate_attractiveness(r);
                            
                            % Generate random movement
                            random_move = obj.alpha * (rand(1, obj.dim) - 0.5) .* scale;
                            
                            % Update position
                            new_position = (population{i}.position * (1 - beta) + ...
                                           population{j}.position * beta + ...
                                           random_move);
                            
                            % Apply bounds
                            new_position = max(min(new_position, obj.ub), obj.lb);
                            
                            % Update firefly position
                            population{i}.position = new_position;
                        end
                    end
                end
                
                % Store the best solution at this iteration
                history_step_solver{end+1} = best_solver.copy();
                
                % Reduce alpha (randomness) over iterations if enabled
                if obj.alpha_reduction
                    obj.alpha = obj.reduce_alpha(obj.alpha, obj.alpha_delta);
                end
                
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
            sort_population - Sort population based on fitness (brightness)
            
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
        
        function beta = calculate_attractiveness(obj, distance)
            %{
            calculate_attractiveness - Calculate attractiveness based on distance between fireflies
            
            Parameters:
            -----------
            distance : float
                Euclidean distance between two fireflies
                
            Returns:
            --------
            beta : float
                Attractiveness value
            %}
            
            beta0 = 1.0;  % Attractiveness at distance 0
            beta = (beta0 - obj.betamin) * exp(-obj.gamma * distance^2) + obj.betamin;
        end
        
        function new_alpha = reduce_alpha(obj, current_alpha, delta)
            %{
            reduce_alpha - Reduce the randomness parameter alpha over iterations
            
            Parameters:
            -----------
            current_alpha : float
                Current alpha value
            delta : float
                Reduction factor
                
            Returns:
            --------
            new_alpha : float
                Reduced alpha value
            %}
            
            new_alpha = current_alpha * delta;
        end
    end
end
