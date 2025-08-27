classdef BatOptimizer < Solver
    %{
    Bat Algorithm Optimizer implementation.
    
    The Bat Algorithm is a metaheuristic optimization algorithm inspired by 
    the echolocation behavior of microbats. It uses frequency tuning, loudness,
    and pulse emission rate to control the search process.
    
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
    maximize : bool, optional
        Optimization direction, default is True (maximize)
    varargin : cell array
        Additional algorithm parameters:
        - fmin: Minimum frequency (default: 0)
        - fmax: Maximum frequency (default: 2)
        - alpha: Loudness decay constant (default: 0.9)
        - gamma: Pulse rate increase constant (default: 0.9)
        - ro: Initial pulse emission rate (default: 0.5)
    %}
    
    properties
        fmin
        fmax
        alpha
        gamma
        ro
    end
    
    methods
        function obj = BatOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            BatOptimizer constructor - Initialize the Bat Optimizer
            
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
            obj.name_solver = "Bat Optimizer";
            
            % Set default BAT parameters
            obj.fmin = obj.get_kw('fmin', 0.0);          % Minimum frequency
            obj.fmax = obj.get_kw('fmax', 2.0);          % Maximum frequency
            obj.alpha = obj.get_kw('alpha', 0.9);        % Loudness decay constant
            obj.gamma = obj.get_kw('gamma', 0.9);        % Pulse rate increase constant
            obj.ro = obj.get_kw('ro', 0.5);              % Initial pulse emission rate
        end
        
        function population = init_population(obj, search_agents_no)
            %{
            init_population - Initialize population of bats with bat-specific parameters
            
            Inputs:
                search_agents_no : int
                    Number of bats to initialize
                    
            Returns:
                population : cell array
                    Cell array of initialized BatMember objects
            %}
            
            population = cell(1, search_agents_no);
            for i = 1:search_agents_no
                position = rand(1, obj.dim) .* (obj.ub - obj.lb) + obj.lb;
                fitness = obj.objective_func(position);
                % Initialize with default bat parameters
                population{i} = BatMember(...
                    position, ...
                    fitness, ...
                    0.0, ...                    % frequency
                    zeros(1, obj.dim), ...      % velocity
                    1.0, ...                    % loudness
                    obj.ro ...                  % pulse_rate
                );
            end
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for the Bat Algorithm
            
            Parameters:
            -----------
            search_agents_no : int
                Number of bats (search agents)
            max_iter : int
                Maximum number of iterations
                
            Returns:
            --------
            Tuple containing:
                - history_step_solver: Cell array of best solutions at each iteration
                - best_solver: Best solution found overall
            %}
            
            % Initialize storage variables
            history_step_solver = {};
            
            % Initialize population
            population = obj.init_population(search_agents_no);
            
            % Initialize best solution
            [sorted_population, ~] = obj.sort_population(population);
            best_solver = sorted_population{1}.copy();
            
            % Call the begin function
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Update each bat
                for i = 1:search_agents_no
                    % Update frequency
                    population{i}.frequency = obj.fmin + (obj.fmax - obj.fmin) * rand();
                    
                    % Update velocity
                    population{i}.velocity = population{i}.velocity + ...
                        (population{i}.position - best_solver.position) * population{i}.frequency;
                    
                    % Update position
                    new_position = population{i}.position + population{i}.velocity;
                    
                    % Apply boundary constraints
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Random walk with probability (1 - pulse_rate)
                    if rand() > population{i}.pulse_rate
                        % Generate random walk step
                        epsilon = -1 + 2 * rand();
                        % Calculate mean loudness of all bats
                        loudness_values = zeros(1, search_agents_no);
                        for j = 1:search_agents_no
                            loudness_values(j) = population{j}.loudness;
                        end
                        mean_loudness = mean(loudness_values);
                        new_position = best_solver.position + epsilon * mean_loudness;
                        new_position = max(min(new_position, obj.ub), obj.lb);
                    end
                    
                    % Evaluate new fitness
                    new_fitness = obj.objective_func(new_position);
                    
                    % Create temporary BatMember for comparison
                    new_bat = BatMember(...
                        new_position, ...
                        new_fitness, ...
                        population{i}.frequency, ...
                        population{i}.velocity, ...
                        population{i}.loudness, ...
                        population{i}.pulse_rate ...
                    );
                    
                    % Update if solution improves and meets loudness criteria
                    if obj.is_better(new_bat, population{i}) && rand() < population{i}.loudness
                        % Update position and fitness
                        population{i}.position = new_position;
                        population{i}.fitness = new_fitness;
                        
                        % Update loudness and pulse rate
                        population{i}.loudness = obj.alpha * population{i}.loudness;
                        population{i}.pulse_rate = obj.ro * (1 - exp(-obj.gamma * iter));
                    end
                    
                    % Update best solution if improved
                    if obj.is_better(population{i}, best_solver)
                        best_solver = population{i}.copy();
                    end
                end
                
                % Save history
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
    end
end
