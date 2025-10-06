classdef CuckooSearchOptimizer < Solver
    %{
    Cuckoo Search optimization algorithm.
    
    Cuckoo Search is a nature-inspired metaheuristic algorithm based on the 
    brood parasitism of some cuckoo species combined with Levy flight behavior.
    
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
        Additional algorithm parameters including:
        - pa: Discovery rate of alien eggs/solutions (default: 0.25)
        - beta: Levy exponent for flight steps (default: 1.5)
    %}
    
    properties
        pa
        beta
    end
    
    methods
        function obj = CuckooSearchOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            CuckooSearchOptimizer constructor - Initialize the Cuckoo Search Optimizer
            
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
            obj.name_solver = "Cuckoo Search Optimizer";
            
            % Set algorithm parameters with defaults
            obj.pa = obj.get_kw('pa', 0.25);  % Discovery rate
            obj.beta = obj.get_kw('beta', 1.5);  % Levy exponent
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for Cuckoo Search algorithm
            
            Parameters:
            -----------
            search_agents_no : int
                Number of nests (search agents)
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
            
            % Initialize population of nests
            population = obj.init_population(search_agents_no);
            
            % Find initial best solution
            [sorted_population, ~] = obj.sort_population(population);
            best_solver = sorted_population{1}.copy();
            
            % Call the begin function
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Generate new solutions via Levy flights (keep current best)
                new_population = obj.get_cuckoos(population, best_solver);
                
                % Evaluate new solutions and update population
                population = obj.update_population(population, new_population);
                
                % Discovery and randomization: abandon some nests and build new ones
                abandoned_nests = obj.empty_nests(population);
                
                % Evaluate abandoned nests and update population
                population = obj.update_population(population, abandoned_nests);
                
                % Update best solution
                [sorted_population, ~] = obj.sort_population(population);
                current_best = sorted_population{1};
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
        
        function new_population = get_cuckoos(obj, population, best_solver)
            %{
            get_cuckoos - Generate new solutions via Levy flights
            
            Inputs:
                population : cell array
                    Current population of nests
                best_solver : Member
                    Current best solution
                    
            Returns:
                new_population : cell array
                    New solutions generated via Levy flights
            %}
            
            new_population = cell(1, length(population));
            
            for i = 1:length(population)
                % Generate Levy flight step
                step = obj.levy_flight();
                
                % Scale step size (0.01 factor as in original implementation)
                step_size = 0.01 .* step .* (population{i}.position - best_solver.position);
                
                % Generate new position
                new_position = population{i}.position + step_size .* randn(1, obj.dim);
                
                % Apply bounds
                new_position = max(min(new_position, obj.ub), obj.lb);
                
                % Evaluate fitness
                new_fitness = obj.objective_func(new_position);
                
                % Create new member
                new_population{i} = Member(new_position, new_fitness);
            end
        end
        
        function new_nests = empty_nests(obj, population)
            %{
            empty_nests - Discover and replace abandoned nests
            
            Inputs:
                population : cell array
                    Current population of nests
                    
            Returns:
                new_nests : cell array
                    New nests to replace abandoned ones
            %}
            
            n = length(population);
            new_nests = cell(1, n);
            
            % Create discovery status vector
            discovery_status = rand(1, n) > obj.pa;
            
            for i = 1:n
                if discovery_status(i)
                    % This nest is discovered and will be abandoned
                    % Generate new solution via random walk
                    idx1 = randi(n);
                    idx2 = randi(n);
                    while idx2 == idx1
                        idx2 = randi(n);
                    end
                    
                    step_size = rand() * (population{idx1}.position - population{idx2}.position);
                    
                    new_position = population{i}.position + step_size;
                    
                    % Apply bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Evaluate fitness
                    new_fitness = obj.objective_func(new_position);
                    
                    new_nests{i} = Member(new_position, new_fitness);
                else
                    % Keep the original nest
                    new_nests{i} = population{i}.copy();
                end
            end
        end
        
        function updated_population = update_population(obj, current_population, new_population)
            %{
            update_population - Update population by keeping better solutions
            
            Inputs:
                current_population : cell array
                    Current population
                new_population : cell array
                    Newly generated population
                    
            Returns:
                updated_population : cell array
                    Updated population with better solutions
            %}
            
            updated_population = cell(1, length(current_population));
            
            for i = 1:length(current_population)
                if obj.is_better(new_population{i}, current_population{i})
                    updated_population{i} = new_population{i};
                else
                    updated_population{i} = current_population{i};
                end
            end
        end
        
        function step = levy_flight(obj)
            %{
            levy_flight - Generate Levy flight step
            
            Returns:
                step : array
                    Levy flight step vector
            %}
            
            % Generate Levy flight step using Mantegna's algorithm
            sigma_u = (gamma(1 + obj.beta) * sin(pi * obj.beta / 2) / ...
                      (gamma((1 + obj.beta) / 2) * obj.beta * 2^((obj.beta - 1) / 2)))^(1 / obj.beta);
            
            u = randn(1, obj.dim) * sigma_u;
            v = randn(1, obj.dim);
            
            step = u ./ (abs(v).^(1 / obj.beta));
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
