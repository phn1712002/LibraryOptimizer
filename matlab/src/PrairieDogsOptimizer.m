classdef PrairieDogsOptimizer < Solver
    %{
    Prairie Dogs Optimization (PDO) algorithm.
    
    Based on the MATLAB implementation from:
    Absalom E. Ezugwu, Jeffrey O. Agushaka, Laith Abualigah, Seyedali Mirjalili, Amir H Gandomi
    "Prairie Dogs Optimization: A Nature-inspired Metaheuristic"
    
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
        - rho: float (default=0.005) - Account for individual PD difference
        - eps_pd: float (default=0.1) - Food source alarm parameter
        - eps: float (default=1e-10) - Small epsilon value for numerical stability
        - beta: float (default=1.5) - Levy flight parameter
    %}
    
    properties
        rho       % Account for individual PD difference
        eps_pd    % Food source alarm parameter
        eps       % Small epsilon for numerical stability
        beta      % Levy flight parameter
    end
    
    methods
        function obj = PrairieDogsOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            PrairieDogsOptimizer constructor - Initialize the PDO solver
            
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
                    Additional PDO parameters:
                    - rho: float (default=0.005) - Account for individual PD difference
                    - eps_pd: float (default=0.1) - Food source alarm parameter
                    - eps: float (default=1e-10) - Small epsilon value for numerical stability
                    - beta: float (default=1.5) - Levy flight parameter
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set algorithm name
            obj.name_solver = "Prairie Dogs Optimizer";
            
            % kwargs (Name-Value pairs)
            if ~isempty(varargin)
                obj.kwargs = obj.nv2struct(varargin{:});
            end
            obj.rho = obj.get_kw('rho', 0.005);    % Account for individual PD difference
            obj.eps_pd = obj.get_kw('eps_pd', 0.1); % Food source alarm
            obj.eps = obj.get_kw('eps', 1e-10);    % Small epsilon for numerical stability
            obj.beta = obj.get_kw('beta', 1.5);    % Levy flight parameter
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for PDO algorithm
            
            Inputs:
                search_agents_no : int
                    Number of search agents (prairie dogs)
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
            
            % Initialize the population of prairie dogs
            population = obj.init_population(search_agents_no);
            
            % Initialize best solution
            sorted_population = obj.sort_population(population);
            best_solver = sorted_population{1}.copy();
            
            % Call the begin function
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Determine mu value based on iteration parity
                if mod(iter, 2) == 0
                    mu = -1;
                else
                    mu = 1;
                end
                
                % Calculate dynamic parameters
                DS = 1.5 * randn() * (1 - iter/max_iter) ^ (2 * iter/max_iter) * mu;  % Digging strength
                PE = 1.5 * (1 - iter/max_iter) ^ (2 * iter/max_iter) * mu;            % Predator effect
                
                % Generate Levy flight steps for all prairie dogs
                RL = zeros(search_agents_no, obj.dim);
                for i = 1:search_agents_no
                    RL(i, :) = obj.levy_flight();
                end
                
                % Create matrix of best positions for all prairie dogs
                TPD = repmat(best_solver.position, search_agents_no, 1);
                
                % Update each prairie dog's position
                for i = 1:search_agents_no
                    new_position = zeros(1, obj.dim);
                    
                    for j = 1:obj.dim
                        % Choose a random prairie dog different from current one
                        k = randi(search_agents_no);
                        while k == i
                            k = randi(search_agents_no);
                        end
                        
                        % Calculate PDO-specific parameters
                        cpd = rand() * (TPD(i, j) - population{k}.position(j)) / (TPD(i, j) + obj.eps);
                        P = obj.rho + (population{i}.position(j) - mean(population{i}.position)) / ...
                            (TPD(i, j) * (obj.ub(j) - obj.lb(j)) + obj.eps);
                        eCB = best_solver.position(j) * P;
                        
                        % Different position update strategies based on iteration phase
                        if iter < max_iter / 4
                            new_position(j) = best_solver.position(j) - eCB * obj.eps_pd - cpd * RL(i, j);
                        elseif iter < 2 * max_iter / 4
                            new_position(j) = best_solver.position(j) * population{k}.position(j) * DS * RL(i, j);
                        elseif iter < 3 * max_iter / 4
                            new_position(j) = best_solver.position(j) * PE * rand();
                        else
                            new_position(j) = best_solver.position(j) - eCB * obj.eps - cpd * rand();
                        end
                    end
                    
                    % Apply bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Evaluate new fitness
                    new_fitness = obj.objective_func(new_position);
                    
                    % Create new prairie dog candidate
                    new_prairie_dog = Member(new_position, new_fitness);
                    
                    % Greedy selection
                    if obj.is_better(new_prairie_dog, population{i})
                        population{i} = new_prairie_dog;
                    end
                    
                    % Update global best solution
                    if obj.is_better(population{i}, best_solver)
                        best_solver = population{i}.copy();
                    end
                end
                
                % Store the best solution at this iteration
                history_step_solver{end+1} = best_solver.copy();
                
                % Call the callbacks
                obj.callbacks(iter, max_iter, best_solver);
                
                % Print progress every 50 iterations
                if mod(iter, 50) == 0
                    fprintf('At iteration %d, the best solution fitness is %.6f\n', iter, best_solver.fitness);
                end
            end
            
            % Final evaluation and storage
            obj.history_step_solver = history_step_solver;
            obj.best_solver = best_solver;
            
            % Call the end function
            obj.end_step_solver();
        end
        
        function step = levy_flight(obj)
            %{
            levy_flight - Generate Levy flight step
            
            Returns:
                step : array
                    Levy flight step vector
            %}
            
            % Generate Levy flight step
            sigma = (gamma(1 + obj.beta) * sin(pi * obj.beta / 2) / ...
                    (gamma((1 + obj.beta) / 2) * obj.beta * 2^((obj.beta - 1)/2)))^(1/obj.beta);
            
            u = randn(1, obj.dim) * sigma;
            v = randn(1, obj.dim);
            step = u ./ (abs(v).^(1/obj.beta));
        end
        
        function sorted_population = sort_population(obj, population)
            %{
            sort_population - Sort population based on fitness
            
            Inputs:
                population : cell array
                    Population to sort
                    
            Returns:
                sorted_population : cell array
                    Sorted population (best first)
            %}
            
            % Extract fitness values
            fitness_values = obj.get_fitness(population);
            
            % Sort based on optimization direction
            if obj.maximize
                [~, sorted_indices] = sort(fitness_values, 'descend');
            else
                [~, sorted_indices] = sort(fitness_values, 'ascend');
            end
            
            % Sort population
            sorted_population = cell(1, length(population));
            for i = 1:length(population)
                sorted_population{i} = population{sorted_indices(i)};
            end
        end
    end
end