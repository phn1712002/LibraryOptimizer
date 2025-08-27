classdef MarinePredatorsOptimizer < Solver
    %{
    Marine Predators Algorithm (MPA) Optimizer.
    
    MPA is a nature-inspired metaheuristic optimization algorithm that mimics
    the optimal foraging strategy and encounter rate policy between predator 
    and prey in marine ecosystems. The algorithm follows three main phases:
    
    1. Phase 1 (Iter < Max_iter/3): High velocity ratio - Brownian motion
    2. Phase 2 (Max_iter/3 < Iter < 2*Max_iter/3): Unit velocity ratio - Mixed strategy
    3. Phase 3 (Iter > 2*Max_iter/3): Low velocity ratio - Levy flight
    
    The algorithm also includes environmental effects like FADs and eddy formation.
    
    References:
        Faramarzi, A., Heidarinejad, M., Mirjalili, S., & Gandomi, A. H. (2020).
        Marine Predators Algorithm: A Nature-inspired Metaheuristic.
        Expert Systems with Applications, 152, 113377.
    %}
    
    properties
        FADs  % Fish Aggregating Devices effect probability
        P     % Memory rate parameter
    end
    
    methods
        function obj = MarinePredatorsOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MarinePredatorsOptimizer constructor - Initialize the MPA solver
            
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
                    Additional MPA parameters:
                    - FADs: Fish Aggregating Devices effect probability (default: 0.2)
                    - P: Memory rate parameter (default: 0.5)
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set solver name
            obj.name_solver = "Marine Predators Algorithm";
            
            % Algorithm-specific parameters with defaults
            % kwargs (Name-Value pairs)
            if ~isempty(varargin)
                obj.kwargs = obj.nv2struct(varargin{:});
            end
            obj.FADs = obj.get_kw('FADs', 0.2);  % FADs effect probability
            obj.P = obj.get_kw('P', 0.5);        % Memory rate parameter
        end
        
        function RL = levy_flight(~, n, m, beta)
            %{
            levy_flight - Generate Levy flight random numbers
            
            Inputs:
                n: Number of samples
                m: Number of dimensions
                beta: Power law index (1 < beta < 2)
                
            Returns:
                RL: Levy flight random numbers of shape (n, m)
            %}
            
            num = gamma(1 + beta) * sin(pi * beta / 2);
            den = gamma((1 + beta) / 2) * beta * (2 ^ ((beta - 1) / 2));
            sigma_u = (num / den) ^ (1 / beta);
            
            u = normrnd(0, sigma_u, n, m);
            v = normrnd(0, 1, n, m);
            
            RL = u ./ (abs(v) .^ (1 / beta));
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for MPA algorithm
            
            The algorithm follows three main phases based on velocity ratio:
            1. High velocity ratio (Iter < Max_iter/3): Brownian motion
            2. Unit velocity ratio (Max_iter/3 < Iter < 2*Max_iter/3): Mixed strategy
            3. Low velocity ratio (Iter > 2*Max_iter/3): Levy flight
            
            Inputs:
                search_agents_no : int
                    Number of search agents (predators/prey)
                max_iter : int
                    Maximum number of iterations for optimization
                    
            Returns:
                history_step_solver : cell array
                    History of best solutions at each iteration
                best_solver : Member
                    Best solution found overall
            %}
            
            % Initialize the population of search agents
            population = obj.init_population(search_agents_no);
            
            % Initialize storage variables
            history_step_solver = {};
            
            % Initialize best solution
            sorted_population = obj.sort_population(population);
            best_solver = sorted_population{1}.copy();
            
            % Memory for previous population and fitness
            Prey_old = cell(1, search_agents_no);
            for i = 1:search_agents_no
                Prey_old{i} = population{i}.copy();
            end
            fit_old = obj.get_fitness(population);
            
            % Initialize top predator
            if obj.maximize
                Top_predator_fitness = -inf;
            else
                Top_predator_fitness = inf;
            end
            Top_predator_position = zeros(1, obj.dim);
            
            % Boundary matrices for FADs effect
            Xmin = repmat(obj.lb, search_agents_no, 1);
            Xmax = repmat(obj.ub, search_agents_no, 1);
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % ------------------- Detecting top predator -------------------
                for i = 1:search_agents_no
                    % Ensure positions stay within bounds
                    population{i}.position = max(min(population{i}.position, obj.ub), obj.lb);
                    population{i}.fitness = obj.objective_func(population{i}.position);
                    
                    % Update top predator
                    if obj.is_better(population{i}, Member(Top_predator_position, Top_predator_fitness))
                        Top_predator_fitness = population{i}.fitness;
                        Top_predator_position = population{i}.position;
                    end
                end
                
                % ------------------- Marine Memory saving -------------------
                if iter == 1
                    fit_old = obj.get_fitness(population);
                    for i = 1:search_agents_no
                        Prey_old{i} = population{i}.copy();
                    end
                end
                
                % Update population based on memory
                current_fitness = obj.get_fitness(population);
                if obj.maximize
                    Inx = fit_old > current_fitness;
                else
                    Inx = fit_old < current_fitness;
                end
                
                Indx = repmat(Inx', 1, obj.dim);
                
                % Update positions based on memory
                positions = obj.get_positions(population);
                old_positions = obj.get_positions(Prey_old);
                new_positions = positions;
                new_positions(Indx) = old_positions(Indx);
                
                % Update fitness based on memory
                new_fitness = current_fitness;
                new_fitness(Inx) = fit_old(Inx);
                
                % Update population
                for i = 1:search_agents_no
                    population{i}.position = new_positions(i, :);
                    population{i}.fitness = new_fitness(i);
                end
                
                % Update memory
                fit_old = new_fitness;
                for i = 1:search_agents_no
                    Prey_old{i} = population{i}.copy();
                end
                % ------------------------------------------------------------
                
                % Create elite matrix (replicate top predator)
                Elite = repmat(Top_predator_position, search_agents_no, 1);
                
                % Compute convergence factor
                CF = (1 - iter / max_iter) ^ (2 * iter / max_iter);
                
                % Generate random vectors
                RL = 0.05 * obj.levy_flight(search_agents_no, obj.dim, 1.5);  % Levy flight
                RB = randn(search_agents_no, obj.dim);                         % Brownian motion
                
                % Update positions based on current phase
                positions = obj.get_positions(population);
                
                for i = 1:search_agents_no
                    for j = 1:obj.dim
                        R = rand();
                        
                        % ------------------- Phase 1 (Eq.12) -------------------
                        if iter < max_iter / 3
                            stepsize = RB(i, j) * (Elite(i, j) - RB(i, j) * positions(i, j));
                            positions(i, j) = positions(i, j) + obj.P * R * stepsize;
                        
                        % --------------- Phase 2 (Eqs. 13 & 14)----------------
                        elseif iter < 2 * max_iter / 3
                            if i > search_agents_no / 2
                                stepsize = RB(i, j) * (RB(i, j) * Elite(i, j) - positions(i, j));
                                positions(i, j) = Elite(i, j) + obj.P * CF * stepsize;
                            else
                                stepsize = RL(i, j) * (Elite(i, j) - RL(i, j) * positions(i, j));
                                positions(i, j) = positions(i, j) + obj.P * R * stepsize;
                            end
                        
                        % ------------------ Phase 3 (Eq. 15)-------------------
                        else
                            stepsize = RL(i, j) * (RL(i, j) * Elite(i, j) - positions(i, j));
                            positions(i, j) = Elite(i, j) + obj.P * CF * stepsize;
                        end
                    end
                end
                
                % Update population positions
                for i = 1:search_agents_no
                    population{i}.position = positions(i, :);
                end
                
                % ------------------- Detecting top predator -------------------
                for i = 1:search_agents_no
                    % Ensure positions stay within bounds
                    population{i}.position = max(min(population{i}.position, obj.ub), obj.lb);
                    population{i}.fitness = obj.objective_func(population{i}.position);
                    
                    % Update top predator
                    if obj.is_better(population{i}, Member(Top_predator_position, Top_predator_fitness))
                        Top_predator_fitness = population{i}.fitness;
                        Top_predator_position = population{i}.position;
                    end
                end
                
                % ----------------------- Marine Memory saving ----------------
                current_fitness = obj.get_fitness(population);
                if obj.maximize
                    Inx = fit_old > current_fitness;
                else
                    Inx = fit_old < current_fitness;
                end
                
                Indx = repmat(Inx', 1, obj.dim);
                
                % Update positions based on memory
                positions = obj.get_positions(population);
                old_positions = obj.get_positions(Prey_old);
                new_positions = positions;
                new_positions(Indx) = old_positions(Indx);
                
                % Update fitness based on memory
                new_fitness = current_fitness;
                new_fitness(Inx) = fit_old(Inx);
                
                % Update population
                for i = 1:search_agents_no
                    population{i}.position = new_positions(i, :);
                    population{i}.fitness = new_fitness(i);
                end
                
                % Update memory
                fit_old = new_fitness;
                for i = 1:search_agents_no
                    Prey_old{i} = population{i}.copy();
                end
                
                % ---------- Eddy formation and FADs' effect (Eq 16) -----------
                if rand() < obj.FADs
                    % FADs effect
                    U = rand(search_agents_no, obj.dim) < obj.FADs;
                    random_positions = Xmin + rand(search_agents_no, obj.dim) .* (Xmax - Xmin);
                    positions = positions + CF * random_positions .* U;
                else
                    % Eddy formation effect
                    r = rand();
                    Rs = search_agents_no;
                    idx1 = randperm(Rs);
                    idx2 = randperm(Rs);
                    stepsize = (obj.FADs * (1 - r) + r) * (positions(idx1, :) - positions(idx2, :));
                    positions = positions + stepsize;
                end
                
                % Update population positions
                for i = 1:search_agents_no
                    population{i}.position = positions(i, :);
                end
                
                % Update best solution
                sorted_population = obj.sort_population(population);
                current_best = sorted_population{1};
                if obj.is_better(current_best, best_solver)
                    best_solver = current_best.copy();
                end
                
                % Store the best solution at this iteration
                history_step_solver{end+1} = best_solver.copy();
                
                % Update progress
                obj.callbacks(iter, max_iter, best_solver);
            end
            
            % Final evaluation
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
