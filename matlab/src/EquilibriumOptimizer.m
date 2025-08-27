classdef EquilibriumOptimizer < Solver
    %{
    Equilibrium Optimizer (EO) Algorithm.
    
    EO is a physics-inspired optimization algorithm that mimics the control volume
    mass balance model to estimate both dynamic and equilibrium states. The algorithm
    uses equilibrium candidates to guide the search process towards optimal solutions.
    
    The algorithm maintains an equilibrium pool consisting of:
    - 4 best candidates (Ceq1, Ceq2, Ceq3, Ceq4)
    - 1 average candidate (Ceq_ave)
    
    References:
        Faramarzi, A., Heidarinejad, M., Stephens, B., & Mirjalili, S. (2020).
        Equilibrium optimizer: A novel optimization algorithm.
        Knowledge-Based Systems, 191, 105190.
    %}
    
    properties
        a1  % Exploration parameter
        a2  % Exploitation parameter
        GP  % Generation probability
    end
    
    methods
        function obj = EquilibriumOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            EquilibriumOptimizer constructor - Initialize the EO solver
            
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
                    Additional EO parameters:
                    - a1: Exploration parameter (default: 2)
                    - a2: Exploitation parameter (default: 1)
                    - GP: Generation probability (default: 0.5)
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set solver name
            obj.name_solver = "Equilibrium Optimizer";
            
            % Algorithm-specific parameters with defaults
            % kwargs (Name-Value pairs)
            if ~isempty(varargin)
                obj.kwargs = obj.nv2struct(varargin{:});
            end
            obj.a1 = obj.get_kw('a1', 2);   % Exploration parameter
            obj.a2 = obj.get_kw('a2', 1);   % Exploitation parameter
            obj.GP = obj.get_kw('GP', 0.5); % Generation probability
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for EO algorithm
            
            The algorithm uses an equilibrium pool of candidates to guide the search
            process towards optimal solutions through physics-inspired update rules.
            
            Inputs:
                search_agents_no : int
                    Number of search agents
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
            
            % Initialize equilibrium candidates
            if obj.maximize
                initial_fitness = -inf;
            else
                initial_fitness = inf;
            end
            Ceq1 = Member(zeros(1, obj.dim), initial_fitness);
            Ceq2 = Member(zeros(1, obj.dim), initial_fitness);
            Ceq3 = Member(zeros(1, obj.dim), initial_fitness);
            Ceq4 = Member(zeros(1, obj.dim), initial_fitness);
            
            % Memory for previous population and fitness
            C_old = cell(1, search_agents_no);
            for i = 1:search_agents_no
                C_old{i} = population{i}.copy();
            end
            fit_old = obj.get_fitness(population);
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Update equilibrium candidates
                for i = 1:search_agents_no
                    % Ensure positions stay within bounds
                    population{i}.position = max(min(population{i}.position, obj.ub), obj.lb);
                    population{i}.fitness = obj.objective_func(population{i}.position);
                    
                    % Update equilibrium candidates based on fitness
                    if obj.is_better(population{i}, Ceq1)
                        Ceq4 = Ceq3.copy();
                        Ceq3 = Ceq2.copy();
                        Ceq2 = Ceq1.copy();
                        Ceq1 = population{i}.copy();
                    elseif obj.is_better(population{i}, Ceq2) && ~obj.is_better(population{i}, Ceq1)
                        Ceq4 = Ceq3.copy();
                        Ceq3 = Ceq2.copy();
                        Ceq2 = population{i}.copy();
                    elseif obj.is_better(population{i}, Ceq3) && ~obj.is_better(population{i}, Ceq2)
                        Ceq4 = Ceq3.copy();
                        Ceq3 = population{i}.copy();
                    elseif obj.is_better(population{i}, Ceq4) && ~obj.is_better(population{i}, Ceq3)
                        Ceq4 = population{i}.copy();
                    end
                end
                
                % ----------------- Memory saving -----------------
                if iter == 1
                    fit_old = obj.get_fitness(population);
                    for i = 1:search_agents_no
                        C_old{i} = population{i}.copy();
                    end
                end
                
                % Update population based on memory
                current_fitness = obj.get_fitness(population);
                for i = 1:search_agents_no
                    if (not(obj.maximize) && fit_old(i) < current_fitness(i)) || ...
                       (obj.maximize && fit_old(i) > current_fitness(i))
                        population{i}.fitness = fit_old(i);
                        population{i}.position = C_old{i}.position;
                    end
                end
                
                % Update memory
                for i = 1:search_agents_no
                    C_old{i} = population{i}.copy();
                end
                fit_old = obj.get_fitness(population);
                % -------------------------------------------------
                
                % Create equilibrium pool
                Ceq_ave = Member((Ceq1.position + Ceq2.position + Ceq3.position + Ceq4.position) / 4, 0);
                Ceq_pool = {Ceq1, Ceq2, Ceq3, Ceq4, Ceq_ave};
                
                % Compute time parameter
                t = (1 - iter / max_iter) ^ (obj.a2 * iter / max_iter);
                
                % Update all search agents
                for i = 1:search_agents_no
                    % Randomly select one candidate from the pool
                    Ceq_idx = randi(length(Ceq_pool));
                    Ceq = Ceq_pool{Ceq_idx};
                    
                    % Generate random vectors
                    lambda_vec = rand(1, obj.dim);
                    r = rand(1, obj.dim);
                    
                    % Compute F parameter
                    F = obj.a1 * sign(r - 0.5) .* (exp(-lambda_vec * t) - 1);
                    
                    % Compute generation control parameter
                    r1 = rand();
                    r2 = rand();
                    GCP = 0.5 * r1 * ones(1, obj.dim) * (r2 >= obj.GP);
                    
                    % Compute generation rate
                    G0 = GCP .* (Ceq.position - lambda_vec .* population{i}.position);
                    G = G0 .* F;
                    
                    % Update position using EO equation
                    new_position = Ceq.position + ...
                                  (population{i}.position - Ceq.position) .* F + ...
                                  (G ./ (lambda_vec * 1.0)) .* (1 - F);
                    
                    % Ensure positions stay within bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    population{i}.position = new_position;
                    population{i}.fitness = obj.objective_func(new_position);
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
        
        function s = get_kw(obj, name, default)
            if isfield(obj.kwargs, name), s = obj.kwargs.(name);
            else, s = default; end
        end
    end
end
