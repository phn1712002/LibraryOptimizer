classdef ModifiedSocialGroupOptimizer < Solver
    %{
    Modified Social Group Optimization (MSGO) algorithm.
    
    MSGO is a metaheuristic optimization algorithm inspired by social group behavior,
    featuring guru phase (learning from best) and learner phase (mutual learning).
    
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
        - c: Learning coefficient for guru phase (default: 0.2)
        - sap: Self-adaptive probability for random exploration (default: 0.7)
    %}
    
    properties
        c          % Learning coefficient for guru phase
        sap        % Self-adaptive probability for random exploration
    end
    
    methods
        function obj = ModifiedSocialGroupOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            ModifiedSocialGroupOptimizer constructor - Initialize the MSGO solver
            
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
                    Additional MSGO parameters:
                    - c: Learning coefficient for guru phase (default: 0.2)
                    - sap: Self-adaptive probability for random exploration (default: 0.7)
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set algorithm name
            obj.name_solver = "Modified Social Group Optimizer";
            
            % kwargs (Name-Value pairs)
            if ~isempty(varargin)
                obj.kwargs = obj.nv2struct(varargin{:});
            end
            obj.c = obj.get_kw('c', 0.2);    % Learning coefficient for guru phase
            obj.sap = obj.get_kw('sap', 0.7); % Self-adaptive probability for random exploration
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for MSGO algorithm
            
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
            
            % Initialize best solution
            sorted_population = obj.sort_population(population);
            best_solver = sorted_population{1}.copy();
            
            % Call the begin function
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Phase 1: Guru Phase (improvement due to best person)
                sorted_population = obj.sort_population(population);
                guru = sorted_population{1}.copy();  % Best individual
                
                % Create new population for guru phase
                new_population = cell(1, search_agents_no);
                for i = 1:search_agents_no
                    % Update position: c*current + rand*(guru - current)
                    new_position = obj.c * population{i}.position + ...
                                  rand(1, obj.dim) .* (guru.position - population{i}.position);
                    
                    % Apply bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Evaluate new fitness
                    new_fitness = obj.objective_func(new_position);
                    
                    % Greedy selection: accept if better
                    if obj.is_better(Member(new_position, new_fitness), population{i})
                        population{i}.position = new_position;
                        population{i}.fitness = new_fitness;
                    end
                    
                    new_population{i} = population{i}.copy();
                end
                
                population = new_population;
                
                % Phase 2: Learner Phase (mutual learning with random exploration)
                sorted_population = obj.sort_population(population);
                global_best = sorted_population{1}.copy();  % Global best for guidance
                
                % Create new population for learner phase
                new_population_learner = cell(1, search_agents_no);
                for i = 1:search_agents_no
                    % Choose a random partner different from current individual
                    r1 = randi(search_agents_no);
                    while r1 == i
                        r1 = randi(search_agents_no);
                    end
                    
                    if (obj.maximize && population{i}.fitness > population{r1}.fitness) || ...
                       (~obj.maximize && population{i}.fitness < population{r1}.fitness)
                        % Current individual is better than random partner
                        if rand() > obj.sap
                            % Learning strategy: current + rand*(current - partner) + rand*(global_best - current)
                            new_position = population{i}.position + ...
                                          rand(1, obj.dim) .* (population{i}.position - population{r1}.position) + ...
                                          rand(1, obj.dim) .* (global_best.position - population{i}.position);
                        else
                            % Random exploration
                            new_position = obj.lb + (obj.ub - obj.lb) .* rand(1, obj.dim);
                        end
                    else
                        % Current individual is worse than random partner
                        % Learning strategy: current + rand*(partner - current) + rand*(global_best - current)
                        new_position = population{i}.position + ...
                                      rand(1, obj.dim) .* (population{r1}.position - population{i}.position) + ...
                                      rand(1, obj.dim) .* (global_best.position - population{i}.position);
                    end
                    
                    % Apply bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Evaluate new fitness
                    new_fitness = obj.objective_func(new_position);
                    
                    % Greedy selection: accept if better
                    if obj.is_better(Member(new_position, new_fitness), population{i})
                        population{i}.position = new_position;
                        population{i}.fitness = new_fitness;
                    end
                    
                    new_population_learner{i} = population{i}.copy();
                end
                
                population = new_population_learner;
                
                % Update best solution
                sorted_population = obj.sort_population(population);
                current_best = sorted_population{1};
                if obj.is_better(current_best, best_solver)
                    best_solver = current_best.copy();
                end
                
                % Store the best solution at this iteration
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
        
        function sorted_population = sort_population(obj, population)
            %{
            sort_population - Sort population based on fitness (best first)
            
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
