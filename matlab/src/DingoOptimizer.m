classdef DingoOptimizer < Solver
    %{
    Dingo Optimization Algorithm (DOA) implementation.
    
    A bio-inspired optimization method inspired by dingoes hunting strategies.
    
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
        - p: Hunting probability (default: 0.5)
        - q: Group attack probability (default: 0.7)
        - na_min: Minimum number of attacking dingoes (default: 2)
    %}
    
    properties
        p          % Hunting probability
        q          % Group attack probability
        na_min     % Minimum number of attacking dingoes
    end
    
    methods
        function obj = DingoOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            DingoOptimizer constructor - Initialize the DOA solver
            
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
                    Additional DOA parameters:
                    - p: Hunting probability (default: 0.5)
                    - q: Group attack probability (default: 0.7)
                    - na_min: Minimum number of attacking dingoes (default: 2)
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set algorithm name
            obj.name_solver = "Dingo Optimizer";
            
            % kwargs (Name-Value pairs)
            if ~isempty(varargin)
                obj.kwargs = obj.nv2struct(varargin{:});
            end
            obj.p = obj.get_kw('p', 0.5);      % Hunting probability
            obj.q = obj.get_kw('q', 0.7);      % Group attack probability
            obj.na_min = obj.get_kw('na_min', 2); % Minimum attacking dingoes
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for Dingo Optimization Algorithm
            
            Inputs:
                search_agents_no : int
                    Number of search agents (dingoes)
                max_iter : int
                    Maximum number of iterations
                    
            Returns:
                history_step_solver : cell array
                    History of best solutions at each iteration
                best_solver : Member
                    Best solution found overall
            %}
            
            % Initialize population
            population = obj.init_population(search_agents_no);
            
            % Initialize best solution
            [sorted_population, ~] = obj.sort_population(population);
            best_solver = sorted_population{1}.copy();
            
            % Initialize history
            history_step_solver = {};
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Calculate number of attacking dingoes for this iteration
                na = obj.calculate_attacking_dingoes(search_agents_no);
                
                % Update each search agent
                for i = 1:search_agents_no
                    % Generate new position based on hunting strategy
                    new_position = obj.update_position(population, i, na);
                    
                    % Ensure positions stay within bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Evaluate new fitness
                    new_fitness = obj.objective_func(new_position);
                    
                    % Compare and update if better
                    new_member = Member(new_position, new_fitness);
                    if obj.is_better(new_member, population{i})
                        population{i}.position = new_position;
                        population{i}.fitness = new_fitness;
                    end
                end
                
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
        
        function na = calculate_attacking_dingoes(obj, search_agents_no)
            %{
            calculate_attacking_dingoes - Calculate number of dingoes that will attack
            
            Inputs:
                search_agents_no : int
                    Total number of search agents
                    
            Returns:
                na : int
                    Number of attacking dingoes
            %}
            
            na_end = search_agents_no / obj.na_min;
            na = round(obj.na_min + (na_end - obj.na_min) * rand());
            na = max(obj.na_min, min(na, search_agents_no));
        end
        
        function new_position = update_position(obj, population, current_idx, na)
            %{
            update_position - Update position of a search agent based on hunting strategies
            
            Inputs:
                population : cell array
                    Current population
                current_idx : int
                    Index of current search agent
                na : int
                    Number of attacking dingoes
                    
            Returns:
                new_position : array
                    New position vector
            %}
            
            if rand() < obj.p  % Hunting strategy
                if rand() < obj.q  % Group attack
                    % Strategy 1: Group attack
                    sumatory = obj.group_attack(population, na, current_idx);
                    beta1 = -2 + 4 * rand();  % -2 < beta1 < 2
                    new_position = beta1 * sumatory - obj.best_solver.position;
                else  % Persecution
                    % Strategy 2: Persecution
                    r1 = randi([1, length(population)]);
                    beta1 = -2 + 4 * rand();  % -2 < beta1 < 2
                    beta2 = -1 + 2 * rand();  % -1 < beta2 < 1
                    new_position = (obj.best_solver.position + ...
                                   beta1 * exp(beta2) * ...
                                   (population{r1}.position - population{current_idx}.position));
                end
            else  % Scavenger strategy
                % Strategy 3: Scavenging
                r1 = randi([1, length(population)]);
                beta2 = -1 + 2 * rand();  % -1 < beta2 < 1
                binary_val = randi([0, 1]);  % 0 or 1
                new_position = (exp(beta2) * population{r1}.position - ...
                               ((-1) ^ binary_val) * population{current_idx}.position) / 2;
            end
            
            % Apply survival strategy if needed
            survival_rate = obj.calculate_survival_rate(population, current_idx);
            if survival_rate <= 0.3
                % Strategy 4: Survival
                [r1, r2] = obj.get_two_distinct_indices(length(population), current_idx);
                binary_val = randi([0, 1]);  % 0 or 1
                new_position = (obj.best_solver.position + ...
                               (population{r1}.position - ...
                                ((-1) ^ binary_val) * population{r2}.position) / 2);
            end
        end
        
        function sumatory = group_attack(obj, population, na, current_idx)
            %{
            group_attack - Perform group attack strategy
            
            Inputs:
                population : cell array
                    Current population
                na : int
                    Number of attacking dingoes
                current_idx : int
                    Index of current search agent
                    
            Returns:
                sumatory : array
                    Sumatory vector for group attack
            %}
            
            attack_indices = obj.get_attack_indices(length(population), na, current_idx);
            sumatory = zeros(1, obj.dim);
            
            for idx = attack_indices
                sumatory = sumatory + (population{idx}.position - population{current_idx}.position);
            end
            
            sumatory = sumatory / na;
        end
        
        function attack_indices = get_attack_indices(~, population_size, na, exclude_idx)
            %{
            get_attack_indices - Get indices of dingoes that will participate in group attack
            
            Inputs:
                population_size : int
                    Total population size
                na : int
                    Number of attacking dingoes
                exclude_idx : int
                    Index to exclude (current dingo)
                    
            Returns:
                attack_indices : array
                    List of attack indices
            %}
            
            attack_indices = [];
            while length(attack_indices) < na
                idx = randi([1, population_size]);
                if idx ~= exclude_idx && ~ismember(idx, attack_indices)
                    attack_indices = [attack_indices, idx];
                end
            end
        end
        
        function [r1, r2] = get_two_distinct_indices(~, population_size, exclude_idx)
            %{
            get_two_distinct_indices - Get two distinct random indices excluding the specified index
            
            Inputs:
                population_size : int
                    Total population size
                exclude_idx : int
                    Index to exclude
                    
            Returns:
                r1 : int
                    First random index
                r2 : int
                    Second random index
            %}
            
            while true
                r1 = randi([1, population_size]);
                r2 = randi([1, population_size]);
                if r1 ~= r2 && r1 ~= exclude_idx && r2 ~= exclude_idx
                    return;
                end
            end
        end
        
        function survival_rate = calculate_survival_rate(obj, population, current_idx)
            %{
            calculate_survival_rate - Calculate survival rate for a search agent
            
            Inputs:
                population : cell array
                    Current population
                current_idx : int
                    Index of current search agent
                    
            Returns:
                survival_rate : float
                    Survival rate (0 to 1)
            %}
            
            fitness_values = zeros(1, length(population));
            for i = 1:length(population)
                fitness_values(i) = population{i}.fitness;
            end
            
            min_fitness = min(fitness_values);
            max_fitness = max(fitness_values);
            
            if max_fitness == min_fitness
                survival_rate = 1.0;  % All have same fitness
                return;
            end
            
            current_fitness = population{current_idx}.fitness;
            if obj.maximize
                survival_rate = (max_fitness - current_fitness) / (max_fitness - min_fitness);
            else
                survival_rate = (current_fitness - min_fitness) / (max_fitness - min_fitness);
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
