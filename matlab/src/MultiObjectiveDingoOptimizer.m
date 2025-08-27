classdef MultiObjectiveDingoOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveDingoOptimizer - Multi-Objective Dingo Optimization Algorithm (DOA)
    
    This algorithm extends the standard DOA for multi-objective optimization
    using archive management and grid-based selection for leader selection.
    
    Parameters:
    -----------
    objective_func : function handle
        Objective function that returns an array of fitness values
    lb : float or array
        Lower bounds for variables
    ub : float or array
        Upper bounds for variables
    dim : int
        Problem dimension
    maximize : bool
        Optimization direction (true for maximize, false for minimize)
    varargin : cell array
        Additional parameters:
        - p: Hunting probability (default: 0.5)
        - q: Group attack probability (default: 0.7)
        - na_min: Minimum number of attacking dingoes (default: 2)
    %}
    
    properties
        p
        q
        na_min
    end
    
    methods
        function obj = MultiObjectiveDingoOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveDingoOptimizer constructor - Initialize the MO-DOA solver
            
            Inputs:
                objective_func : function handle
                    Objective function to optimize (returns array for multiple objectives)
                lb : float or array
                    Lower bounds of search space
                ub : float or array
                    Upper bounds of search space
                dim : int
                    Number of dimensions in the problem
                maximize : bool
                    Whether to maximize (true) or minimize (false) objectives
                varargin : cell array
                    Additional solver parameters
            %}
            
            % Call parent constructor
            obj@MultiObjectiveSolver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set solver name
            obj.name_solver = "Multi-Objective Dingo Optimizer";
            
            % Algorithm-specific parameters
            obj.p = obj.get_kw('p', 0.5);  % Hunting probability
            obj.q = obj.get_kw('q', 0.7);  % Group attack probability
            obj.na_min = obj.get_kw('na_min', 2);  % Minimum attacking dingoes
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective DOA
            
            Inputs:
                search_agents_no : int
                    Number of search agents (population size)
                max_iter : int
                    Maximum number of iterations
                    
            Returns:
                history_archive : cell array
                    History of archive states
                archive : cell array
                    Final archive of non-dominated solutions
            %}
            
            % Initialize storage
            history_archive = {};
            
            % Initialize population
            population = obj.init_population(search_agents_no);
            
            % Initialize archive with non-dominated solutions
            obj.determine_domination(population);
            non_dominated = obj.get_non_dominated_particles(population);
            obj.archive = [obj.archive, non_dominated];
            
            % Initialize grid for archive
            costs = obj.get_fitness(obj.archive);
            if ~isempty(costs)
                obj.grid = obj.create_hypercubes(costs);
                for k = 1:numel(obj.archive)
                    [gi, gs] = obj.get_grid_index(obj.archive(k));
                    obj.archive(k).grid_index = gi;
                    obj.archive(k).grid_sub_index = gs;
                end
            end
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Calculate number of attacking dingoes for this iteration
                na = obj.calculate_attacking_dingoes(search_agents_no);
                
                % Update all search agents
                for i = 1:numel(population)
                    % Generate new position based on hunting strategy
                    new_position = obj.update_position(population, i, na);
                    
                    % Ensure positions stay within bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Update dingo position and fitness
                    population(i).position = new_position;
                    new_fitness = obj.objective_func(new_position);
                    population(i).multi_fitness = new_fitness(:).';
                end
                
                % Update archive with current population
                obj = obj.add_to_archive(population);
                
                % Store archive state for history
                archive_copy = cell(1, numel(obj.archive));
                for idx = 1:numel(obj.archive)
                    archive_copy{idx} = obj.archive(idx).copy();
                end
                history_archive{end+1} = archive_copy;
                
                % Update progress
                if ~isempty(obj.archive)
                    best_member = obj.archive(1);
                else
                    best_member = [];
                end
                obj.callbacks(iter, max_iter, best_member);
            end
            
            % Final processing
            obj.history_step_solver = history_archive;
            obj.best_solver = obj.archive;
            
            % End solver
            obj.end_step_solver();
            
            archive = obj.archive;
        end
        
        function na = calculate_attacking_dingoes(obj, search_agents_no)
            %{
            calculate_attacking_dingoes - Calculate number of dingoes that will attack in current iteration.
            
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
            update_position - Update position of a search agent based on hunting strategies.
            
            Inputs:
                population : cell array
                    Current population
                current_idx : int
                    Index of current search agent
                na : int
                    Number of attacking dingoes
                iter : int
                    Current iteration
                max_iter : int
                    Maximum iterations
                    
            Returns:
                new_position : array
                    New position vector
            %}
            
            % Select leader from archive for guidance
            leader = obj.select_leader();
            if isempty(leader)
                % If no leader in archive, use random member from population
                leader_idx = randi(numel(population));
                leader = population(leader_idx);
            end
            
            if rand() < obj.p  % Hunting strategy
                if rand() < obj.q  % Group attack
                    % Strategy 1: Group attack
                    sumatory = obj.group_attack(population, na, current_idx);
                    beta1 = -2 + 4 * rand();  % -2 < beta1 < 2
                    new_position = beta1 * sumatory - leader.position;
                else  % Persecution
                    % Strategy 2: Persecution
                    r1 = randi(numel(population));
                    beta1 = -2 + 4 * rand();  % -2 < beta1 < 2
                    beta2 = -1 + 2 * rand();  % -1 < beta2 < 1
                    new_position = (leader.position + ...
                                   beta1 * exp(beta2) * ...
                                   (population(r1).position - population(current_idx).position));
                end
            else  % Scavenger strategy
                % Strategy 3: Scavenging
                r1 = randi(numel(population));
                beta2 = -1 + 2 * rand();  % -1 < beta2 < 1
                binary_val = 0;
                if rand() < 0.5
                    binary_val = 1;
                end
                new_position = (exp(beta2) * population(r1).position - ...
                               ((-1) ^ binary_val) * population(current_idx).position) / 2;
            end
            
            % Apply survival strategy if needed
            survival_rate = obj.calculate_survival_rate(population, current_idx);
            if survival_rate <= 0.3
                % Strategy 4: Survival
                [r1, r2] = obj.get_two_distinct_indices(numel(population), current_idx);
                binary_val = 0;
                if rand() < 0.5
                    binary_val = 1;
                end
                new_position = (leader.position + ...
                               (population(r1).position - ...
                                ((-1) ^ binary_val) * population(r2).position) / 2);
            end
        end
        
        function sumatory = group_attack(obj, population, na, current_idx)
            %{
            group_attack - Perform group attack strategy.
            
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
            attack_indices = obj.get_attack_indices(numel(population), na, current_idx);
            sumatory = zeros(1, obj.dim);
            
            for idx = attack_indices
                sumatory = sumatory + (population(idx).position - population(current_idx).position);
            end
            
            sumatory = sumatory / na;
        end
        
        function attack_indices = get_attack_indices(~, population_size, na, exclude_idx)
            %{
            get_attack_indices - Get indices of dingoes that will participate in group attack.
            
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
            while numel(attack_indices) < na
                idx = randi(population_size);
                if idx ~= exclude_idx && ~ismember(idx, attack_indices)
                    attack_indices = [attack_indices, idx];
                end
            end
        end
        
        function [r1, r2] = get_two_distinct_indices(~, population_size, exclude_idx)
            %{
            get_two_distinct_indices - Get two distinct random indices excluding the specified index.
            
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
                r1 = randi(population_size);
                r2 = randi(population_size);
                if r1 ~= r2 && r1 ~= exclude_idx && r2 ~= exclude_idx
                    return;
                end
            end
        end
        
        function survival_rate = calculate_survival_rate(obj, population, current_idx)
            %{
            calculate_survival_rate - Calculate survival rate for a search agent.
            
            Inputs:
                population : cell array
                    Current population
                current_idx : int
                    Index of current search agent
                    
            Returns:
                survival_rate : float
                    Survival rate (0 to 1)
            %}
            
            % For multi-objective, we need a different approach
            % Use the concept of dominance count instead of single fitness
            
            % Count how many solutions dominate the current solution
            dominated_count = 0;
            current_solution = population(current_idx);
            
            for i = 1:numel(population)
                if i ~= current_idx
                    other_solution = population(i);
                    if obj.dominates(other_solution, current_solution)
                        dominated_count = dominated_count + 1;
                    end
                end
            end
            
            % Survival rate is inversely proportional to dominance count
            survival_rate = 1.0 - (dominated_count / numel(population));
            
            survival_rate = max(0.0, min(1.0, survival_rate));
        end
    end
end
