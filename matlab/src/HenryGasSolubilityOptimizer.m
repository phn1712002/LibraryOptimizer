classdef HenryGasSolubilityOptimizer < Solver
    %{
    Henry Gas Solubility Optimization (HGSO) Algorithm.
    
    HGSO is a physics-inspired metaheuristic optimization algorithm that mimics
    the behavior of gas solubility in liquid based on Henry's law. The algorithm
    uses the principles of Henry's gas solubility to optimize search processes.
    
    The algorithm features:
    - Group-based population structure with different Henry constants
    - Temperature-dependent solubility updates
    - Position updates based on gas solubility principles
    - Worst agent replacement mechanism for diversity
    
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
        - n_types: Number of gas types/groups (default: 5)
        - l1: Constant for Henry's constant initialization (default: 5e-3)
        - l2: Constant for partial pressure initialization (default: 100)
        - l3: Constant for constant C initialization (default: 1e-2)
        - alpha: Constant for position update (default: 1)
        - beta: Constant for position update (default: 1)
        - M1: Minimum fraction of worst agents to replace (default: 0.1)
        - M2: Maximum fraction of worst agents to replace (default: 0.2)
    
    References:
        Original MATLAB implementation by Essam Houssein
        Based on Henry's gas solubility law
    %}
    
    properties
        n_types    % Number of gas types/groups
        l1         % Constant for Henry's constant
        l2         % Constant for partial pressure
        l3         % Constant for constant C
        alpha      % Position update constant
        beta       % Position update constant
        M1         % Minimum fraction of worst agents
        M2         % Maximum fraction of worst agents
    end
    
    methods
        function obj = HenryGasSolubilityOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            HenryGasSolubilityOptimizer constructor - Initialize the HGSO solver
            
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
                    Additional HGSO parameters:
                    - n_types: Number of gas types/groups (default: 5)
                    - l1: Constant for Henry's constant initialization (default: 5e-3)
                    - l2: Constant for partial pressure initialization (default: 100)
                    - l3: Constant for constant C initialization (default: 1e-2)
                    - alpha: Constant for position update (default: 1)
                    - beta: Constant for position update (default: 1)
                    - M1: Minimum fraction of worst agents to replace (default: 0.1)
                    - M2: Maximum fraction of worst agents to replace (default: 0.2)
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set solver name
            obj.name_solver = "Henry Gas Solubility Optimizer";
            
            % Algorithm-specific parameters with defaults
            % kwargs (Name-Value pairs)
            if ~isempty(varargin)
                obj.kwargs = obj.nv2struct(varargin{:});
            end
            obj.n_types = obj.get_kw('n_types', 5);   % Number of gas types/groups
            obj.l1 = obj.get_kw('l1', 5e-3);          % Constant for Henry's constant
            obj.l2 = obj.get_kw('l2', 100);           % Constant for partial pressure
            obj.l3 = obj.get_kw('l3', 1e-2);          % Constant for constant C
            obj.alpha = obj.get_kw('alpha', 1);       % Position update constant
            obj.beta = obj.get_kw('beta', 1);         % Position update constant
            obj.M1 = obj.get_kw('M1', 0.1);           % Min fraction of worst agents
            obj.M2 = obj.get_kw('M2', 0.2);           % Max fraction of worst agents
            
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for HGSO algorithm
            
            The algorithm simulates gas solubility behavior based on Henry's law
            with group-based optimization and temperature-dependent updates.
            
            Inputs:
                search_agents_no : int
                    Number of gas particles in the population
                max_iter : int
                    Maximum number of iterations for optimization
                    
            Returns:
                history_step_solver : cell array
                    History of best solutions at each iteration
                best_solver : Member
                    Best solution found overall
            %}

            % Initialize storage variables
            history_step_solver = {};
            
            % Initialize population
            population = obj.init_population(search_agents_no);
            
            % Initialize best solution
            sorted_population = obj.sort_population(population);
            best_solver = sorted_population{1}.copy();
            
            % Initialize algorithm parameters
            K = obj.l1 * rand(1, obj.n_types);  % Henry's constants
            P = obj.l2 * rand(1, search_agents_no);  % Partial pressures
            C = obj.l3 * rand(1, obj.n_types);  % Constants
            
            % Create groups
            groups = obj.create_groups(population, search_agents_no);
            
            % Evaluate initial groups
            group_best_fitness = zeros(1, obj.n_types);
            group_best_positions = cell(1, obj.n_types);
            
            for i = 1:obj.n_types
                [groups{i}, group_best_fitness(i), group_best_positions{i}] = ...
                    obj.evaluate_group(groups{i}, [], true);
            end
            
            % Find global best
            if obj.maximize
                [global_best_fitness, global_best_idx] = max(group_best_fitness);
            else
                [global_best_fitness, global_best_idx] = min(group_best_fitness);
            end
            global_best_position = group_best_positions{global_best_idx};
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Update variables (solubility)
                S = obj.update_variables(search_agents_no, iter, max_iter, K, P, C);
                
                % Update positions
                new_groups = obj.update_positions(...
                    groups, group_best_positions, global_best_position, S, ...
                    global_best_fitness, search_agents_no);
                
                % Ensure positions stay within bounds
                new_groups = obj.check_positions(new_groups);
                
                % Evaluate new groups and update
                for i = 1:obj.n_types
                    [groups{i}, group_best_fitness(i), group_best_positions{i}] = ...
                        obj.evaluate_group(groups{i}, new_groups{i}, false);
                    
                    % Replace worst agents
                    groups{i} = obj.worst_agents(groups{i});
                end
                
                % Update global best
                if obj.maximize
                    [current_best_fitness, current_best_idx] = max(group_best_fitness);
                else
                    [current_best_fitness, current_best_idx] = min(group_best_fitness);
                end
                current_best_position = group_best_positions{current_best_idx};
                
                if obj.is_better(Member(current_best_position, current_best_fitness), ...
                                Member(global_best_position, global_best_fitness))
                    global_best_fitness = current_best_fitness;
                    global_best_position = current_best_position;
                    best_solver = Member(global_best_position, global_best_fitness);
                end
                
                % Store history
                history_step_solver{end+1} = best_solver.copy();
                
                % Update progress
                obj.callbacks(iter, max_iter, best_solver);
            end
            
            % Final processing
            obj.history_step_solver = history_step_solver;
            obj.best_solver = best_solver;
            
            % End solver
            obj.end_step_solver();
        end
        
        function groups = create_groups(obj, population, search_agents_no)
            % Create groups from population.
            group_size = search_agents_no / obj.n_types;
            groups = cell(1, obj.n_types);
            
            for i = 1:obj.n_types
                start_idx = (i - 1) * group_size + 1;
                end_idx = i * group_size;
                groups{i} = population(start_idx:end_idx);
            end
        end
        
        function [group, best_fitness, best_position] = evaluate_group(obj, group, new_group, init_flag)
            % Evaluate group fitness and find best solution.
            group_size = length(group);
            
            if init_flag
                % Initial evaluation
                for j = 1:group_size
                    group{j}.fitness = obj.objective_func(group{j}.position);
                end
            else
                % Update evaluation
                for j = 1:group_size
                    new_fitness = obj.objective_func(new_group{j}.position);
                    if (not(obj.maximize) && new_fitness < group{j}.fitness) || ...
                       (obj.maximize && new_fitness > group{j}.fitness)
                        group{j}.fitness = new_fitness;
                        group{j}.position = new_group{j}.position;
                    end
                end
            end
            
            % Find best in group
            fitness_values = obj.get_fitness(group);
            if obj.maximize
                [best_fitness, best_idx] = max(fitness_values);
            else
                [best_fitness, best_idx] = min(fitness_values);
            end
            
            best_position = group{best_idx}.position;
        end
        
        function S = update_variables(obj, search_agents_no, iter, max_iter, K, P, C)
            % Update solubility variables.
            T = exp(-iter / max_iter);  % Temperature
            T0 = 298.15;  % Reference temperature
            
            group_size = search_agents_no / obj.n_types;
            S = zeros(1, search_agents_no);  % Solubility
            
            for j = 1:obj.n_types
                % Update Henry's constant
                K(j) = K(j) * exp(-C(j) * (1/T - 1/T0));
                
                % Update solubility for this group
                start_idx = (j - 1) * group_size + 1;
                end_idx = j * group_size;
                S(start_idx:end_idx) = P(start_idx:end_idx) * K(j);
            end
        end
        
        function new_groups = update_positions(obj, groups, group_best_positions, global_best_position, S, ...
                                             global_best_fitness, search_agents_no)
            % Update particle positions.
            new_groups = cell(1, obj.n_types);
            group_size = search_agents_no / obj.n_types;
            flag_options = [1, -1];  % Direction flags
            
            for i = 1:obj.n_types
                new_group = cell(1, group_size);
                for j = 1:group_size
                    % Calculate gamma parameter
                    current_fitness = groups{i}{j}.fitness;
                    gamma = obj.beta * exp(...
                        -(global_best_fitness + 0.05) / (current_fitness + 0.05)...
                    );
                    
                    % Random direction flag
                    flag_idx = randi([1, 2]);
                    direction_flag = flag_options(flag_idx);
                    
                    % Update position
                    new_position = groups{i}{j}.position;
                    for k = 1:obj.dim
                        % Group best influence
                        group_best_influence = direction_flag * rand() * gamma * ...
                                              (group_best_positions{i}(k) - groups{i}{j}.position(k));
                        
                        % Global best influence
                        global_best_influence = rand() * obj.alpha * direction_flag * ...
                                               (S((i-1)*group_size + j) * global_best_position(k) - groups{i}{j}.position(k));
                        
                        new_position(k) = new_position(k) + group_best_influence + global_best_influence;
                    end
                    
                    new_member = Member(new_position, 0.0);
                    new_group{j} = new_member;
                end
                new_groups{i} = new_group;
            end
        end
        
        function groups = check_positions(obj, groups)
            % Ensure positions stay within bounds.
            for i = 1:obj.n_types
                for j = 1:length(groups{i})
                    groups{i}{j}.position = max(min(groups{i}{j}.position, obj.ub), obj.lb);
                end
            end
        end
        
        function group = worst_agents(obj, group)
            % Replace worst agents in group.
            group_size = length(group);
            
            % Calculate number of worst agents to replace
            M1N = obj.M1 * group_size;
            M2N = obj.M2 * group_size;
            Nw = round((M2N - M1N) * rand() + M1N);
            
            if Nw > 0
                % Sort by fitness (worst first)
                fitness_values = obj.get_fitness(group);
                if obj.maximize
                    [~, sorted_indices] = sort(fitness_values, 'ascend');  % Ascending for maximization
                else
                    [~, sorted_indices] = sort(fitness_values, 'descend'); % Descending for minimization
                end
                
                % Replace worst agents with random positions
                for k = 1:min(Nw, group_size)
                    worst_idx = sorted_indices(k);
                    new_position = rand(1, obj.dim) .* (obj.ub - obj.lb) + obj.lb;
                    group{worst_idx}.position = new_position;
                    group{worst_idx}.fitness = obj.objective_func(new_position);
                end
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
        
        function s = get_kw(obj, name, default)
            if isfield(obj.kwargs, name), s = obj.kwargs.(name);
            else, s = default; end
        end
    end
end
