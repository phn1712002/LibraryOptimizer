classdef MultiObjectiveHenryGasSolubilityOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveHenryGasSolubilityOptimizer - Multi-Objective Henry Gas Solubility Optimization (HGSO) Algorithm.
    
    This algorithm extends the standard HGSO for multi-objective optimization
    using archive management and grid-based selection for maintaining diversity.
    
    Parameters:
    -----------
    objective_func : function handle
        Objective function that returns a list of fitness values
    lb : float or array
        Lower bounds for variables
    ub : float or array
        Upper bounds for variables
    dim : int
        Problem dimension
    maximize : bool
        Optimization direction (true: maximize, false: minimize)
    varargin : cell array
        Additional parameters:
        - n_types: Number of gas types/groups (default: 5)
        - l1: Constant for Henry's constant initialization (default: 5e-3)
        - l2: Constant for partial pressure initialization (default: 100)
        - l3: Constant for constant C initialization (default: 1e-2)
        - alpha: Constant for position update (default: 1)
        - beta: Constant for position update (default: 1)
        - M1: Minimum fraction of worst agents to replace (default: 0.1)
        - M2: Maximum fraction of worst agents to replace (default: 0.2)
    %}
    
    properties
        n_types
        l1
        l2
        l3
        alpha
        beta
        M1
        M2
    end
    
    methods
        function obj = MultiObjectiveHenryGasSolubilityOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveHenryGasSolubilityOptimizer constructor - Initialize the MOHGSO solver
            
            Inputs:
                objective_func : function handle
                    Multi-objective function to optimize (returns array for multiple objectives)
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
            obj.name_solver = "Multi-Objective Henry Gas Solubility Optimizer";
            
            % Set algorithm parameters with defaults
            obj.n_types = obj.get_kw('n_types', 5);   % Number of gas types/groups
            obj.l1 = obj.get_kw('l1', 5e-3);          % Constant for Henry's constant
            obj.l2 = obj.get_kw('l2', 100);           % Constant for partial pressure
            obj.l3 = obj.get_kw('l3', 1e-2);          % Constant for constant C
            obj.alpha = obj.get_kw('alpha', 1);       % Position update constant
            obj.beta = obj.get_kw('beta', 1);         % Position update constant
            obj.M1 = obj.get_kw('M1', 0.1);           % Min fraction of worst agents
            obj.M2 = obj.get_kw('M2', 0.2);           % Max fraction of worst agents
        end
        
        function groups = create_groups(obj, population)
            %{
            create_groups - Create groups from population.
            
            Inputs:
                population : object array
                    Population to group
                    
            Returns:
                groups : cell array
                    Cell array of groups
            %}
            
            group_size = floor(length(population) / obj.n_types);
            groups = cell(1, obj.n_types);
            
            for i = 1:obj.n_types
                start_idx = (i-1) * group_size + 1;
                end_idx = i * group_size;
                groups{i} = population(start_idx:end_idx);
            end
        end
        
        function [evaluated_group, best_member] = evaluate_group_multi(obj, group, new_group, init_flag)
            %{
            evaluate_group_multi - Evaluate group fitness for multi-objective optimization.
            
            Inputs:
                group : object array
                    Current group
                new_group : object array
                    New group positions
                init_flag : bool
                    Whether this is initial evaluation
                    
            Returns:
                evaluated_group : object array
                    Evaluated group
                best_member : MultiObjectiveMember
                    Best member in group
            %}
            
            group_size = length(group);
            
            if init_flag
                % Initial evaluation
                for j = 1:group_size
                    fitness = obj.objective_func(group(j).position);
                    group(j).multi_fitness = fitness(:).';
                end
            else
                % Update evaluation
                for j = 1:group_size
                    new_fitness = obj.objective_func(new_group(j).position);
                    group(j).multi_fitness = new_fitness(:).';
                    group(j).position = new_group(j).position;
                end
            end
            
            % Find best in group based on random fitness for selection
            best_member = obj.select_best_from_group(group);
            
            evaluated_group = group;
        end
        
        function best_member = select_best_from_group(obj, group)
            %{
            select_best_from_group - Select best member from group using random fitness for diversity.
            
            Inputs:
                group : object array
                    Group to select from
                    
            Returns:
                best_member : MultiObjectiveMember
                    Best member in group
            %}
            
            if isempty(group)
                best_member = [];
                return;
            end
            
            % Use sum of fitness for selection (simple approach)
            fitness_values = zeros(1, length(group));
            for i = 1:length(group)
                fitness_values(i) = sum(group(i).multi_fitness);
            end
            
            if obj.maximize
                [~, best_idx] = max(fitness_values);
            else
                [~, best_idx] = min(fitness_values);
            end
            
            best_member = group(best_idx);
        end
        
        function S = update_variables(obj, search_agents_no, iter, max_iter, K, P, C)
            %{
            update_variables - Update solubility variables.
            
            Inputs:
                search_agents_no : int
                    Number of search agents
                iter : int
                    Current iteration
                max_iter : int
                    Maximum iterations
                K : array
                    Henry's constants
                P : array
                    Partial pressures
                C : array
                    Constants
                    
            Returns:
                S : array
                    Solubility values
            %}
            
            T = exp(-iter / max_iter);  % Temperature
            T0 = 298.15;  % Reference temperature
            
            group_size = floor(search_agents_no / obj.n_types);
            S = zeros(1, search_agents_no);  % Solubility
            
            for j = 1:obj.n_types
                % Update Henry's constant
                K(j) = K(j) * exp(-C(j) * (1/T - 1/T0));
                
                % Update solubility for this group
                start_idx = (j-1) * group_size + 1;
                end_idx = j * group_size;
                S(start_idx:end_idx) = P(start_idx:end_idx) * K(j);
            end
        end
        
        function new_groups = update_positions_multi(obj, groups, group_best_members, leader, S, search_agents_no)
            %{
            update_positions_multi - Update particle positions for multi-objective optimization.
            
            Inputs:
                groups : cell array
                    Current groups
                group_best_members : object array
                    Best members in each group
                leader : MultiObjectiveMember
                    Leader from archive
                S : array
                    Solubility values
                search_agents_no : int
                    Number of search agents
                    
            Returns:
                new_groups : cell array
                    New groups with updated positions
            %}
            
            new_groups = cell(1, obj.n_types);
            group_size = floor(search_agents_no / obj.n_types);
            flag_options = [1, -1];  % Direction flags
            
            % If no leader available, use a random member from archive
            if isempty(leader) && ~isempty(obj.archive)
                leader_idx = randi(length(obj.archive));
                leader = obj.archive(leader_idx);
            elseif isempty(leader)
                % If no archive, use best from first group
                if ~isempty(group_best_members) && ~isempty(group_best_members{1})
                    leader = group_best_members{1};
                else
                    leader = groups{1}(1);
                end
            end
            
            for i = 1:obj.n_types
                new_group = repmat(MultiObjectiveMember(0, 0), 1, group_size);
                group_best = group_best_members{i};
                if isempty(group_best)
                    group_best = groups{i}(1);
                end
                
                for j = 1:group_size
                    % Calculate gamma parameter using random fitness
                    current_fitness = sum(groups{i}(j).multi_fitness);
                    leader_fitness = sum(leader.multi_fitness);
                    gamma = obj.beta * exp(...
                        -(leader_fitness + 0.05) / (current_fitness + 0.05)...
                    );
                    
                    % Random direction flag
                    flag_idx = randi(2);
                    direction_flag = flag_options(flag_idx);
                    
                    % Update position
                    new_position = groups{i}(j).position;
                    for k = 1:obj.dim
                        % Group best influence
                        group_best_influence = direction_flag * rand() * gamma * ...
                                              (group_best.position(k) - groups{i}(j).position(k));
                        
                        % Leader influence
                        leader_influence = rand() * obj.alpha * direction_flag * ...
                                          (S((i-1)*group_size + j) * leader.position(k) - groups{i}(j).position(k));
                        
                        new_position(k) = new_position(k) + group_best_influence + leader_influence;
                    end
                    
                    new_group(j) = MultiObjectiveMember(new_position, zeros(1, obj.n_objectives));
                end
                
                new_groups{i} = new_group;
            end
        end
        
        function groups = check_positions_multi(obj, groups)
            %{
            check_positions_multi - Ensure positions stay within bounds for multi-objective.
            
            Inputs:
                groups : cell array
                    Groups to check
                    
            Returns:
                groups : cell array
                    Groups with bounded positions
            %}
            
            for i = 1:obj.n_types
                for j = 1:length(groups{i})
                    groups{i}(j).position = max(min(groups{i}(j).position, obj.ub), obj.lb);
                end
            end
        end
        
        function group = worst_agents_multi(obj, group)
            %{
            worst_agents_multi - Replace worst agents in group for multi-objective optimization.
            
            Inputs:
                group : object array
                    Group to process
                    
            Returns:
                group : object array
                    Group with worst agents replaced
            %}
            
            group_size = length(group);
            
            % Calculate number of worst agents to replace
            M1N = obj.M1 * group_size;
            M2N = obj.M2 * group_size;
            Nw = round((M2N - M1N) * rand() + M1N);
            
            if Nw > 0
                % Sort by fitness (worst first)
                fitness_values = zeros(1, group_size);
                for i = 1:group_size
                    fitness_values(i) = sum(group(i).multi_fitness);
                end
                
                if obj.maximize
                    [~, sorted_indices] = sort(fitness_values);  % Ascending for maximization
                else
                    [~, sorted_indices] = sort(fitness_values, 'descend');  % Descending for minimization
                end
                
                % Replace worst agents with random positions
                for k = 1:min(Nw, group_size)
                    worst_idx = sorted_indices(k);
                    new_position = obj.lb + (obj.ub - obj.lb) .* rand(1, obj.dim);
                    new_fitness = obj.objective_func(new_position);
                    group(worst_idx).position = new_position;
                    group(worst_idx).multi_fitness = new_fitness(:).';
                end
            end
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective HGSO.
            
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
            
            % Initialize algorithm parameters
            K = obj.l1 * rand(1, obj.n_types);  % Henry's constants
            P = obj.l2 * rand(1, search_agents_no);  % Partial pressures
            C = obj.l3 * rand(1, obj.n_types);  % Constants
            
            % Create groups
            groups = obj.create_groups(population);
            
            % Evaluate initial groups
            group_best_members = cell(1, obj.n_types);
            
            for i = 1:obj.n_types
                [groups{i}, group_best_members{i}] = obj.evaluate_group_multi(...
                    groups{i}, [], true...
                );
            end
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Update variables (solubility)
                S = obj.update_variables(search_agents_no, iter, max_iter, K, P, C);
                
                % Select leader from archive using grid-based selection
                leader = obj.select_leader();
                
                % Update positions
                new_groups = obj.update_positions_multi(...
                    groups, group_best_members, leader, S, search_agents_no...
                );
                
                % Ensure positions stay within bounds
                new_groups = obj.check_positions_multi(new_groups);
                
                % Evaluate new groups
                new_population = repmat(MultiObjectiveMember(0, 0), 1, 0);
                for i = 1:obj.n_types
                    [evaluated_group, ~] = obj.evaluate_group_multi(...
                        groups{i}, new_groups{i}, false...
                    );
                    new_population = [new_population, evaluated_group];
                end
                
                % Update archive with new population
                obj = obj.add_to_archive(new_population);
                
                % Update groups with evaluated groups
                for i = 1:obj.n_types
                    [groups{i}, group_best_members{i}] = obj.evaluate_group_multi(...
                        groups{i}, new_groups{i}, false...
                    );
                    
                    % Replace worst agents
                    groups{i} = obj.worst_agents_multi(groups{i});
                end
                
                % Store archive state for history
                archive_copy = cell(1, length(obj.archive));
                for idx = 1:length(obj.archive)
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
    end
end
