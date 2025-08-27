classdef MultiObjectiveShuffledFrogLeapingOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveShuffledFrogLeapingOptimizer - Multi-Objective Shuffled Frog Leaping Optimizer
    
    This algorithm extends the standard SFLA for multi-objective optimization
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
        - n_memeplex: Number of memeplexes (default: 5)
        - memeplex_size: Size of each memeplex (default: 10)
        - fla_q: Number of parents in FLA (default: 30% of memeplex size)
        - fla_alpha: Number of offsprings in FLA (default: 3)
        - fla_beta: Maximum iterations in FLA (default: 5)
        - fla_sigma: Step size in FLA (default: 2.0)
    %}
    
    properties
        n_memeplex
        memeplex_size
        fla_q
        fla_alpha
        fla_beta
        fla_sigma
    end
    
    methods
        function obj = MultiObjectiveShuffledFrogLeapingOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveShuffledFrogLeapingOptimizer constructor - Initialize the MO-SFLA solver
            
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
            obj.name_solver = "Multi-Objective Shuffled Frog Leaping Optimizer";
            
            % Set SFLA parameters with defaults
            obj.n_memeplex = obj.get_kw('n_memeplex', 5);
            obj.memeplex_size = obj.get_kw('memeplex_size', 10);
            obj.fla_q = obj.get_kw('fla_q', []);  % Number of parents
            obj.fla_alpha = obj.get_kw('fla_alpha', 3);  % Number of offsprings
            obj.fla_beta = obj.get_kw('fla_beta', 5);  % Maximum FLA iterations
            obj.fla_sigma = obj.get_kw('fla_sigma', 2.0);  % Step size
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective SFLA
            
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
            
            % Initialize parameters
            if isempty(obj.fla_q)
                % Default: 30% of memeplex size, at least 2
                obj.fla_q = max(round(0.3 * obj.memeplex_size), 2);
            end
            
            % Ensure memeplex size is at least dimension + 1 (Nelder-Mead standard)
            obj.memeplex_size = max(obj.memeplex_size, obj.dim + 1);
            
            % Calculate total population size
            total_pop_size = obj.n_memeplex * obj.memeplex_size;
            if total_pop_size ~= search_agents_no
                fprintf("Warning: Adjusted population size from %d to %d " + ...
                      "to match memeplex structure (%d memeplexes × %d frogs)\n", ...
                      search_agents_no, total_pop_size, obj.n_memeplex, obj.memeplex_size);
                search_agents_no = total_pop_size;
            end
            
            % Initialize storage
            history_archive = {};
            
            % Initialize population
            population = obj.init_population(search_agents_no);
            
            % Initialize archive with non-dominated solutions
            obj.determine_domination(population);
            non_dominated = obj.get_non_dominated_particles(population);
            obj.archive = [obj.archive, non_dominated];
            
            % Initialize grid for archive
            if ~isempty(obj.archive)
                costs = obj.get_fitness(obj.archive);
                if ~isempty(costs)
                    obj.grid = obj.create_hypercubes(costs);
                    for k = 1:numel(obj.archive)
                        [gi, gs] = obj.get_grid_index(obj.archive(k));
                        obj.archive(k).grid_index = gi;
                        obj.archive(k).grid_sub_index = gs;
                    end
                end
            end
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Create memeplex indices
            memeplex_indices = reshape(1:search_agents_no, obj.n_memeplex, obj.memeplex_size);
            
            % Main SFLA loop
            for iter = 1:max_iter
                % Shuffle population (main SFLA step)
                population = population(randperm(numel(population)));
                
                % Process each memeplex
                for j = 1:obj.n_memeplex
                    % Extract memeplex
                    start_idx = (j - 1) * obj.memeplex_size + 1;
                    end_idx = j * obj.memeplex_size;
                    memeplex = population(start_idx:end_idx);
                    
                    % Run FLA on memeplex
                    updated_memeplex = obj.run_fla(memeplex);
                    
                    % Update population
                    population(start_idx:end_idx) = updated_memeplex;
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
        
        function in_range = is_in_range(obj, x)
            %{
            is_in_range - Check if position is within variable bounds.
            
            Inputs:
                x : array
                    Position to check
                    
            Returns:
                in_range : bool
                    True if position is within bounds, False otherwise
            %}
            in_range = all(x >= obj.lb) && all(x <= obj.ub);
        end
        
        function selected_indices = rand_sample(obj, probabilities, q, replacement)
            %{
            rand_sample - Random sampling with probabilities.
            
            Inputs:
                probabilities : array
                    Selection probabilities
                q : int
                    Number of samples to draw
                replacement : bool
                    Whether to sample with replacement
                    
            Returns:
                selected_indices : array
                    List of selected indices
            %}
            if nargin < 4
                replacement = false;
            end
            
            selected_indices = [];
            current_probs = probabilities(:)';
            
            for sample_idx = 1:q
                % Normalize probabilities
                if sum(current_probs) == 0
                    % If all probabilities are zero, use uniform distribution
                    current_probs = ones(1, numel(current_probs)) / numel(current_probs);
                else
                    current_probs = current_probs / sum(current_probs);
                end
                
                % Select one index
                r = rand();
                cumulative_sum = cumsum(current_probs);
                selected_idx = find(r <= cumulative_sum, 1, 'first');
                selected_indices = [selected_indices, selected_idx];
                
                if ~replacement
                    % Set probability to zero for selected index
                    current_probs(selected_idx) = 0;
                end
            end
        end
        
        function updated_memeplex = run_fla(obj, memeplex)
            %{
            run_fla - Run Frog Leaping Algorithm on a memeplex for multi-objective optimization.
            
            Inputs:
                memeplex : cell array
                    Current memeplex to optimize
                    
            Returns:
                updated_memeplex : cell array
                    Updated memeplex after FLA (same size as input)
            %}
            n_pop = numel(memeplex);
            
            if n_pop == 0
                updated_memeplex = MultiObjectiveMember.empty;
                return;
            end
            
            % Calculate selection probabilities (rank-based)
            % For multi-objective, we use grid-based diversity for ranking
            if ~isempty(obj.archive) && numel(obj.archive) > 0
                % Calculate diversity scores based on grid occupancy
                grid_counts = containers.Map('KeyType', 'double', 'ValueType', 'double');
                for frog_idx = 1:numel(memeplex)
                    frog = memeplex(frog_idx);
                    if ~isempty(frog.grid_index)
                        if isKey(grid_counts, frog.grid_index)
                            grid_counts(frog.grid_index) = grid_counts(frog.grid_index) + 1;
                        else
                            grid_counts(frog.grid_index) = 1;
                        end
                    end
                end
                
                % Higher probability for frogs in less crowded grid cells
                selection_probs = ones(1, n_pop);
                for frog_idx = 1:n_pop
                    frog = memeplex(frog_idx);
                    if ~isempty(frog.grid_index) && isKey(grid_counts, frog.grid_index)
                        selection_probs(frog_idx) = 1.0 / (grid_counts(frog.grid_index) + 1);
                    end
                end
            else
                % Fallback: uniform probabilities
                selection_probs = ones(1, n_pop) / n_pop;
            end
            
            % Calculate population range (smallest hypercube)
            positions = zeros(n_pop, obj.dim);
            for frog_idx = 1:n_pop
                positions(frog_idx, :) = memeplex(frog_idx).position;
            end
            lower_bound = min(positions, [], 1);
            upper_bound = max(positions, [], 1);
            
            % FLA main loop - work on the entire memeplex, not just parents
            updated_memeplex = memeplex;  % Start with original memeplex
            
            for fla_iter = 1:obj.fla_beta
                % Select parents from the current memeplex
                parent_indices = obj.rand_sample(selection_probs, obj.fla_q);
                parents = updated_memeplex(parent_indices);
                
                % Generate offsprings
                for offspring_idx = 1:obj.fla_alpha
                    % For multi-objective, we need to select leaders from archive
                    % Select multiple leaders from archive for guidance
                    leaders = obj.select_multiple_leaders(3);  % Select 3 leaders
                    
                    % If we don't have enough leaders, use random frogs from memeplex
                    if numel(leaders) < 3
                        available_frogs = setdiff(updated_memeplex, leaders);
                        needed = 3 - numel(leaders);
                        if ~isempty(available_frogs)
                            additional = available_frogs(randperm(numel(available_frogs), min(needed, numel(available_frogs))));
                            leaders = [leaders, additional];
                        end
                    end
                    
                    % Ensure we have exactly 3 leaders
                    if numel(leaders) > 3
                        leaders = leaders(1:3);
                    end
                    
                    % Select a random parent to improve
                    worst_idx = parent_indices(randi(numel(parent_indices)));
                    worst_parent = updated_memeplex(worst_idx);
                    
                    % Flags for improvement steps
                    improvement_step2 = false;
                    censorship = false;
                    
                    % Improvement Step 1: Move towards best leader
                    new_sol_1 = worst_parent.copy();
                    step = obj.fla_sigma * rand(1, obj.dim) .* (...
                        leaders(1).position - worst_parent.position...
                    );
                    new_sol_1.position = worst_parent.position + step;
                    
                    if obj.is_in_range(new_sol_1.position)
                        new_sol_1.multi_fitness = obj.objective_func(new_sol_1.position);
                        new_sol_1.multi_fitness = new_sol_1.multi_fitness(:).';
                        % For multi-objective, we check if new solution is non-dominated
                        if ~obj.dominates(worst_parent, new_sol_1)  % New solution is not worse
                            updated_memeplex(worst_idx) = new_sol_1;
                        else
                            improvement_step2 = true;
                        end
                    else
                        improvement_step2 = true;
                    end
                    
                    % Improvement Step 2: Move towards other leaders
                    if improvement_step2
                        new_sol_2 = worst_parent.copy();
                        % Choose a different leader (not the first one)
                        leader_idx = randi([2, numel(leaders)]);
                        step = obj.fla_sigma * rand(1, obj.dim) .* (...
                            leaders(leader_idx).position - worst_parent.position...
                        );
                        new_sol_2.position = worst_parent.position + step;
                        
                        if obj.is_in_range(new_sol_2.position)
                            new_sol_2.multi_fitness = obj.objective_func(new_sol_2.position);
                            new_sol_2.multi_fitness = new_sol_2.multi_fitness(:).';
                            if ~obj.dominates(worst_parent, new_sol_2)  % New solution is not worse
                                updated_memeplex(worst_idx) = new_sol_2;
                            else
                                censorship = true;
                            end
                        else
                            censorship = true;
                        end
                    end
                    
                    % Censorship: Replace with random solution
                    if censorship
                        random_position = lower_bound + (upper_bound - lower_bound) .* rand(1, obj.dim);
                        random_fitness = obj.objective_func(random_position);
                        random_fitness = random_fitness(:).';
                        updated_memeplex(worst_idx) = MultiObjectiveMember(random_position, random_fitness);
                    end
                end
            end
        end
    end
end
