classdef ShuffledFrogLeapingOptimizer < Solver
    %{
    Shuffled Frog Leaping Algorithm (SFLA) optimizer.
    
    SFLA is a memetic metaheuristic that combines elements of particle swarm
    optimization and shuffled complex evolution. It works by dividing the
    population into memeplexes and performing local search within each memeplex.
    
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
        Additional SFLA parameters:
        - n_memeplex: Number of memeplexes (default: 5)
        - memeplex_size: Size of each memeplex (default: 10)
        - fla_q: Number of parents in FLA (default: 30% of memeplex size)
        - fla_alpha: Number of offsprings in FLA (default: 3)
        - fla_beta: Maximum iterations in FLA (default: 5)
        - fla_sigma: Step size in FLA (default: 2.0)
    %}
    
    properties
        n_memeplex    % Number of memeplexes
        memeplex_size % Size of each memeplex
        fla_q         % Number of parents in FLA
        fla_alpha     % Number of offsprings in FLA
        fla_beta      % Maximum iterations in FLA
        fla_sigma     % Step size in FLA
    end
    
    methods
        function obj = ShuffledFrogLeapingOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            ShuffledFrogLeapingOptimizer constructor - Initialize the SFLA solver
            
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
                    Additional SFLA parameters:
                    - n_memeplex: Number of memeplexes (default: 5)
                    - memeplex_size: Size of each memeplex (default: 10)
                    - fla_q: Number of parents in FLA (default: 30% of memeplex size)
                    - fla_alpha: Number of offsprings in FLA (default: 3)
                    - fla_beta: Maximum iterations in FLA (default: 5)
                    - fla_sigma: Step size in FLA (default: 2.0)
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set algorithm name
            obj.name_solver = "Shuffled Frog Leaping Optimizer";
            
            % kwargs (Name-Value pairs)
            if ~isempty(varargin)
                obj.kwargs = obj.nv2struct(varargin{:});
            end
            obj.n_memeplex = obj.get_kw('n_memeplex', 5);
            obj.memeplex_size = obj.get_kw('memeplex_size', 10);
            obj.fla_q = obj.get_kw('fla_q', []);  % Number of parents
            obj.fla_alpha = obj.get_kw('fla_alpha', 3);  % Number of offsprings
            obj.fla_beta = obj.get_kw('fla_beta', 5);    % Maximum FLA iterations
            obj.fla_sigma = obj.get_kw('fla_sigma', 2.0); % Step size
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for SFLA
            
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
            
            % Initialize population
            population = obj.init_population(search_agents_no);
            
            % Initialize best solution
            sorted_population = obj.sort_population(population);
            best_solver = sorted_population{1}.copy();
            
            % Initialize history
            history_step_solver = {};
            
            % Begin optimization
            obj.begin_step_solver(max_iter);
            
            % Main SFLA loop
            for iter = 1:max_iter
                % Shuffle population (main SFLA step)
                population = population(randperm(search_agents_no));
                
                % Process each memeplex
                for j = 1:obj.n_memeplex
                    % Extract memeplex
                    start_idx = (j - 1) * obj.memeplex_size + 1;
                    end_idx = j * obj.memeplex_size;
                    memeplex = population(start_idx:end_idx);
                    
                    % Run FLA on memeplex
                    updated_memeplex = obj.run_fla(memeplex, best_solver);
                    
                    % Update population
                    population(start_idx:end_idx) = updated_memeplex;
                end
                
                % Sort population and update best solution
                sorted_population = obj.sort_population(population);
                current_best = sorted_population{1};
                
                if obj.is_better(current_best, best_solver)
                    best_solver = current_best.copy();
                end
                
                % Store history
                history_step_solver{end+1} = best_solver.copy();
                
                % Call callbacks
                obj.callbacks(iter, max_iter, best_solver);
            end
            
            % Finalize optimization
            obj.history_step_solver = history_step_solver;
            obj.best_solver = best_solver;
            obj.end_step_solver();
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
        
        function in_range = is_in_range(obj, x)
            %{
            is_in_range - Check if position is within variable bounds
            
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
            rand_sample - Random sampling with probabilities
            
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
            current_probs = probabilities;
            
            for sample_idx = 1:q
                % Normalize probabilities
                if sum(current_probs) == 0
                    % If all probabilities are zero, use uniform distribution
                    current_probs = ones(size(current_probs)) / length(current_probs);
                else
                    current_probs = current_probs / sum(current_probs);
                end
                
                % Select one index
                r = rand();
                cumulative_sum = cumsum(current_probs);
                selected_idx = find(r <= cumulative_sum, 1);
                selected_indices = [selected_indices, selected_idx];
                
                if ~replacement
                    % Set probability to zero for selected index
                    current_probs(selected_idx) = 0;
                end
            end
        end
        
        function updated_memeplex = run_fla(obj, memeplex, best_solver)
            %{
            run_fla - Run Frog Leaping Algorithm on a memeplex
            
            Inputs:
                memeplex : cell array
                    Current memeplex to optimize
                best_solver : Member
                    Global best solution
                    
            Returns:
                updated_memeplex : cell array
                    Updated memeplex after FLA
            %}
            
            n_pop = length(memeplex);
            
            % Calculate selection probabilities (rank-based)
            ranks = n_pop:-1:1;  % Higher rank for better solutions
            selection_probs = 2 * (n_pop + 1 - ranks) / (n_pop * (n_pop + 1));
            
            % Calculate population range (smallest hypercube)
            positions = obj.get_positions(memeplex);
            lower_bound = min(positions, [], 1);
            upper_bound = max(positions, [], 1);
            
            % FLA main loop
            for fla_iter = 1:obj.fla_beta
                % Select parents
                parent_indices = obj.rand_sample(selection_probs, obj.fla_q);
                parents = memeplex(parent_indices);
                
                % Generate offsprings
                for alpha_iter = 1:obj.fla_alpha
                    % Sort parents (best to worst)
                    sorted_parents = obj.sort_population(parents);
                    
                    % Get worst parent
                    worst_parent = sorted_parents{end};
                    worst_idx = parent_indices(end);
                    
                    % Flags for improvement steps
                    improvement_step2 = false;
                    censorship = false;
                    
                    % Improvement Step 1: Move towards best parent
                    new_sol_1 = worst_parent.copy();
                    step = obj.fla_sigma * rand(1, obj.dim) .* ...
                          (sorted_parents{1}.position - worst_parent.position);
                    new_sol_1.position = worst_parent.position + step;
                    
                    if obj.is_in_range(new_sol_1.position)
                        new_sol_1.fitness = obj.objective_func(new_sol_1.position);
                        if obj.is_better(new_sol_1, worst_parent)
                            memeplex{worst_idx} = new_sol_1;
                        else
                            improvement_step2 = true;
                        end
                    else
                        improvement_step2 = true;
                    end
                    
                    % Improvement Step 2: Move towards global best
                    if improvement_step2
                        new_sol_2 = worst_parent.copy();
                        step = obj.fla_sigma * rand(1, obj.dim) .* ...
                              (best_solver.position - worst_parent.position);
                        new_sol_2.position = worst_parent.position + step;
                        
                        if obj.is_in_range(new_sol_2.position)
                            new_sol_2.fitness = obj.objective_func(new_sol_2.position);
                            if obj.is_better(new_sol_2, worst_parent)
                                memeplex{worst_idx} = new_sol_2;
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
                        memeplex{worst_idx} = Member(random_position, random_fitness);
                    end
                end
            end
            
            updated_memeplex = memeplex;
        end
        
        function s = get_kw(obj, name, default)
            if isfield(obj.kwargs, name), s = obj.kwargs.(name);
            else, s = default; end
        end
    end
end
