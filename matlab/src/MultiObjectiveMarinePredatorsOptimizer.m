classdef MultiObjectiveMarinePredatorsOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveMarinePredatorsOptimizer - Multi-Objective Marine Predators Algorithm (MPA) Optimizer.
    
    Multi-objective version of the Marine Predators Algorithm that handles
    optimization problems with multiple conflicting objectives. The algorithm
    maintains an archive of non-dominated solutions and uses grid-based
    selection to maintain diversity in the Pareto front.
    
    The algorithm follows the same three phases as the single-objective version:
    1. Phase 1 (Iter < Max_iter/3): High velocity ratio - Brownian motion
    2. Phase 2 (Max_iter/3 < Iter < 2*Max_iter/3): Unit velocity ratio - Mixed strategy
    3. Phase 3 (Iter > 2*Max_iter/3): Low velocity ratio - Levy flight
    
    References:
        Faramarzi, A., Heidarinejad, M., Mirjalili, S., & Gandomi, A. H. (2020).
        Marine Predators Algorithm: A Nature-inspired Metaheuristic.
        Expert Systems with Applications, 152, 113377.
    
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
        Additional algorithm parameters including:
        - FADs: Fish Aggregating Devices effect probability (default: 0.2)
        - P: Memory rate parameter (default: 0.5)
    %}
    
    properties
        FADs  % Fish Aggregating Devices effect probability
        P     % Memory rate parameter
    end
    
    methods
        function obj = MultiObjectiveMarinePredatorsOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveMarinePredatorsOptimizer constructor - Initialize the MOMPA solver
            
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
            obj.name_solver = "Multi-Objective Marine Predators Algorithm";
            
            % Set algorithm parameters with defaults
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
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Execute the Multi-Objective Marine Predators Algorithm.
            
            The algorithm maintains an archive of non-dominated solutions and uses
            grid-based selection to maintain diversity in the Pareto front.
            
            Inputs:
                search_agents_no : int
                    Number of search agents (predators/prey)
                max_iter : int
                    Maximum number of iterations for optimization
                    
            Returns:
                history_archive : cell array
                    History of archive states
                archive : cell array
                    Final archive of non-dominated solutions
            %}
            
            % Initialize the population of search agents
            population = obj.init_population(search_agents_no);
            
            % Initialize storage for archive history
            history_archive = {};
            
            % Memory for previous population
            Prey_old = repmat(MultiObjectiveMember(0, 0), 1, search_agents_no);
            for i = 1:search_agents_no
                Prey_old(i) = population(i).copy();
            end
            
            % Initialize top predator (for guidance)
            Top_predator_pos = zeros(1, obj.dim);
            if obj.maximize
                Top_predator_fit = -inf(1, obj.n_objectives);
            else
                Top_predator_fit = inf(1, obj.n_objectives);
            end
            
            % Boundary matrices for FADs effect
            Xmin = repmat(obj.lb, search_agents_no, 1);
            Xmax = repmat(obj.ub, search_agents_no, 1);
            
            % Initialize archive with non-dominated solutions
            obj.determine_domination(population);
            non_dominated = obj.get_non_dominated_particles(population);
            obj.archive = [obj.archive, non_dominated];
            
            % Build initial grid
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
                % ------------------- Detecting top predator -------------------
                for i = 1:search_agents_no
                    % Ensure positions stay within bounds
                    population(i).position = max(min(population(i).position, obj.ub), obj.lb);
                    population(i).multi_fitness = obj.objective_func(population(i).position);
                    
                    % Update top predator (for single best guidance)
                    top_predator_member = MultiObjectiveMember(Top_predator_pos, Top_predator_fit);
                    if obj.dominates(population(i), top_predator_member)
                        Top_predator_fit = population(i).multi_fitness;
                        Top_predator_pos = population(i).position;
                    end
                end
                
                % ------------------- Marine Memory saving -------------------
                if iter == 1
                    Prey_old = repmat(MultiObjectiveMember(0, 0), 1, search_agents_no);
                    for i = 1:search_agents_no
                        Prey_old(i) = population(i).copy();
                    end
                end
                
                % Update population based on memory (Pareto dominance)
                Inx = false(1, search_agents_no);
                for i = 1:search_agents_no
                    if obj.dominates(Prey_old(i), population(i))
                        Inx(i) = true;
                    end
                end
                
                % Update positions based on memory
                for i = 1:search_agents_no
                    if Inx(i)
                        population(i).position = Prey_old(i).position;
                        population(i).multi_fitness = Prey_old(i).multi_fitness;
                    end
                end
                
                % Update memory
                for i = 1:search_agents_no
                    Prey_old(i) = population(i).copy();
                end
                % ------------------------------------------------------------
                
                % Create elite matrix (replicate top predator)
                Elite = repmat(Top_predator_pos, search_agents_no, 1);
                
                % Compute convergence factor
                CF = (1 - iter / max_iter) ^ (2 * iter / max_iter);
                
                % Generate random vectors
                RL = 0.05 * obj.levy_flight(search_agents_no, obj.dim, 1.5);  % Levy flight
                RB = randn(search_agents_no, obj.dim);                         % Brownian motion
                
                % Get current positions
                positions = zeros(search_agents_no, obj.dim);
                for i = 1:search_agents_no
                    positions(i, :) = population(i).position;
                end
                
                % Update positions based on current phase
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
                    population(i).position = positions(i, :);
                    % Re-evaluate fitness after position update
                    population(i).multi_fitness = obj.objective_func(population(i).position);
                end
                
                % ------------------- Detecting top predator -------------------
                for i = 1:search_agents_no
                    % Ensure positions stay within bounds
                    population(i).position = max(min(population(i).position, obj.ub), obj.lb);
                    population(i).multi_fitness = obj.objective_func(population(i).position);
                    
                    % Update top predator
                    top_predator_member = MultiObjectiveMember(Top_predator_pos, Top_predator_fit);
                    if obj.dominates(population(i), top_predator_member)
                        Top_predator_fit = population(i).multi_fitness;
                        Top_predator_pos = population(i).position;
                    end
                end
                
                % ----------------------- Marine Memory saving ----------------
                % Update population based on memory (Pareto dominance)
                Inx = false(1, search_agents_no);
                for i = 1:search_agents_no
                    if obj.dominates(Prey_old(i), population(i))
                        Inx(i) = true;
                    end
                end
                
                % Update positions based on memory
                for i = 1:search_agents_no
                    if Inx(i)
                        population(i).position = Prey_old(i).position;
                        population(i).multi_fitness = Prey_old(i).multi_fitness;
                    end
                end
                
                % Update memory
                for i = 1:search_agents_no
                    Prey_old(i) = population(i).copy();
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
                
                % Update population positions and re-evaluate fitness
                for i = 1:search_agents_no
                    population(i).position = positions(i, :);
                    population(i).multi_fitness = obj.objective_func(population(i).position);
                end
                
                % Update archive with new solutions
                obj = obj.add_to_archive(population);
                
                % Store archive history
                archive_copy = cell(1, length(obj.archive));
                for idx = 1:length(obj.archive)
                    archive_copy{idx} = obj.archive(idx).copy();
                end
                history_archive{end+1} = archive_copy;
                
                % Update progress
                leader = obj.select_leader();
                obj.callbacks(iter, max_iter, leader);
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
