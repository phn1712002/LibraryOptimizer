classdef MultiObjectiveEquilibriumOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveEquilibriumOptimizer - Multi-Objective Equilibrium Optimizer (EO) Algorithm.
    
    Multi-objective version of the Equilibrium Optimizer that handles
    optimization problems with multiple conflicting objectives. The algorithm
    maintains an archive of non-dominated solutions and uses grid-based
    selection to maintain diversity in the Pareto front.
    
    The algorithm uses an equilibrium pool consisting of:
    - 4 best candidates (Ceq1, Ceq2, Ceq3, Ceq4)
    - 1 average candidate (Ceq_ave)
    
    References:
        Faramarzi, A., Heidarinejad, M., Stephens, B., & Mirjalili, S. (2020).
        Equilibrium optimizer: A novel optimization algorithm.
        Knowledge-Based Systems, 191, 105190.
    
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
        - a1: Exploration parameter (default: 2)
        - a2: Exploitation parameter (default: 1)
        - GP: Generation probability (default: 0.5)
    %}
    
    properties
        a1  % Exploration parameter
        a2  % Exploitation parameter
        GP  % Generation probability
    end
    
    methods
        function obj = MultiObjectiveEquilibriumOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveEquilibriumOptimizer constructor - Initialize the MOEO solver
            
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
            obj.name_solver = "Multi-Objective Equilibrium Optimizer";
            
            % Set algorithm parameters with defaults
            obj.a1 = obj.get_kw('a1', 2);  % Exploration parameter
            obj.a2 = obj.get_kw('a2', 1);  % Exploitation parameter
            obj.GP = obj.get_kw('GP', 0.5);  % Generation probability
        end
        
        function population = init_population(obj, N)
            %{
            init_population - Initialize multi-objective equilibrium population
            
            Inputs:
                N : int
                    Number of particles to initialize
                    
            Returns:
                population : object array
                    Array of EquilibriumMultiMember objects
            %}
            pop_example = EquilibriumMultiMember(0, 0);
            population = repmat(pop_example, 1, N);
            for i = 1:N
                pos = obj.lb + (obj.ub - obj.lb).*rand(1, obj.dim);
                fit = obj.objective_func(pos); fit = fit(:).';
                population(i) = EquilibriumMultiMember(pos, fit);
            end
        end
        
        function leaders = select_multiple_leaders(obj, n_leaders)
            %{
            select_multiple_leaders - Select multiple diverse leaders from archive
            
            Inputs:
                n_leaders : int
                    Number of leaders to select
                    
            Returns:
                leaders : object array
                    Array of diverse leaders
            %}
            
            if isempty(obj.archive) || n_leaders <= 0
                leaders = repmat(EquilibriumMultiMember(0, 0), 1, 0);
                return;
            end
            
            % Simple approach: select the first n_leaders from archive
            % For better diversity, you might want to implement grid-based selection
            n_to_select = min(n_leaders, length(obj.archive));
            leaders = repmat(EquilibriumMultiMember(0, 0), 1, n_to_select);
            
            for i = 1:n_to_select
                leaders(i) = obj.archive(i).copy();
            end
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Execute the Multi-Objective Equilibrium Optimization Algorithm.
            
            The algorithm maintains an archive of non-dominated solutions and uses
            grid-based selection to maintain diversity in the Pareto front.
            
            Inputs:
                search_agents_no : int
                    Number of search agents
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
            
            % Initialize equilibrium candidates (for guidance)
            if obj.maximize
                inf_fit = -inf(1, obj.n_objectives);
            else
                inf_fit = inf(1, obj.n_objectives);
            end
            
            Ceq1 = EquilibriumMultiMember(zeros(1, obj.dim), inf_fit);
            Ceq2 = EquilibriumMultiMember(zeros(1, obj.dim), inf_fit);
            Ceq3 = EquilibriumMultiMember(zeros(1, obj.dim), inf_fit);
            Ceq4 = EquilibriumMultiMember(zeros(1, obj.dim), inf_fit);
            
            % Memory for previous population
            C_old = repmat(EquilibriumMultiMember(0, 0), 1, search_agents_no);
            for i = 1:search_agents_no
                C_old(i) = population(i).copy();
            end
            
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
                % Update equilibrium candidates (for single best guidance)
                for i = 1:search_agents_no
                    % Ensure positions stay within bounds
                    population(i).position = max(min(population(i).position, obj.ub), obj.lb);
                    population(i).multi_fitness = obj.objective_func(population(i).position);
                    
                    % Update equilibrium candidates based on Pareto dominance
                    if obj.dominates(population(i), Ceq1)
                        Ceq4 = Ceq3.copy();
                        Ceq3 = Ceq2.copy();
                        Ceq2 = Ceq1.copy();
                        Ceq1 = population(i).copy();
                    elseif ~obj.dominates(Ceq1, population(i)) && obj.dominates(population(i), Ceq2)
                        Ceq4 = Ceq3.copy();
                        Ceq3 = Ceq2.copy();
                        Ceq2 = population(i).copy();
                    elseif ~obj.dominates(Ceq2, population(i)) && obj.dominates(population(i), Ceq3)
                        Ceq4 = Ceq3.copy();
                        Ceq3 = population(i).copy();
                    elseif ~obj.dominates(Ceq3, population(i)) && obj.dominates(population(i), Ceq4)
                        Ceq4 = population(i).copy();
                    end
                end
                
                % ----------------- Memory saving -----------------
                if iter == 1
                    for i = 1:search_agents_no
                        C_old(i) = population(i).copy();
                    end
                end
                
                % Update population based on memory using Pareto dominance
                for i = 1:search_agents_no
                    if obj.dominates(C_old(i), population(i))
                        population(i).multi_fitness = C_old(i).multi_fitness;
                        population(i).position = C_old(i).position;
                    end
                end
                
                % Update memory
                for i = 1:search_agents_no
                    C_old(i) = population(i).copy();
                end
                % -------------------------------------------------
                
                % Create equilibrium pool from archive leaders
                if length(obj.archive) >= 4
                    % Select 4 diverse leaders from archive
                    leaders = obj.select_multiple_leaders(4);
                    Ceq1_arch = leaders(1);
                    Ceq2_arch = leaders(2);
                    Ceq3_arch = leaders(3);
                    Ceq4_arch = leaders(4);
                    
                    % Create average candidate
                    Ceq_ave_pos = (Ceq1_arch.position + Ceq2_arch.position + Ceq3_arch.position + Ceq4_arch.position) / 4;
                    Ceq_ave_fit = obj.objective_func(Ceq_ave_pos);
                    Ceq_ave = EquilibriumMultiMember(Ceq_ave_pos, Ceq_ave_fit);
                    
                    Ceq_pool = {Ceq1_arch, Ceq2_arch, Ceq3_arch, Ceq4_arch, Ceq_ave};
                else
                    % Fallback to original candidates if archive is small
                    Ceq_ave_pos = (Ceq1.position + Ceq2.position + Ceq3.position + Ceq4.position) / 4;
                    Ceq_ave_fit = obj.objective_func(Ceq_ave_pos);
                    Ceq_ave = EquilibriumMultiMember(Ceq_ave_pos, Ceq_ave_fit);
                    Ceq_pool = {Ceq1, Ceq2, Ceq3, Ceq4, Ceq_ave};
                end
                
                % Compute time parameter
                t = (1 - iter / max_iter) ^ (obj.a2 * iter / max_iter);
                
                % Update all search agents
                for i = 1:search_agents_no
                    % Randomly select one candidate from the pool
                    pool_idx = randi(length(Ceq_pool));
                    Ceq = Ceq_pool{pool_idx};
                    
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
                    G0 = GCP .* (Ceq.position - lambda_vec .* population(i).position);
                    G = G0 .* F;
                    
                    % Update position using EO equation
                    new_position = Ceq.position + ...
                                  (population(i).position - Ceq.position) .* F + ...
                                  (G ./ (lambda_vec * 1.0)) .* (1 - F);
                    
                    % Ensure positions stay within bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    population(i).position = new_position;
                    population(i).multi_fitness = obj.objective_func(new_position);
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
