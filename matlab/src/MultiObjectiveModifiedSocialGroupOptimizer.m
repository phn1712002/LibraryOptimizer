classdef MultiObjectiveModifiedSocialGroupOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveModifiedSocialGroupOptimizer - Multi-Objective Modified Social Group Optimization (MSGO) algorithm.
    
    A metaheuristic optimization algorithm inspired by social group behavior,
    featuring guru phase (learning from best) and learner phase (mutual learning)
    adapted for multi-objective optimization using archive management and grid-based selection.
    
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
        - c: Learning coefficient for guru phase (default: 0.2)
        - sap: Self-adaptive probability for random exploration (default: 0.7)
    %}
    
    properties
        c
        sap
    end
    
    methods
        function obj = MultiObjectiveModifiedSocialGroupOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveModifiedSocialGroupOptimizer constructor - Initialize the MO-MSGO solver
            
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
            obj.name_solver = "Multi-Objective Modified Social Group Optimizer";
            
            % MSGO-specific parameters
            obj.c = obj.get_kw('c', 0.2);  % Learning coefficient for guru phase
            obj.sap = obj.get_kw('sap', 0.7);  % Self-adaptive probability for random exploration
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective MSGO
            
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
                % Phase 1: Guru Phase (improvement due to best person)
                % Select guru from archive using grid-based selection
                guru = obj.select_leader();
                if isempty(guru)
                    % If no leader in archive, use random member from population
                    guru = population(randi(numel(population)));
                end
                
                % Create new population for guru phase
                new_population = MultiObjectiveMember.empty;
                for i = 1:numel(population)
                    new_position = zeros(1, obj.dim);
                    for j = 1:obj.dim
                        % Update position: c*current + rand*(guru - current)
                        new_position(j) = (obj.c * population(i).position(j) + ...
                                          rand() * (guru.position(j) - population(i).position(j)));
                    end
                    
                    % Apply bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Evaluate new fitness
                    new_fitness = obj.objective_func(new_position);
                    new_fitness = new_fitness(:).';
                    
                    % Create new member for comparison
                    new_member = MultiObjectiveMember(new_position, new_fitness);
                    
                    % Greedy selection: accept if new solution dominates current or is non-dominated
                    if obj.dominates(new_member, population(i)) || ~obj.dominates(population(i), new_member)
                        population(i).position = new_position;
                        population(i).multi_fitness = new_fitness;
                    end
                    
                    new_population = [new_population, population(i).copy()];
                end
                
                population = new_population;
                
                % Phase 2: Learner Phase (mutual learning with random exploration)
                % Select global best from archive for guidance
                global_best = obj.select_leader();
                if isempty(global_best)
                    % If no leader in archive, use random member from population
                    global_best = population(randi(numel(population)));
                end
                
                % Create new population for learner phase
                new_population_learner = MultiObjectiveMember.empty;
                for i = 1:numel(population)
                    % Choose a random partner different from current individual
                    r1 = randi(numel(population));
                    while r1 == i
                        r1 = randi(numel(population));
                    end
                    
                    % Check dominance between current and random partner
                    current_dominates_partner = obj.dominates(population(i), population(r1));
                    partner_dominates_current = obj.dominates(population(r1), population(i));
                    
                    if current_dominates_partner && ~partner_dominates_current
                        % Current individual dominates partner
                        if rand() > obj.sap
                            % Learning strategy: current + rand*(current - partner) + rand*(global_best - current)
                            new_position = zeros(1, obj.dim);
                            for j = 1:obj.dim
                                new_position(j) = (population(i).position(j) + ...
                                                  rand() * (population(i).position(j) - population(r1).position(j)) + ...
                                                  rand() * (global_best.position(j) - population(i).position(j)));
                            end
                        else
                            % Random exploration
                            new_position = obj.lb + (obj.ub - obj.lb) .* rand(1, obj.dim);
                        end
                    else
                        % Current individual is dominated by or non-dominated with partner
                        % Learning strategy: current + rand*(partner - current) + rand*(global_best - current)
                        new_position = zeros(1, obj.dim);
                        for j = 1:obj.dim
                            new_position(j) = (population(i).position(j) + ...
                                              rand() * (population(r1).position(j) - population(i).position(j)) + ...
                                              rand() * (global_best.position(j) - population(i).position(j)));
                        end
                    end
                    
                    % Apply bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Evaluate new fitness
                    new_fitness = obj.objective_func(new_position);
                    new_fitness = new_fitness(:).';
                    
                    % Create new member for comparison
                    new_member = MultiObjectiveMember(new_position, new_fitness);
                    
                    % Greedy selection: accept if new solution dominates current or is non-dominated
                    if obj.dominates(new_member, population(i)) || ~obj.dominates(population(i), new_member)
                        population(i).position = new_position;
                        population(i).multi_fitness = new_fitness;
                    end
                    
                    new_population_learner = [new_population_learner, population(i).copy()];
                end
                
                population = new_population_learner;
                
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
    end
end
