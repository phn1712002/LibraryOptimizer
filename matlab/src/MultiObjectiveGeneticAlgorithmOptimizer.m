classdef MultiObjectiveGeneticAlgorithmOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveGeneticAlgorithmOptimizer - Multi-Objective Genetic Algorithm Optimizer
    
    This algorithm extends the standard GA for multi-objective optimization
    using archive management and grid-based selection for parent selection.
    
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
        - num_groups: Number of independent populations (default: 5)
        - crossover_rate: Probability of crossover (default: 0.8)
        - mutation_rate: Probability of mutation (default: 0.1)
        - tournament_size: Size for tournament selection (default: 3)
    %}
    
    properties
        num_groups
        crossover_rate
        mutation_rate
        tournament_size
    end
    
    methods
        function obj = MultiObjectiveGeneticAlgorithmOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveGeneticAlgorithmOptimizer constructor - Initialize the MOGA solver
            
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
            obj.name_solver = "Multi-Objective Genetic Algorithm Optimizer";
            
            % Set GA-specific parameters with defaults
            obj.num_groups = obj.get_kw('num_groups', 5);
            obj.crossover_rate = obj.get_kw('crossover_rate', 0.8);
            obj.mutation_rate = obj.get_kw('mutation_rate', 0.1);
            obj.tournament_size = obj.get_kw('tournament_size', 3);
        end
        
        function child = uniform_crossover(obj, parent1, parent2)
            %{
            uniform_crossover - Perform uniform crossover between two parents
            
            Inputs:
                parent1 : MultiObjectiveMember
                    First parent
                parent2 : MultiObjectiveMember
                    Second parent
                    
            Returns:
                child : MultiObjectiveMember
                    Offspring
            %}
            
            child_position = zeros(1, obj.dim);
            
            for d = 1:obj.dim
                if rand() < 0.5  % 50% chance to inherit from parent1
                    child_position(d) = parent1.position(d);
                else
                    child_position(d) = parent2.position(d);
                end
            end
            
            % Ensure positions stay within bounds
            child_position = max(min(child_position, obj.ub), obj.lb);
            
            % Create child with evaluated fitness
            child_fitness = obj.objective_func(child_position);
            child_fitness = child_fitness(:).';
            child = MultiObjectiveMember(child_position, child_fitness);
        end
        
        function mutated_individual = mutation(obj, individual)
            %{
            mutation - Perform mutation on an individual
            
            Inputs:
                individual : MultiObjectiveMember
                    Individual to mutate
                    
            Returns:
                mutated_individual : MultiObjectiveMember
                    Mutated individual
            %}
            
            mutated_position = individual.position;
            
            for d = 1:obj.dim
                if rand() < obj.mutation_rate
                    % Gaussian mutation
                    mutation_strength = 0.1 * (obj.ub(d) - obj.lb(d));
                    mutated_position(d) = mutated_position(d) + randn() * mutation_strength;
                end
            end
            
            % Ensure positions stay within bounds
            mutated_position = max(min(mutated_position, obj.ub), obj.lb);
            
            % Create mutated individual with evaluated fitness
            mutated_fitness = obj.objective_func(mutated_position);
            mutated_fitness = mutated_fitness(:).';
            mutated_individual = MultiObjectiveMember(mutated_position, mutated_fitness);
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective GA
            
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
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Run multiple populations (as in original GA)
            for pop_idx = 1:obj.num_groups
                % Initialize population for this run
                population = obj.init_population(search_agents_no);
                
                % Initialize archive with non-dominated solutions from this population
                population = obj.determine_domination(population);
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
                
                % Main optimization loop for this population
                for iter = 1:max_iter
                    % Create new generation
                    new_population = [];
                    
                    % Elitism: keep the best individuals
                    population = obj.determine_domination(population);
                    non_dominated_pop = obj.get_non_dominated_particles(population);
                    elite_count = min(numel(non_dominated_pop), floor(search_agents_no / 4));
                    if elite_count > 0
                        new_population = [new_population, non_dominated_pop(1:elite_count)];
                    end
                    
                    % Generate offspring until we reach population size
                    while numel(new_population) < search_agents_no
                        % Selection
                        parent1 = obj.tournament_selection_multi(population, obj.tournament_size);
                        parent2 = obj.tournament_selection_multi(population, obj.tournament_size);
                        
                        % Crossover with probability
                        if rand() < obj.crossover_rate
                            child = obj.uniform_crossover(parent1, parent2);
                        else
                            % No crossover, select one parent randomly
                            if rand() < 0.5
                                child = parent1.copy();
                            else
                                child = parent2.copy();
                            end
                        end
                        
                        % Mutation
                        if rand() < obj.mutation_rate
                            child = obj.mutation(child);
                        end
                        
                        new_population = [new_population, child];
                    end
                    
                    % Ensure we have exactly the population size
                    population = new_population(1:search_agents_no);
                    
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
                    current_iter = iter + (pop_idx - 1) * max_iter;
                    total_iters = obj.num_groups * max_iter;
                    obj.callbacks(current_iter, total_iters, best_member);
                end
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
