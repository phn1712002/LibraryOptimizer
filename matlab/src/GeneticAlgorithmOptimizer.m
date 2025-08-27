classdef GeneticAlgorithmOptimizer < Solver
    %{
    Genetic Algorithm Optimizer implementation.
    
    This optimizer implements a genetic algorithm with uniform crossover,
    mutation, and natural selection based on the MATLAB implementation.
    
    Parameters:
    -----------
    objective_func : function handle
        Objective function to optimize
    lb : float or array
        Lower bounds for variables
    ub : float or array
        Upper bounds for variables  
    dim : int
        Problem dimension
    maximize : bool
        Whether to maximize (true) or minimize (false) objective
    varargin : cell array
        Additional algorithm parameters including:
        - num_groups: Number of groups of people (default: 5)
        - crossover_rate: Probability of crossover (default: 0.8)
        - mutation_rate: Probability of mutation (default: 0.1)
    %}
    
    properties
        num_groups      % Number of groups of people
        crossover_rate  % Probability of crossover
        mutation_rate   % Probability of mutation
    end
    
    methods
        function obj = GeneticAlgorithmOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            GeneticAlgorithmOptimizer constructor - Initialize the GA solver
            
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
                    Additional GA parameters:
                    - num_groups: Number of groups of people (default: 5)
                    - crossover_rate: Probability of crossover (default: 0.8)
                    - mutation_rate: Probability of mutation (default: 0.1)
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set solver name
            obj.name_solver = "Genetic Algorithm Optimizer";
            
            % Algorithm-specific parameters with defaults
            % kwargs (Name-Value pairs)
            if ~isempty(varargin)
                obj.kwargs = obj.nv2struct(varargin{:});
            end
            obj.num_groups = obj.get_kw('num_groups', 5);         % Number of groups
            obj.crossover_rate = obj.get_kw('crossover_rate', 0.8); % Crossover probability
            obj.mutation_rate = obj.get_kw('mutation_rate', 0.1);  % Mutation probability
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main genetic algorithm optimization method
            
            Parameters:
                search_agents_no : int
                    Number of search agents (chromosomes) per population
                max_iter : int
                    Maximum number of iterations
                    
            Returns:
                history_step_solver : cell array
                    Optimization history
                best_solver : Member
                    Best solution found overall
            %}
            
            % Initialize storage variables
            history_step_solver = {};
            
            % Initialize best solution
            if obj.maximize
                initial_fitness = -inf;
            else
                initial_fitness = inf;
            end
            best_solver = Member(zeros(1, obj.dim), initial_fitness);
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Run multiple populations (as in MATLAB implementation)
            for pop_idx = 1:obj.num_groups
                % Initialize population for this run
                population = obj.init_population(search_agents_no);
                
                % Initialize best solution for this population
                sorted_population = obj.sort_population(population);
                population_best = sorted_population{1}.copy();
                
                % Main optimization loop for this population
                for iter = 1:max_iter
                    % Evaluate fitness for all chromosomes
                    for i = 1:search_agents_no
                        population{i}.fitness = obj.objective_func(population{i}.position);
                    end
                    
                    % Natural selection - sort population by fitness
                    [sorted_population, sorted_indices] = obj.sort_population(population);
                    current_best = sorted_population{1};
                    
                    % Update population best
                    if obj.is_better(current_best, population_best)
                        population_best = current_best.copy();
                    end
                    
                    % Perform crossover
                    population = obj.uniform_crossover(population, sorted_indices);
                    
                    % Perform mutation
                    population = obj.mutation(population, sorted_indices);
                    
                    % Update global best
                    if obj.is_better(population_best, best_solver)
                        best_solver = population_best.copy();
                    end
                    
                    % Store the best solution at this iteration
                    history_step_solver{end+1} = best_solver.copy();
                    
                    % Update progress
                    obj.callbacks(iter + (pop_idx - 1) * max_iter, obj.num_groups * max_iter, best_solver);
                end
            end
            
            % Final evaluation and storage
            obj.history_step_solver = history_step_solver;
            obj.best_solver = best_solver;
            
            % End solver
            obj.end_step_solver();
        end
        
        function new_population = uniform_crossover(obj, population, sorted_indices)
            %{
            uniform_crossover - Perform uniform crossover operation
            
            Inputs:
                population : cell array
                    Current population
                sorted_indices : array
                    Indices of population sorted by fitness
                    
            Returns:
                new_population : cell array
                    Population after crossover
            %}
            
            new_population = cell(1, length(population));
            for i = 1:length(population)
                new_population{i} = population{i}.copy();
            end
            
            for i = 1:length(population)
                % Skip the best two chromosomes (elitism)
                if i == sorted_indices(1) || i == sorted_indices(2)
                    continue;
                end
                
                % Perform crossover with probability crossover_rate
                if rand() < obj.crossover_rate
                    % Choose random parent from the best two
                    parent_idx = sorted_indices(randi([1, 2]));
                    
                    % Perform uniform crossover for each dimension
                    for d = 1:obj.dim
                        if rand() < 0.5 % 50% chance to inherit from parent
                            new_population{i}.position(d) = population{parent_idx}.position(d);
                        end
                    end
                end
            end
        end
        
        function new_population = mutation(obj, population, sorted_indices)
            %{
            mutation - Perform mutation operation
            
            Inputs:
                population : cell array
                    Current population
                sorted_indices : array
                    Indices of population sorted by fitness
                    
            Returns:
                new_population : cell array
                    Population after mutation
            %}
            
            new_population = cell(1, length(population));
            for i = 1:length(population)
                new_population{i} = population{i}.copy();
            end
            
            % Mutate the worst chromosome with probability mutation_rate
            worst_idx = sorted_indices(end);
            if rand() < obj.mutation_rate
                % Replace with random solution
                new_population{worst_idx}.position = rand(1, obj.dim) .* (obj.ub - obj.lb) + obj.lb;
                new_population{worst_idx}.fitness = obj.objective_func(new_population{worst_idx}.position);
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
