classdef BacteriaForagingOptimizer < Solver
    %{
    Bacteria Foraging Optimization (BFO) Algorithm.
    
    BFO is a nature-inspired metaheuristic optimization algorithm that mimics
    the foraging behavior of E. coli bacteria. The algorithm simulates three
    main processes in bacterial foraging:
    1. Chemotaxis: Movement towards nutrients (better solutions)
    2. Reproduction: Reproduction of successful bacteria
    3. Elimination-dispersal: Random elimination and dispersal to avoid local optima
    
    The algorithm maintains a population of bacteria that move through the search
    space using a combination of random walks and directed movement based on
    previous success.
    
    Parameters:
    -----------
    n_elimination : int
        Number of elimination-dispersal events (Ne)
    n_reproduction : int
        Number of reproduction steps (Nr)
    n_chemotaxis : int
        Number of chemotaxis steps (Nc)
    n_swim : int
        Number of swim steps (Ns)
    step_size : float
        Step size for movement (C)
    elimination_prob : float
        Probability of elimination-dispersal (Ped)
    
    References:
        Passino, K. M. (2002). Biomimicry of bacterial foraging for distributed 
        optimization and control. IEEE Control Systems Magazine, 22(3), 52-67.
    %}
    
    properties
        n_reproduction
        n_chemotaxis
        n_swim
        step_size
        elimination_prob
    end
    
    methods
        function obj = BacteriaForagingOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            BacteriaForagingOptimizer constructor - Initialize the BFO solver
            
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
                    Additional solver parameters
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set solver name
            obj.name_solver = "Bacteria Foraging Optimizer";
            
            % Parse additional parameters
            if ~isempty(varargin)
                obj.kwargs = obj.nv2struct(varargin{:});
            end
            
            % Algorithm-specific parameters with defaults
            obj.n_reproduction = obj.get_kw('n_reproduction', 4);
            obj.n_chemotaxis = obj.get_kw('n_chemotaxis', 10);
            obj.n_swim = obj.get_kw('n_swim', 4);
            obj.step_size = obj.get_kw('step_size', 0.1);
            obj.elimination_prob = obj.get_kw('elimination_prob', 0.25);
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for BFO algorithm
            
            The algorithm simulates bacterial foraging through three main processes:
            1. Chemotaxis: Bacteria move towards better solutions
            2. Reproduction: Successful bacteria reproduce
            3. Elimination-dispersal: Random elimination and dispersal
            
            Inputs:
                search_agents_no : int
                    Number of bacteria in the population
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
            [sorted_population, ~] = obj.sort_population(population);
            best_solver = sorted_population{1}.copy();
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop (elimination-dispersal events)
            for iter = 1:max_iter
                
                % Reproduction loop
                for reproduction_iter = 1:obj.n_reproduction
                    
                    % Chemotaxis loop
                    for chemotaxis_iter = 1:obj.n_chemotaxis
                        
                        % Update each bacterium
                        for i = 1:length(population)
                            % Generate random direction vector
                            direction = rand(1, obj.dim) * 2 - 1;
                            direction_norm = norm(direction);
                            
                            if direction_norm > 0
                                direction = direction / direction_norm;
                            end
                            
                            % Move bacterium
                            new_position = population{i}.position + obj.step_size * direction;
                            new_position = max(min(new_position, obj.ub), obj.lb);
                            
                            % Evaluate new position
                            new_fitness = obj.objective_func(new_position);
                            
                            % Swim behavior - continue moving in same direction if improvement
                            swim_count = 0;
                            while swim_count < obj.n_swim
                                if obj.is_better_fitness(new_fitness, population{i}.fitness)
                                    % Accept move and continue swimming
                                    population{i}.position = new_position;
                                    population{i}.fitness = new_fitness;
                                    
                                    % Move further in same direction
                                    new_position = population{i}.position + obj.step_size * direction;
                                    new_position = max(min(new_position, obj.ub), obj.lb);
                                    new_fitness = obj.objective_func(new_position);
                                    swim_count = swim_count + 1;
                                else
                                    % Stop swimming
                                    break;
                                end
                            end
                        end
                        
                        % Update best solution
                        [sorted_population, ~] = obj.sort_population(population);
                        current_best = sorted_population{1};
                        if obj.is_better(current_best, best_solver)
                            best_solver = current_best.copy();
                        end
                    end
                end
                
                % Reproduction: Keep best half and duplicate
                [sorted_population, ~] = obj.sort_population(population);
                best_half = sorted_population(1:search_agents_no / 2);
                
                % Create new population by duplicating best half
                new_population = {};
                for j = 1:length(best_half)
                    new_population{end+1} = best_half{j}.copy();
                end
                for j = 1:length(best_half)
                    new_population{end+1} = best_half{j}.copy();
                end
                
                population = new_population;

                % Store history
                history_step_solver{end+1} = best_solver.copy();
                % Callbacks
                obj.callbacks(iter, max_iter, best_solver);
            end
            
            % Elimination-dispersal: Random elimination of some bacteria
            for i = 1:length(population)
                if rand() < obj.elimination_prob
                    % Randomly disperse this bacterium
                    new_position = rand(1, obj.dim) .* (obj.ub - obj.lb) + obj.lb;
                    new_fitness = obj.objective_func(new_position);
                    population{i} = Member(new_position, new_fitness);
                end
            end
            
            % Final update of best solution
            [sorted_population, ~] = obj.sort_population(population);
            current_best = sorted_population{1};
            if obj.is_better(current_best, best_solver)
                best_solver = current_best.copy();
            end
            
            % Store final history
            history_step_solver{end+1} = best_solver.copy();
            
            % Finalize solver
            obj.history_step_solver = history_step_solver;
            obj.best_solver = best_solver;
            obj.end_step_solver();
        end
        
        function [sorted_population, sorted_indices] = sort_population(obj, population)
            %{
            sort_population - Sort population based on fitness
            
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
        
        function result = is_better_fitness(obj, fitness1, fitness2)
            %{
            is_better_fitness - Compare two fitness values based on optimization direction
            
            Inputs:
                fitness1 : float
                    First fitness value
                fitness2 : float
                    Second fitness value
                    
            Returns:
                result : bool
                    True if fitness1 is better than fitness2
            %}
            
            if obj.maximize
                result = fitness1 > fitness2;
            else
                result = fitness1 < fitness2;
            end
        end
    end
end
