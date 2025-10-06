classdef MultiObjectiveBacteriaForagingOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveBacteriaForagingOptimizer - Multi-Objective Bacteria Foraging Optimization (BFO) Algorithm.
    
    This algorithm extends the standard BFO for multi-objective optimization
    using archive management and grid-based selection. The algorithm simulates
    three main processes in bacterial foraging adapted for multi-objective problems:
    1. Chemotaxis: Movement towards better solutions using Pareto dominance
    2. Reproduction: Reproduction of successful bacteria based on health
    3. Elimination-dispersal: Random elimination and dispersal to maintain diversity
    
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
        Optimization direction (true for maximize, false for minimize)
    varargin : cell array
        Additional algorithm parameters:
        - n_reproduction: Number of reproduction steps (default: 4)
        - n_chemotaxis: Number of chemotaxis steps (default: 10)
        - n_swim: Number of swim steps (default: 4)
        - step_size: Step size for movement (default: 0.1)
        - elimination_prob: Probability of elimination-dispersal (default: 0.25)
    
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
        function obj = MultiObjectiveBacteriaForagingOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveBacteriaForagingOptimizer constructor - Initialize the MOBFO solver
            
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
            obj.name_solver = "Multi-Objective Bacteria Foraging Optimizer";
            
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
        
        function population = init_population(obj, N)
            %{
            init_population - Initialize multi-objective bacteria population
            
            Inputs:
                N : int
                    Number of bacteria to initialize
                    
            Returns:
                population : object array
                    Array of BacteriaMultiMember objects
            %}
            pop_example = BacteriaMultiMember(0, 0, 0);
            population = repmat(pop_example, 1, N);
            for i = 1:N
                pos = obj.lb + (obj.ub - obj.lb).*rand(1, obj.dim);
                fit = obj.objective_func(pos); fit = fit(:).';
                population(i) = BacteriaMultiMember(pos, fit, 0.0);
            end
        end
        
        function normalized_costs = get_normalized_costs(obj, population)
            %{
            get_normalized_costs - Get normalized cost matrix from population for aggregation methods
            
            Inputs:
                population : object array
                    Population to normalize
                    
            Returns:
                normalized_costs : array
                    Normalized cost matrix
            %}
            
            costs = obj.get_fitness(population);
            normalized_costs = zeros(size(costs));
            
            for obj_idx = 1:obj.n_objectives
                obj_values = costs(:, obj_idx);
                min_val = min(obj_values);
                max_val = max(obj_values);
                
                if max_val ~= min_val
                    normalized_costs(:, obj_idx) = (obj_values - min_val) / (max_val - min_val);
                else
                    normalized_costs(:, obj_idx) = ones(size(obj_values));
                end
            end
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective BFO
            
            Inputs:
                search_agents_no : int
                    Number of bacteria in the population
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
            
            % Build grid for archive management
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
            
            % Main optimization loop (elimination-dispersal events)
            for iter = 1:max_iter
                
                % Reproduction loop
                for reproduction_iter = 1:obj.n_reproduction
                    
                    % Reset health values for new reproduction cycle
                    for i = 1:length(population)
                        population(i).health = 0.0;
                    end
                    
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
                            new_position = population(i).position + obj.step_size * direction;
                            new_position = max(min(new_position, obj.ub), obj.lb);
                            
                            % Evaluate new position
                            new_fitness = obj.objective_func(new_position);
                            new_bacterium = BacteriaMultiMember(new_position, new_fitness);
                            
                            % Swim behavior - continue moving in same direction if improvement
                            swim_count = 0;
                            while swim_count < obj.n_swim
                                % Use Pareto dominance for multi-objective comparison
                                if obj.dominates(new_bacterium, population(i))
                                    % Accept move and continue swimming
                                    population(i).position = new_position;
                                    population(i).multi_fitness = new_fitness;
                                    
                                    % Move further in same direction
                                    new_position = population(i).position + obj.step_size * direction;
                                    new_position = max(min(new_position, obj.ub), obj.lb);
                                    new_fitness = obj.objective_func(new_position);
                                    new_bacterium = BacteriaMultiMember(new_position, new_fitness);
                                    swim_count = swim_count + 1;
                                else
                                    % Stop swimming
                                    break;
                                end
                            end
                        end
                        
                        % Update health (sum of fitness values for reproduction)
                        for i = 1:length(population)
                            % Use simple aggregation for health calculation
                            % Sum of normalized objectives (assuming minimization)
                            normalized_fitness = obj.get_normalized_costs(population);
                            population(i).health = population(i).health + sum(normalized_fitness(i, :));
                        end
                    end
                    
                    % Reproduction: Keep best half based on health and duplicate
                    % Sort population by health (lower health is better for minimization)
                    [~, sorted_indices] = sort([population.health]);
                    best_half = population(sorted_indices(1:search_agents_no / 2));
                    
                    % Create new population by duplicating best half
                    new_population = repmat(BacteriaMultiMember(0, 0, 0), 1, search_agents_no);
                    for j = 1:length(best_half)
                        new_population(j) = best_half(j).copy();
                    end
                    for j = 1:length(best_half)
                        new_population(j + length(best_half)) = best_half(j).copy();
                    end
                    
                    population = new_population;
                end
                
                % Elimination-dispersal: Random elimination of some bacteria
                for i = 1:length(population)
                    if rand() < obj.elimination_prob
                        % Randomly disperse this bacterium
                        new_position = obj.lb + (obj.ub - obj.lb) .* rand(1, obj.dim);
                        new_fitness = obj.objective_func(new_position);
                        population(i) = BacteriaMultiMember(new_position, new_fitness);
                    end
                end

                % Store archive state for history
                archive_copy = cell(1, length(obj.archive));
                for idx = 1:length(obj.archive)
                    archive_copy{idx} = obj.archive(idx).copy();
                end
                history_archive{end+1} = archive_copy;
                
                if ~isempty(obj.archive)
                    best_member = obj.archive(1);
                else
                    best_member = [];
                end
                obj.callbacks(iter, max_iter, best_member);
            end
            
            % Final archive update
            obj = obj.add_to_archive(population);
            
            % Final processing
            obj.history_step_solver = history_archive;
            obj.best_solver = obj.archive;
            
            % End solver
            obj.end_step_solver();
            
            archive = obj.archive;
        end
    end
end
