classdef MultiObjectiveFireflyOptimizer < MultiObjectiveSolver
    %{
    Multi-Objective Firefly Algorithm Optimizer
    
    This algorithm extends the standard Firefly Algorithm for multi-objective optimization
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
        - alpha: Randomness parameter (default: 0.5)
        - betamin: Minimum attractiveness (default: 0.2)
        - gamma: Absorption coefficient (default: 1.0)
        - alpha_reduction: Whether to reduce alpha over iterations (default: true)
        - alpha_delta: Alpha reduction factor (default: 0.97)
        - grid_alpha: Grid inflation parameter (default: 0.1)
        - n_grid: Number of grids per dimension (default: 7)
        - beta: Leader selection pressure (default: 2)
        - gamma_removal: Archive removal pressure (default: 2)
    %}
    
    properties
        alpha
        betamin
        gamma
        alpha_reduction
        alpha_delta
        alpha_initial
    end
    
    methods
        function obj = MultiObjectiveFireflyOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveFireflyOptimizer constructor - Initialize the MOFA solver
            
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
            obj.name_solver = "Multi-Objective Firefly Optimizer";
            
            % Firefly algorithm specific parameters
            obj.alpha = obj.get_kw('alpha', 0.5);  % Randomness parameter
            obj.betamin = obj.get_kw('betamin', 0.2);  % Minimum attractiveness
            obj.gamma = obj.get_kw('gamma', 1.0);  % Absorption coefficient
            obj.alpha_reduction = obj.get_kw('alpha_reduction', true);  % Reduce alpha over time
            obj.alpha_delta = obj.get_kw('alpha_delta', 0.97);  % Alpha reduction factor
            
            % Store initial alpha for reference
            obj.alpha_initial = obj.alpha;
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective Firefly Algorithm
            
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
            
            % Calculate scale for random movement
            scale = abs(obj.ub - obj.lb);
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Evaluate all fireflies
                for i = 1:search_agents_no
                    population(i).multi_fitness = obj.objective_func(population(i).position);
                end
                
                % Move all fireflies towards brighter ones in the archive
                for i = 1:search_agents_no
                    % Select a leader from archive using grid-based selection
                    leader = obj.select_leader();
                    
                    if ~isempty(leader)
                        % Calculate distance between firefly and leader
                        r = sqrt(sum((population(i).position - leader.position).^2));
                        
                        % Calculate attractiveness
                        beta = obj.calculate_attractiveness(r);
                        
                        % Generate random movement
                        random_move = obj.alpha * (rand(1, obj.dim) - 0.5) .* scale;
                        
                        % Update position
                        new_position = (population(i).position * (1 - beta) + ...
                                       leader.position * beta + ...
                                       random_move);
                        
                        % Apply bounds
                        new_position = max(min(new_position, obj.ub), obj.lb);
                        
                        % Update firefly position and fitness
                        population(i).position = new_position;
                        population(i).multi_fitness = obj.objective_func(new_position);
                    end
                end
                
                % Update archive with current population
                obj = obj.add_to_archive(population);
                
                % Store archive state for history
                archive_copy = cell(1, length(obj.archive));
                for idx = 1:length(obj.archive)
                    archive_copy{idx} = obj.archive(idx).copy();
                end
                history_archive{end+1} = archive_copy;
                
                % Reduce alpha (randomness) over iterations if enabled
                if obj.alpha_reduction
                    obj.alpha = obj.reduce_alpha(obj.alpha, obj.alpha_delta);
                end
                
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
        
        function beta = calculate_attractiveness(obj, distance)
            %{
            calculate_attractiveness - Calculate attractiveness based on distance between fireflies
            
            Parameters:
            -----------
            distance : float
                Euclidean distance between two fireflies
                
            Returns:
            --------
            beta : float
                Attractiveness value
            %}
            
            beta0 = 1.0;  % Attractiveness at distance 0
            beta = (beta0 - obj.betamin) * exp(-obj.gamma * distance^2) + obj.betamin;
        end
        
        function new_alpha = reduce_alpha(~, current_alpha, delta)
            %{
            reduce_alpha - Reduce the randomness parameter alpha over iterations
            
            Parameters:
            -----------
            current_alpha : float
                Current alpha value
            delta : float
                Reduction factor
                
            Returns:
            --------
            new_alpha : float
                Reduced alpha value
            %}
            
            new_alpha = current_alpha * delta;
        end
    end
end
