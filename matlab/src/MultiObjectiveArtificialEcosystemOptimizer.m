classdef MultiObjectiveArtificialEcosystemOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveArtificialEcosystemOptimizer - Multi-Objective Artificial Ecosystem-based Optimization (AEO) algorithm.
    
    This algorithm extends the standard AEO for multi-objective optimization
    using archive management and grid-based selection.
    
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
        - production_weight: Weight for production phase (default: 1.0)
        - consumption_weight: Weight for consumption phase (default: 1.0)
        - decomposition_weight: Weight for decomposition phase (default: 1.0)
    %}
    
    properties
        production_weight
        consumption_weight
        decomposition_weight
    end
    
    methods
        function obj = MultiObjectiveArtificialEcosystemOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveArtificialEcosystemOptimizer constructor - Initialize the MO-AEO solver
            
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
            obj.name_solver = "Multi-Objective Artificial Ecosystem Optimizer";
            
            % Algorithm-specific parameters
            obj.production_weight = obj.get_kw('production_weight', 1.0);
            obj.consumption_weight = obj.get_kw('consumption_weight', 1.0);
            obj.decomposition_weight = obj.get_kw('decomposition_weight', 1.0);
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective AEO
            
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
                % Production phase: Create new organism based on archive leaders
                new_population = obj.production_phase(population, iter, max_iter);
                
                % Consumption phase: Update organisms based on consumption behavior
                new_population = obj.consumption_phase(new_population, population);
                
                % Decomposition phase: Update organisms based on decomposition behavior
                new_population = obj.decomposition_phase(new_population);
                
                % Evaluate new population and update archive
                obj = obj.add_to_archive(new_population);
                
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
        
        function new_population = production_phase(obj, population, iter, max_iter)
            %{
            production_phase - Create new organism based on archive leaders and random position.
            
            Inputs:
                population : cell array
                    Current population
                iter : int
                    Current iteration
                max_iter : int
                    Maximum number of iterations
                    
            Returns:
                new_population : cell array
                    New population after production phase
            %}
            
            % Select leader from archive using grid-based selection
            leader = obj.select_leader();
            
            % If no leader in archive, use random member from population
            if isempty(leader)
                random_idx = randi(numel(population));
                leader = population(random_idx);
            end
            
            % Create random position in search space
            random_position = obj.lb + (obj.ub - obj.lb) .* rand(1, obj.dim);
            
            % Calculate production weight (decreases linearly)
            r1 = rand();
            a = (1 - iter / max_iter) * r1;
            
            % Create first organism: combination of leader and random position
            new_position = (1 - a) * leader.position + a * random_position;
            new_position = max(min(new_position, obj.ub), obj.lb);
            new_fitness = obj.objective_func(new_position);
            new_fitness = new_fitness(:).';
            
            new_population = MultiObjectiveMember(new_position, new_fitness);
        end
        
        function new_population = consumption_phase(obj, new_population, old_population)
            %{
            consumption_phase - Update organisms based on consumption behavior.
            
            Inputs:
                new_population : cell array
                    Population from production phase
                old_population : cell array
                    Original population
                    
            Returns:
                new_population : cell array
                    Population after consumption phase
            %}
            
            % Handle second organism (special case)
            if numel(old_population) >= 2
                % Generate consumption factor C using Levy flight
                C = 0.5 * obj.levy_flight(obj.dim);
                
                % Second organism consumes from producer (first organism)
                new_position = old_population(2).position + C .* (...
                    old_population(2).position - new_population(1).position...
                );
                
                % Apply bounds
                new_position = max(min(new_position, obj.ub), obj.lb);
                new_fitness = obj.objective_func(new_position);
                new_fitness = new_fitness(:).';
                
                new_population = [new_population, MultiObjectiveMember(new_position, new_fitness)];
            end
            
            % For remaining organisms (starting from third one)
            for i = 3:numel(old_population)
                % Generate consumption factor C using Levy flight
                C = 0.5 * obj.levy_flight(obj.dim);
                
                r = rand();
                
                if r < 1/3
                    % Consume from producer (first organism)
                    new_position = old_population(i).position + C .* (...
                        old_population(i).position - new_population(1).position...
                    );
                elseif 1/3 <= r && r < 2/3
                    % Consume from random consumer (between 1 and i-1)
                    random_idx = randi([2, i-1]);
                    new_position = old_population(i).position + C .* (...
                        old_population(i).position - old_population(random_idx).position...
                    );
                else
                    % Consume from both producer and random consumer
                    r2 = rand();
                    random_idx = randi([2, i-1]);
                    new_position = old_population(i).position + C .* (...
                        r2 * (old_population(i).position - new_population(1).position) + ...
                        (1 - r2) * (old_population(i).position - old_population(random_idx).position)...
                    );
                end
                
                % Apply bounds
                new_position = max(min(new_position, obj.ub), obj.lb);
                new_fitness = obj.objective_func(new_position);
                new_fitness = new_fitness(:).';
                
                new_population = [new_population, MultiObjectiveMember(new_position, new_fitness)];
            end
        end
        
        function new_population = decomposition_phase(obj, population)
            %{
            decomposition_phase - Update organisms based on decomposition behavior.
            
            Inputs:
                population : cell array
                    Current population
                    
            Returns:
                new_population : cell array
                    Population after decomposition phase
            %}
            
            new_population = MultiObjectiveMember.empty;
            
            % Select leader from archive for decomposition guidance
            leader = obj.select_leader();
            
            % If no leader in archive, use random member from population
            if isempty(leader)
                random_idx = randi(numel(population));
                leader = population(random_idx);
            end
            
            for i = 1:numel(population)
                % Generate decomposition factors
                r3 = rand();
                weight_factor = 3 * randn();
                
                % Calculate new position using decomposition equation
                random_multiplier = randi([1, 2]);  % This gives 1 or 2
                new_position = leader.position + weight_factor * (...
                    (r3 * random_multiplier - 1) * leader.position - ...
                    (2 * r3 - 1) * population(i).position...
                );
                
                % Apply bounds
                new_position = max(min(new_position, obj.ub), obj.lb);
                new_fitness = obj.objective_func(new_position);
                new_fitness = new_fitness(:).';
                
                new_population = [new_population, MultiObjectiveMember(new_position, new_fitness)];
            end
        end
        
        function step = levy_flight(~, dim)
            %{
            levy_flight - Generate Levy flight step.
            
            Inputs:
                dim : int
                    Dimension of the step vector
                    
            Returns:
                step : array
                    Levy flight step vector
            %}
            beta = 1.5;  % Levy exponent
            sigma_u = (gamma(1 + beta) * sin(pi * beta / 2) / ...
                      (gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2))) ^ (1 / beta);
            sigma_v = 1;
            
            u = randn(1, dim) * sigma_u;
            v = randn(1, dim) * sigma_v;
            
            step = u ./ (abs(v) .^ (1 / beta));
        end
    end
end
