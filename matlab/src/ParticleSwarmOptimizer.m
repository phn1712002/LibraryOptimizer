classdef ParticleSwarmOptimizer < Solver
    %{
    Particle Swarm Optimization (PSO) algorithm implementation.
    
    PSO is a population-based stochastic optimization technique inspired by
    social behavior of bird flocking or fish schooling. Each particle in the
    swarm represents a potential solution and moves through the search space
    by following the current optimum particles.
    
    References:
        Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
        Proceedings of ICNN'95-international conference on neural networks.
    %}
    
    properties
        w          % Inertia weight
        wdamp      % Inertia weight damping ratio
        c1         % Personal learning coefficient
        c2         % Global learning coefficient
        vel_max    % Maximum velocity
        vel_min    % Minimum velocity
    end
    
    methods
        function obj = ParticleSwarmOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            ParticleSwarmOptimizer constructor - Initialize the PSO solver
            
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
                    Additional PSO parameters:
                    - w: Inertia weight (default: 1.0)
                    - wdamp: Inertia weight damping ratio (default: 0.99)
                    - c1: Personal learning coefficient (default: 1.5)
                    - c2: Global learning coefficient (default: 2.0)
                    - vel_max: Maximum velocity (default: 10% of search space range)
                    - vel_min: Minimum velocity (default: -10% of search space range)
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set default PSO parameters
            obj.name_solver = "Particle Swarm Optimizer";
            % kwargs (Name-Value pairs)
            if ~isempty(varargin)
                obj.kwargs = obj.nv2struct(varargin{:});
            end
            obj.w = obj.get_kw('w', 1.0);           % Inertia weight
            obj.wdamp = obj.get_kw('wdamp', 0.99);  % Inertia weight damping ratio
            obj.c1 = obj.get_kw('c1', 1.5);         % Personal learning coefficient
            obj.c2 = obj.get_kw('c2', 2.0);         % Global learning coefficient
            
            % Velocity limits (10% of variable range)
            vel_range = 0.1 * (obj.ub - obj.lb);
            obj.vel_max = obj.get_kw('vel_max', vel_range);
            obj.vel_min = obj.get_kw('vel_min', -vel_range);
        end
        
        function population = init_population(obj, search_agents_no)
            %{
            init_population - Initialize the particle swarm population
            
            Each particle is initialized with:
            - Random position within bounds
            - Random velocity within velocity limits
            - Fitness evaluation of initial position
            
            Inputs:
                search_agents_no : int
                    Number of particles in the swarm
                    
            Returns:
                population : cell array
                    Cell array of initialized Particle objects
            %}
            
            population = cell(1, search_agents_no);
            for i = 1:search_agents_no
                position = obj.lb + (obj.ub - obj.lb) .* rand(1, obj.dim);
                velocity = obj.vel_min + (obj.vel_max - obj.vel_min) .* rand(1, obj.dim);
                fitness = obj.objective_func(position);
                population{i} = Particle(position, fitness, velocity);
            end
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for PSO algorithm
            
            Inputs:
                search_agents_no : int
                    Number of particles in the swarm
                max_iter : int
                    Maximum number of iterations for optimization
                    
            Returns:
                history_step_solver : cell array
                    History of best solutions at each iteration
                best_solver : Particle
                    Best solution found overall
            %}
            
            % Initialize storage variables
            history_step_solver = {};
            
            % Initialize the population of particles
            population = obj.init_population(search_agents_no);
            
            % Initialize personal best particles (copy of initial particles)
            personal_best = cell(1, search_agents_no);
            for i = 1:search_agents_no
                personal_best{i} = population{i}.copy();
            end
            
            % Initialize global best
            [sorted_personal_best, ~] = obj.sort_population(personal_best);
            global_best = sorted_personal_best{1}.copy();
            best_solver = global_best.copy();
            
            % Call the begin function
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Update all particles
                for i = 1:search_agents_no
                    % Update velocity
                    r1 = rand(1, obj.dim);
                    r2 = rand(1, obj.dim);
                    
                    cognitive_component = obj.c1 .* r1 .* (personal_best{i}.position - population{i}.position);
                    social_component = obj.c2 .* r2 .* (global_best.position - population{i}.position);
                    
                    population{i}.velocity = (obj.w .* population{i}.velocity + ...
                                            cognitive_component + ...
                                            social_component);
                    
                    % Apply velocity limits
                    population{i}.velocity = max(min(population{i}.velocity, obj.vel_max), obj.vel_min);
                    
                    % Update position
                    new_position = population{i}.position + population{i}.velocity;
                    
                    % Apply position limits and velocity mirror effect
                    outside_bounds = (new_position < obj.lb) | (new_position > obj.ub);
                    population{i}.velocity(outside_bounds) = -population{i}.velocity(outside_bounds);
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Update particle position
                    population{i}.position = new_position;
                    
                    % Evaluate new fitness
                    new_fitness = obj.objective_func(new_position);
                    population{i}.fitness = new_fitness;
                    
                    % Update personal best
                    if obj.is_better(population{i}, personal_best{i})
                        personal_best{i} = population{i}.copy();
                        
                        % Update global best if needed
                        if obj.is_better(population{i}, global_best)
                            global_best = population{i}.copy();
                            
                            % Update best solver immediately
                            if obj.is_better(global_best, best_solver)
                                best_solver = global_best.copy();
                            end
                        end
                    end
                end
                
                % Store the best solution at this iteration
                history_step_solver{end+1} = best_solver.copy();
                
                % Update inertia weight
                obj.w = obj.w * obj.wdamp;
                
                % Call the callbacks
                obj.callbacks(iter, max_iter, best_solver);
            end
            
            % Final evaluation and storage
            obj.history_step_solver = history_step_solver;
            obj.best_solver = best_solver;
            
            % Call the end function
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
        
        function s = get_kw(obj, name, default)
            if isfield(obj.kwargs, name), s = obj.kwargs.(name);
            else, s = default; end
        end
    end
end
