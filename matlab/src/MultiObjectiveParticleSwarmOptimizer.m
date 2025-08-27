classdef MultiObjectiveParticleSwarmOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveParticleSwarmOptimizer - Multi-Objective Particle Swarm Optimization Algorithm
    
    This algorithm extends the standard PSO for multi-objective optimization
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
        - archive_size: Size of the external archive (default: 100)
        - alpha: Grid inflation parameter (default: 0.1)
        - n_grid: Number of grids per dimension (default: 7)
        - beta: Leader selection pressure (default: 2)
        - gamma: Archive removal pressure (default: 2)
        - w: Inertia weight (default: 1.0)
        - wdamp: Inertia weight damping ratio (default: 0.99)
        - c1: Personal learning coefficient (default: 1.5)
        - c2: Global learning coefficient (default: 2.0)
        - vel_max: Maximum velocity (default: 10% of variable range)
        - vel_min: Minimum velocity (default: -10% of variable range)
    %}
    
    properties
        w
        wdamp
        c1
        c2
        vel_max
        vel_min
    end
    
    methods
        function obj = MultiObjectiveParticleSwarmOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveParticleSwarmOptimizer constructor - Initialize the MOPSO solver
            
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
            obj.name_solver = "Multi-Objective Particle Swarm Optimizer";
            
            % Set PSO-specific parameters with defaults
            obj.w = obj.get_kw('w', 1.0);  % Inertia weight
            obj.wdamp = obj.get_kw('wdamp', 0.99);  % Inertia weight damping ratio
            obj.c1 = obj.get_kw('c1', 1.5);  % Personal learning coefficient
            obj.c2 = obj.get_kw('c2', 2.0);  % Global learning coefficient
            
            % Velocity limits (10% of variable range)
            vel_range = 0.1 * (obj.ub - obj.lb);
            obj.vel_max = obj.get_kw('vel_max', vel_range);
            obj.vel_min = obj.get_kw('vel_min', -vel_range);
        end
        
        function population = init_population(obj, N)
            %{
            init_population - Initialize multi-objective particle population
            
            Inputs:
                N : int
                    Number of particles to initialize
                    
            Returns:
                population : cell array
                    Cell array of ParticleMultiMember objects
            %}
            pop_example = ParticleMultiMember(0, 0, 0);
            population = repmat(pop_example, 1, N);
            for i = 1:N
                pos = obj.lb + (obj.ub - obj.lb).*rand(1, obj.dim);
                fit = obj.objective_func(pos); fit = fit(:).';
                velocity = obj.vel_min + (obj.vel_max - obj.vel_min) .* rand(1, obj.dim);
                population(i) = ParticleMultiMember(pos, fit, velocity);
            end
        end
        
        function dominates = dominates_personal_best(obj, particle)
            %{
            dominates_personal_best - Check if current position dominates personal best
            
            Inputs:
                particle : ParticleMultiMember
                    Particle to check
                    
            Returns:
                dominates : bool
                    true if current position dominates personal best
            %}
            
            % Create a temporary member for personal best
            personal_best_member = ParticleMultiMember(...
                particle.personal_best_position, ...
                particle.personal_best_fitness ...
            );
            
            dominates = obj.dominates(particle, personal_best_member);
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective PSO
            
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
                % Update all particles
                for i = 1:length(population)
                    particle = population(i);
                    
                    % Select leader from archive using grid-based selection
                    leader = obj.select_leader();
                    
                    % If no leader in archive, use random particle from population
                    if isempty(leader)
                        random_idx = randi(length(population));
                        leader = population(random_idx);
                    end
                    
                    % Update velocity
                    r1 = rand(1, obj.dim);
                    r2 = rand(1, obj.dim);
                    
                    cognitive_component = obj.c1 .* r1 .* (particle.personal_best_position - particle.position);
                    social_component = obj.c2 .* r2 .* (leader.position - particle.position);
                    
                    particle.velocity = (obj.w .* particle.velocity + ...
                                       cognitive_component + ...
                                       social_component);
                    
                    % Apply velocity limits
                    particle.velocity = max(min(particle.velocity, obj.vel_max), obj.vel_min);
                    
                    % Update position
                    new_position = particle.position + particle.velocity;
                    
                    % Apply position limits and velocity mirror effect
                    outside_bounds = (new_position < obj.lb) | (new_position > obj.ub);
                    particle.velocity(outside_bounds) = -particle.velocity(outside_bounds);
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Update particle position
                    particle.position = new_position;
                    
                    % Evaluate new fitness
                    new_fitness = obj.objective_func(new_position);
                    particle.multi_fitness = new_fitness;
                    
                    % Update personal best if current position dominates personal best
                    if obj.dominates_personal_best(particle)
                        particle.personal_best_position = particle.position;
                        particle.personal_best_fitness = particle.multi_fitness;
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
                
                % Update inertia weight
                obj.w = obj.w * obj.wdamp;
                
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
