classdef ElectromagneticChargedParticlesOptimizer < Solver
    %{
    Electromagnetic Charged Particles Optimization (ECPO) Algorithm.
    
    ECPO is a physics-inspired metaheuristic optimization algorithm that mimics
    the behavior of charged particles in an electromagnetic field. The algorithm
    uses three different strategies for particle movement based on electromagnetic
    forces between particles.
    
    The algorithm features:
    - Three different movement strategies (V[0] parameter)
    - Archive-based selection for maintaining diversity
    - Force-based movement inspired by electromagnetic interactions
    
    Parameters:
    -----------
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
        Additional algorithm parameters:
        - strategy: Movement strategy (1, 2, or 3, default: 1)
        - npi: Number of particles for interaction (default: 2)
        - archive_ratio: Archive size ratio (default: 1.0)
    
    References:
        Original MATLAB implementation by Houssem
    %}
    
    properties
        strategy       % Movement strategy (1, 2, or 3)
        npi            % Number of particles for interaction
        archive_ratio  % Archive size ratio
    end
    
    methods
        function obj = ElectromagneticChargedParticlesOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            ElectromagneticChargedParticlesOptimizer constructor - Initialize the ECPO solver
            
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
                    Additional ECPO parameters:
                    - strategy: Movement strategy (1, 2, or 3, default: 1)
                    - npi: Number of particles for interaction (default: 2)
                    - archive_ratio: Archive size ratio (default: 1.0)
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set solver name
            obj.name_solver = "Electromagnetic Charged Particles Optimizer";
            
            % Algorithm-specific parameters with defaults
            % kwargs (Name-Value pairs)
            if ~isempty(varargin)
                obj.kwargs = obj.nv2struct(varargin{:});
            end
            obj.strategy = obj.get_kw('strategy', 1);      % Movement strategy (1, 2, or 3)
            obj.npi = obj.get_kw('npi', 2);                % Number of particles for interaction
            obj.archive_ratio = obj.get_kw('archive_ratio', 1.0);  % Archive size ratio
            
            % Validate parameters
            if ~ismember(obj.strategy, [1, 2, 3])
                error("Strategy must be 1, 2, or 3");
            end
            if obj.npi < 2
                error("NPI must be at least 2");
            end
            if obj.archive_ratio <= 0
                error("Archive ratio must be positive");
            end
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for ECPO algorithm
            
            The algorithm uses electromagnetic force-based movement with three different
            strategies for particle interaction and archive-based selection for diversity.
            
            Inputs:
                search_agents_no : int
                    Number of charged particles in the population
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
            
            % Calculate archive size
            archive_size = floor(search_agents_no / obj.archive_ratio);
            
            % Initialize population
            population = obj.init_population(search_agents_no);
            
            % Initialize best solution
            sorted_population = obj.sort_population(population);
            best_solver = sorted_population{1}.copy();
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Sort population and create archive
                sorted_population = obj.sort_population(population);
                archive = sorted_population(1:min(archive_size, length(sorted_population)));
                
                % Generate new particles based on strategy
                new_particles = obj.generate_new_particles(population, archive, search_agents_no);
                
                % Evaluate new particles
                for i = 1:length(new_particles)
                    new_particles{i}.fitness = obj.objective_func(new_particles{i}.position);
                end
                
                % Combine archive and new particles
                combined_population = [archive, new_particles];
                
                % Sort combined population and select best
                sorted_combined = obj.sort_population(combined_population);
                population = sorted_combined(1:min(search_agents_no, length(sorted_combined)));
                
                % Update best solution
                current_best = population{1};
                if obj.is_better(current_best, best_solver)
                    best_solver = current_best.copy();
                end
                
                % Store history
                history_step_solver{end+1} = best_solver.copy();
                
                % Update progress
                obj.callbacks(iter, max_iter, best_solver);
            end
            
            % Final processing
            obj.history_step_solver = history_step_solver;
            obj.best_solver = best_solver;
            
            % End solver
            obj.end_step_solver();
        end
        
        function new_particles = generate_new_particles(obj, population, archive, search_agents_no)
            %{
            generate_new_particles - Generate new particles using electromagnetic force-based movement
            
            Inputs:
                population : cell array
                    Current population
                archive : cell array
                    Archive of best solutions
                search_agents_no : int
                    Population size
                    
            Returns:
                new_particles : cell array
                    Cell array of newly generated particles
            %}
            
            new_particles = {};
            
            % Calculate population factor based on strategy
            if obj.strategy == 1
                pop_factor = 2 * obj.n_choose_2(obj.npi);
            elseif obj.strategy == 2
                pop_factor = obj.npi;
            else % strategy == 3
                pop_factor = 2 * obj.n_choose_2(obj.npi) + obj.npi;
            end
            
            % Number of iterations to generate enough particles
            n_iterations = ceil(search_agents_no / pop_factor);
            
            for iter_idx = 1:n_iterations
                % Generate Gaussian force with mean=0.7, std=0.2
                force = 0.7 + 0.2 * randn();
                
                % Select random particles for interaction
                selected_indices = randperm(length(population), min(obj.npi, length(population)));
                selected_particles = population(selected_indices);
                
                if obj.strategy == 1
                    % Strategy 1: Pairwise interactions
                    new_particles = [new_particles, obj.strategy_1(selected_particles, archive, force)];
                elseif obj.strategy == 2
                    % Strategy 2: Combined interactions
                    new_particles = [new_particles, obj.strategy_2(selected_particles, archive, force)];
                else % strategy == 3
                    % Strategy 3: Hybrid approach
                    new_particles = [new_particles, obj.strategy_3(selected_particles, archive, force)];
                end
            end
            
            % Ensure positions stay within bounds
            for i = 1:length(new_particles)
                new_particles{i}.position = max(min(new_particles{i}.position, obj.ub), obj.lb);
            end
            
            % Apply archive-based mutation
            obj.apply_archive_mutation(new_particles, archive);
        end
        
        function new_particles = strategy_1(obj, selected_particles, archive, force)
            % Strategy 1: Pairwise interactions between particles.
            new_particles = {};
            if ~isempty(archive)
                best_particle = archive{1};
            else
                best_particle = selected_particles{1};
            end
            
            for i = 1:length(selected_particles)
                for j = 1:length(selected_particles)
                    if i == j
                        continue;
                    end
                    
                    % Base movement towards best particle
                    new_position = selected_particles{i}.position;
                    new_position = new_position + force * (best_particle.position - selected_particles{i}.position);
                    
                    % Add interaction with other particle
                    if j < i
                        new_position = new_position + force * (selected_particles{j}.position - selected_particles{i}.position);
                    else % j > i
                        new_position = new_position - force * (selected_particles{j}.position - selected_particles{i}.position);
                    end
                    
                    new_particles{end+1} = Member(new_position, 0.0);
                end
            end
        end
        
        function new_particles = strategy_2(obj, selected_particles, archive, force)
            % Strategy 2: Combined interactions from all particles.
            new_particles = {};
            if ~isempty(archive)
                best_particle = archive{1};
            else
                best_particle = selected_particles{1};
            end
            
            for i = 1:length(selected_particles)
                new_position = selected_particles{i}.position;
                
                % Movement towards best particle (no force for this component)
                new_position = new_position + 0 * force * (best_particle.position - selected_particles{i}.position);
                
                % Combined interactions with all other particles
                for j = 1:length(selected_particles)
                    if j < i
                        new_position = new_position + force * (selected_particles{j}.position - selected_particles{i}.position);
                    elseif j > i
                        new_position = new_position - force * (selected_particles{j}.position - selected_particles{i}.position);
                    end
                end
                
                new_particles{end+1} = Member(new_position, 0.0);
            end
        end
        
        function new_particles = strategy_3(obj, selected_particles, archive, force)
            % Strategy 3: Hybrid approach with two types of movements.
            new_particles_1 = {};
            new_particles_2 = {};
            if ~isempty(archive)
                best_particle = archive{1};
            else
                best_particle = selected_particles{1};
            end
            
            for i = 1:length(selected_particles)
                % Type 1 movement (similar to strategy 1)
                s1_position = selected_particles{i}.position;
                s1_position = s1_position + force * (best_particle.position - selected_particles{i}.position);
                
                % Type 2 movement (full force towards best)
                s2_position = selected_particles{i}.position;
                s2_position = s2_position + 1 * force * (best_particle.position - selected_particles{i}.position);
                
                for j = 1:length(selected_particles)
                    if j < i
                        s1_position = s1_position + force * (selected_particles{j}.position - selected_particles{i}.position);
                        s2_position = s2_position + force * (selected_particles{j}.position - selected_particles{i}.position);
                    elseif j > i
                        s1_position = s1_position - force * (selected_particles{j}.position - selected_particles{i}.position);
                        s2_position = s2_position - force * (selected_particles{j}.position - selected_particles{i}.position);
                    end
                end
                
                new_particles_1{end+1} = Member(s1_position, 0.0);
                new_particles_2{end+1} = Member(s2_position, 0.0);
            end
            
            new_particles = [new_particles_1, new_particles_2];
        end
        
        function apply_archive_mutation(obj, new_particles, archive)
            % Apply archive-based mutation to new particles.
            if isempty(archive)
                return;
            end
            
            for i = 1:length(new_particles)
                for j = 1:obj.dim
                    if rand() < 0.2 % 20% chance of mutation
                        % Replace dimension with value from random archive member
                        archive_idx = randi(length(archive));
                        new_particles{i}.position(j) = archive{archive_idx}.position(j);
                    end
                end
            end
        end
        
        function result = n_choose_2(~, n)
            % Calculate n choose 2 (number of combinations).
            if n < 2
                result = 0;
            else
                result = n * (n - 1) / 2;
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
