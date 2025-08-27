classdef MultiObjectiveElectromagneticChargedParticlesOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveElectromagneticChargedParticlesOptimizer - Multi-Objective Electromagnetic Charged Particles Optimization (ECPO) Algorithm.
    
    This algorithm extends the standard ECPO for multi-objective optimization
    using archive management and grid-based selection for maintaining diversity.
    
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
        Optimization direction (true: maximize, false: minimize)
    varargin : cell array
        Additional parameters:
        - strategy: Movement strategy (1, 2, or 3, default: 1)
        - npi: Number of particles for interaction (default: 2)
    %}
    
    properties
        strategy
        npi
    end
    
    methods
        function obj = MultiObjectiveElectromagneticChargedParticlesOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveElectromagneticChargedParticlesOptimizer constructor - Initialize the MOECPO solver
            
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
            obj.name_solver = "Multi-Objective Electromagnetic Charged Particles Optimizer";
            
            % Set algorithm parameters with defaults
            obj.strategy = obj.get_kw('strategy', 1);  % Movement strategy (1, 2, or 3)
            obj.npi = obj.get_kw('npi', 2);  % Number of particles for interaction
            
            % Validate parameters
            if ~ismember(obj.strategy, [1, 2, 3])
                error("Strategy must be 1, 2, or 3");
            end
            if obj.npi < 2
                error("NPI must be at least 2");
            end
        end
        
        function n_choose_2 = n_choose_2(obj, n)
            %{
            n_choose_2 - Calculate n choose 2 (number of combinations).
            
            Inputs:
                n : int
                    Number of items
                    
            Returns:
                n_choose_2 : int
                    Number of combinations
            %}
            
            if n < 2
                n_choose_2 = 0;
            else
                n_choose_2 = n * (n - 1) / 2;
            end
        end
        
        function best_member = get_best_from_population(obj, population)
            %{
            get_best_from_population - Get the best particle from population based on random fitness.
            
            Inputs:
                population : object array
                    Population to select from
                    
            Returns:
                best_member : MultiObjectiveMember
                    Best particle
            %}
            
            if isempty(population)
                best_member = [];
                return;
            end
            
            % Use sum of fitness for selection (simple approach)
            fitness_values = zeros(1, length(population));
            for i = 1:length(population)
                fitness_values(i) = sum(population(i).multi_fitness);
            end
            
            if obj.maximize
                [~, best_idx] = max(fitness_values);
            else
                [~, best_idx] = min(fitness_values);
            end
            
            best_member = population(best_idx);
        end
        
        function new_particles = strategy_1(obj, selected_particles, leader, force)
            %{
            strategy_1 - Strategy 1: Pairwise interactions between particles.
            
            Inputs:
                selected_particles : object array
                    Selected particles for interaction
                leader : MultiObjectiveMember
                    Leader particle
                force : float
                    Force parameter
                    
            Returns:
                new_particles : object array
                    Newly generated particles
            %}
            
            new_particles = repmat(MultiObjectiveMember(0, 0), 1, 0);
            
            for i = 1:obj.npi
                for j = 1:obj.npi
                    if i == j
                        continue;
                    end
                    
                    % Base movement towards leader
                    new_position = selected_particles(i).position;
                    new_position = new_position + force * (leader.position - selected_particles(i).position);
                    
                    % Add interaction with other particle
                    if j < i
                        new_position = new_position + force * (selected_particles(j).position - selected_particles(i).position);
                    else  % j > i
                        new_position = new_position - force * (selected_particles(j).position - selected_particles(i).position);
                    end
                    
                    new_particles = [new_particles, MultiObjectiveMember(new_position, zeros(1, obj.n_objectives))];
                end
            end
        end
        
        function new_particles = strategy_2(obj, selected_particles, leader, force)
            %{
            strategy_2 - Strategy 2: Combined interactions from all particles.
            
            Inputs:
                selected_particles : object array
                    Selected particles for interaction
                leader : MultiObjectiveMember
                    Leader particle
                force : float
                    Force parameter
                    
            Returns:
                new_particles : object array
                    Newly generated particles
            %}
            
            new_particles = repmat(MultiObjectiveMember(0, 0), 1, obj.npi);
            
            for i = 1:obj.npi
                new_position = selected_particles(i).position;
                
                % Movement towards leader (no force for this component)
                new_position = new_position + 0 * force * (leader.position - selected_particles(i).position);
                
                % Combined interactions with all other particles
                for j = 1:obj.npi
                    if j < i
                        new_position = new_position + force * (selected_particles(j).position - selected_particles(i).position);
                    elseif j > i
                        new_position = new_position - force * (selected_particles(j).position - selected_particles(i).position);
                    end
                end
                
                new_particles(i) = MultiObjectiveMember(new_position, zeros(1, obj.n_objectives));
            end
        end
        
        function new_particles = strategy_3(obj, selected_particles, leader, force)
            %{
            strategy_3 - Strategy 3: Hybrid approach with two types of movements.
            
            Inputs:
                selected_particles : object array
                    Selected particles for interaction
                leader : MultiObjectiveMember
                    Leader particle
                force : float
                    Force parameter
                    
            Returns:
                new_particles : object array
                    Newly generated particles
            %}
            
            new_particles_1 = repmat(MultiObjectiveMember(0, 0), 1, obj.npi);
            new_particles_2 = repmat(MultiObjectiveMember(0, 0), 1, obj.npi);
            
            for i = 1:obj.npi
                % Type 1 movement (similar to strategy 1)
                s1_position = selected_particles(i).position;
                s1_position = s1_position + force * (leader.position - selected_particles(i).position);
                
                % Type 2 movement (full force towards leader)
                s2_position = selected_particles(i).position;
                s2_position = s2_position + 1 * force * (leader.position - selected_particles(i).position);
                
                for j = 1:obj.npi
                    if j < i
                        s1_position = s1_position + force * (selected_particles(j).position - selected_particles(i).position);
                        s2_position = s2_position + force * (selected_particles(j).position - selected_particles(i).position);
                    elseif j > i
                        s1_position = s1_position - force * (selected_particles(j).position - selected_particles(i).position);
                        s2_position = s2_position - force * (selected_particles(j).position - selected_particles(i).position);
                    end
                end
                
                new_particles_1(i) = MultiObjectiveMember(s1_position, zeros(1, obj.n_objectives));
                new_particles_2(i) = MultiObjectiveMember(s2_position, zeros(1, obj.n_objectives));
            end
            
            new_particles = [new_particles_1, new_particles_2];
        end
        
        function apply_archive_mutation(obj, new_particles)
            %{
            apply_archive_mutation - Apply archive-based mutation to new particles.
            
            Inputs:
                new_particles : object array
                    New particles to mutate
            %}
            
            if isempty(obj.archive)
                return;
            end
            
            for i = 1:length(new_particles)
                particle = new_particles(i);
                for j = 1:obj.dim
                    if rand() < 0.2  % 20% chance of mutation
                        % Replace dimension with value from random archive member
                        archive_idx = randi(length(obj.archive));
                        archive_member = obj.archive(archive_idx);
                        particle.position(j) = archive_member.position(j);
                    end
                end
            end
        end
        
        function update_population(obj, population, new_particles)
            %{
            update_population - Update population by replacing worst particles with new ones.
            
            Inputs:
                population : object array
                    Current population
                new_particles : object array
                    Newly generated particles
            %}
            
            if isempty(new_particles)
                return;
            end
            
            % Determine domination status of current population
            obj.determine_domination(population);
            
            % Get dominated particles (worst ones)
            dominated_particles = [];
            for i = 1:length(population)
                if population(i).dominated
                    dominated_particles = [dominated_particles, i];
                end
            end
            
            % Replace dominated particles with new particles
            n_to_replace = min(length(dominated_particles), length(new_particles));
            
            for i = 1:n_to_replace
                idx = dominated_particles(i);
                population(idx) = new_particles(i);
            end
        end
        
        function new_particles = generate_new_particles(obj, population, search_agents_no)
            %{
            generate_new_particles - Generate new particles using electromagnetic force-based movement.
            
            Inputs:
                population : object array
                    Current population
                search_agents_no : int
                    Population size
                    
            Returns:
                new_particles : object array
                    List of newly generated particles
            %}
            
            new_particles = repmat(MultiObjectiveMember(0, 0), 1, 0);
            
            % Calculate population factor based on strategy
            if obj.strategy == 1
                pop_factor = 2 * obj.n_choose_2(obj.npi);
            elseif obj.strategy == 2
                pop_factor = obj.npi;
            else  % strategy == 3
                pop_factor = 2 * obj.n_choose_2(obj.npi) + obj.npi;
            end
            
            % Number of iterations to generate enough particles
            n_iterations = ceil(search_agents_no / pop_factor);
            
            for iter = 1:n_iterations
                % Generate Gaussian force with mean=0.7, std=0.2
                force = 0.7 + 0.2 * randn();
                
                % Select random particles for interaction
                selected_indices = randperm(length(population), min(obj.npi, length(population)));
                selected_particles = population(selected_indices);
                
                % Select leader from archive using grid-based selection
                leader = obj.select_leader();
                if isempty(leader)
                    % If no leader available, use best from selected particles
                    leader = obj.get_best_from_population(selected_particles);
                end
                
                if obj.strategy == 1
                    % Strategy 1: Pairwise interactions
                    strategy_particles = obj.strategy_1(selected_particles, leader, force);
                elseif obj.strategy == 2
                    % Strategy 2: Combined interactions
                    strategy_particles = obj.strategy_2(selected_particles, leader, force);
                else  % strategy == 3
                    % Strategy 3: Hybrid approach
                    strategy_particles = obj.strategy_3(selected_particles, leader, force);
                end
                
                new_particles = [new_particles, strategy_particles];
            end
            
            % Ensure positions stay within bounds
            for i = 1:length(new_particles)
                new_particles(i).position = max(min(new_particles(i).position, obj.ub), obj.lb);
            end
            
            % Apply archive-based mutation
            obj.apply_archive_mutation(new_particles);
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective ECPO.
            
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
                % Generate new particles based on strategy
                new_particles = obj.generate_new_particles(population, search_agents_no);
                
                % Evaluate new particles
                for i = 1:length(new_particles)
                    new_particles(i).multi_fitness = obj.objective_func(new_particles(i).position);
                end
                
                % Update archive with new particles
                obj = obj.add_to_archive(new_particles);
                
                % Update population with new particles (replace worst ones)
                obj.update_population(population, new_particles);
                
                % Store archive state for history
                archive_copy = cell(1, length(obj.archive));
                for idx = 1:length(obj.archive)
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
