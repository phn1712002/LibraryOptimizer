classdef MultiObjectiveBatOptimizer < MultiObjectiveSolver
    %{
    Multi-Objective Bat Optimizer
    
    This algorithm extends the standard Bat Algorithm for multi-objective optimization
    using archive management and grid-based selection for solution evaluation.
    
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
    maximize : bool or array
        Optimization direction for each objective
    varargin : cell array
        Additional parameters:
        - archive_size: Size of the external archive (default: 100)
        - alpha_grid: Grid inflation parameter (default: 0.1)
        - n_grid: Number of grids per dimension (default: 7)
        - beta_leader: Leader selection pressure (default: 2)
        - gamma_archive: Archive removal pressure (default: 2)
        - fmin: Minimum frequency (default: 0)
        - fmax: Maximum frequency (default: 2)
        - alpha_loud: Loudness decay constant (default: 0.9)
        - gamma_pulse: Pulse rate increase constant (default: 0.9)
        - ro: Initial pulse emission rate (default: 0.5)
    %}
    
    properties
        fmin
        fmax
        alpha_loud
        gamma_pulse
        ro
    end
    
    methods
        function obj = MultiObjectiveBatOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveBatOptimizer constructor
            
            Inputs:
                objective_func : function handle
                    Multi-objective function to optimize
                lb : float or array
                    Lower bounds of search space
                ub : float or array
                    Upper bounds of search space
                dim : int
                    Number of dimensions in the problem
                maximize : bool or array
                    Optimization direction for each objective
                varargin : cell array
                    Additional solver parameters
            %}
            
            % Call parent constructor
            obj@MultiObjectiveSolver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set solver name
            obj.name_solver = "Multi-Objective Bat Optimizer";
            
            % Set default BAT parameters
            obj.fmin = obj.get_kw('fmin', 0.0);          % Minimum frequency
            obj.fmax = obj.get_kw('fmax', 2.0);          % Maximum frequency
            obj.alpha_loud = obj.get_kw('alpha_loud', 0.9);  % Loudness decay constant
            obj.gamma_pulse = obj.get_kw('gamma_pulse', 0.9);  % Pulse rate increase constant
            obj.ro = obj.get_kw('ro', 0.5);              % Initial pulse emission rate
        end
        
        function population = init_population(obj, search_agents_no)
            %{
            init_population - Initialize multi-objective bat population
            
            Inputs:
                search_agents_no : int
                    Number of bats to initialize
                    
            Returns:
                population : array
                    Array of initialized BatMultiMember objects
            %}
            
            population = BatMultiMember.empty(search_agents_no, 0);
            for i = 1:search_agents_no
                position = rand(1, obj.dim) .* (obj.ub - obj.lb) + obj.lb;
                fitness = obj.objective_func(position);
                population(i) = BatMultiMember(...
                    position, ...
                    fitness, ...
                    0.0, ...                    % frequency
                    zeros(1, obj.dim), ...      % velocity
                    1.0, ...                    % loudness
                    obj.ro ...                  % pulse_rate
                );
            end
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective Bat Algorithm
            
            Inputs:
                search_agents_no : int
                    Number of bats (search agents)
                max_iter : int
                    Maximum number of iterations
                    
            Returns:
                Tuple containing:
                    - history_archive: Cell array of archive states at each iteration
                    - archive: Final Pareto archive
            %}
            
            % Initialize storage
            history_archive = {};
            
            % Initialize population
            population = obj.init_population(search_agents_no);
            
            % Initialize archive with non-dominated solutions
            obj.determine_domination(population);
            non_dominated = obj.get_non_dominated_particles(population);
            obj.archive = non_dominated;
            
            % Initialize grid for archive
            costs = obj.get_fitness(obj.archive);
            if ~isempty(costs)
                obj.grid = obj.create_hypercubes(costs);
                for particle = obj.archive
                    [particle.grid_index, particle.grid_sub_index] = obj.get_grid_index(particle);
                end
            end
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Update each bat
                for i = 1:search_agents_no
                    % Select a leader from archive using grid-based selection
                    leader = obj.select_leader();
                    
                    % If no leader in archive, use random bat from population
                    if isempty(leader)
                        leader_idx = randi(search_agents_no);
                        leader = population(leader_idx);
                    end
                    
                    % Update frequency
                    population(i).frequency = obj.fmin + (obj.fmax - obj.fmin) * rand();
                    
                    % Update velocity towards leader
                    population(i).velocity = population(i).velocity + ...
                        (population(i).position - leader.position) * population(i).frequency;
                    
                    % Update position
                    new_position = population(i).position + population(i).velocity;
                    
                    % Apply boundary constraints
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Random walk with probability (1 - pulse_rate)
                    if rand() > population(i).pulse_rate
                        % Generate random walk step
                        epsilon = -1 + 2 * rand();
                        % Calculate mean loudness of all bats
                        loudness_values = zeros(1, search_agents_no);
                        for j = 1:search_agents_no
                            loudness_values(j) = population(j).loudness;
                        end
                        mean_loudness = mean(loudness_values);
                        new_position = leader.position + epsilon * mean_loudness;
                        new_position = max(min(new_position, obj.ub), obj.lb);
                    end
                    
                    % Evaluate new fitness
                    new_fitness = obj.objective_func(new_position);
                    
                    % Create temporary BatMultiMember for comparison
                    new_bat = BatMultiMember(...
                        new_position, ...
                        new_fitness, ...
                        population(i).frequency, ...
                        population(i).velocity, ...
                        population(i).loudness, ...
                        population(i).pulse_rate ...
                    );
                    
                    % Check if new solution is non-dominated compared to current bat
                    current_dominates_new = obj.dominates(population(i), new_bat);
                    
                    % Update if new solution is better and meets loudness criteria
                    if ~current_dominates_new && rand() < population(i).loudness
                        % Update position and fitness
                        population(i).position = new_position;
                        population(i).multi_fitness = new_fitness;
                        
                        % Update loudness and pulse rate
                        population(i).loudness = obj.alpha_loud * population(i).loudness;
                        population(i).pulse_rate = obj.ro * (1 - exp(-obj.gamma_pulse * iter));
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
                
                % Update progress
                if ~isempty(obj.archive)
                    obj.callbacks(iter, max_iter, obj.archive(1));
                else
                    obj.callbacks(iter, max_iter, []);
                end
            end
            
            % Final processing
            obj.history_step_solver = history_archive;
            obj.best_solver = obj.archive;
            
            % End solver
            obj.end_step_solver();
            
            history_archive = obj.history_step_solver;
            archive = obj.archive;
        end
    end
end
