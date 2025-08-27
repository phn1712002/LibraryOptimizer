classdef MultiObjectiveCuckooSearchOptimizer < MultiObjectiveSolver
    %{
    Multi-Objective Cuckoo Search Optimizer
    
    This algorithm extends the standard Cuckoo Search for multi-objective optimization
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
        - pa: Discovery rate of alien eggs/solutions (default: 0.25)
        - beta: Levy exponent for flight steps (default: 1.5)
    %}
    
    properties
        pa
        beta
    end
    
    methods
        function obj = MultiObjectiveCuckooSearchOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveCuckooSearchOptimizer constructor
            
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
            obj.name_solver = "Multi-Objective Cuckoo Search Optimizer";
            
            % Set algorithm parameters with defaults
            obj.pa = obj.get_kw('pa', 0.25);  % Discovery rate
            obj.beta = obj.get_kw('beta', 1.5);  % Levy exponent
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective Cuckoo Search
            
            Inputs:
                search_agents_no : int
                    Number of nests (search agents)
                max_iter : int
                    Maximum number of iterations
                    
            Returns:
                Tuple containing:
                    - history_archive: Cell array of archive states at each iteration
                    - archive: Final Pareto archive
            %}
            
            % Initialize storage
            history_archive = {};
            
            % Initialize population of nests
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
                % Generate new solutions via Levy flights
                new_population = obj.get_cuckoos(population);
                
                % Evaluate new solutions and update population
                population = obj.update_population(population, new_population);
                
                % Discovery and randomization: abandon some nests and build new ones
                abandoned_nests = obj.empty_nests(population);
                
                % Evaluate abandoned nests and update population
                population = obj.update_population(population, abandoned_nests);
                
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
        
        function new_population = get_cuckoos(obj, population)
            %{
            get_cuckoos - Generate new solutions via Levy flights
            
            Inputs:
                population : array
                    Current population of nests
                    
            Returns:
                new_population : array
                    New solutions generated via Levy flights
            %}
            
            new_population = MultiObjectiveMember.empty(length(population), 0);
            
            for i = 1:length(population)
                % Select a leader from archive using grid-based selection
                leader = obj.select_leader();
                
                % If no leader in archive, use random nest from population
                if isempty(leader)
                    leader_idx = randi(length(population));
                    leader = population(leader_idx);
                end
                
                % Generate Levy flight step
                step = obj.levy_flight();
                
                % Scale step size (0.01 factor as in original implementation)
                step_size = 0.01 * step .* (population(i).position - leader.position);
                
                % Generate new position
                new_position = population(i).position + step_size .* randn(1, obj.dim);
                
                % Apply bounds
                new_position = max(min(new_position, obj.ub), obj.lb);
                
                % Evaluate fitness
                new_fitness = obj.objective_func(new_position);
                
                % Create new member
                new_population(i) = MultiObjectiveMember(new_position, new_fitness);
            end
        end
        
        function new_nests = empty_nests(obj, population)
            %{
            empty_nests - Discover and replace abandoned nests
            
            Inputs:
                population : array
                    Current population of nests
                    
            Returns:
                new_nests : array
                    New nests to replace abandoned ones
            %}
            
            n = length(population);
            new_nests = MultiObjectiveMember.empty(n, 0);
            
            % Create discovery status vector
            discovery_status = rand(1, n) > obj.pa;
            
            for i = 1:n
                if discovery_status(i)
                    % This nest is discovered and will be abandoned
                    % Generate new solution via random walk
                    idx1 = randi(n);
                    idx2 = randi(n);
                    while idx2 == idx1
                        idx2 = randi(n);
                    end
                    
                    step_size = rand() * (population(idx1).position - population(idx2).position);
                    
                    new_position = population(i).position + step_size;
                    
                    % Apply bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Evaluate fitness
                    new_fitness = obj.objective_func(new_position);
                    
                    new_nests(i) = MultiObjectiveMember(new_position, new_fitness);
                else
                    % Keep the original nest
                    new_nests(i) = population(i).copy();
                end
            end
        end
        
        function updated_population = update_population(obj, current_population, new_population)
            %{
            update_population - Update population by keeping better solutions
            
            Inputs:
                current_population : array
                    Current population
                new_population : array
                    Newly generated population
                    
            Returns:
                updated_population : array
                    Updated population with better solutions
            %}
            
            updated_population = MultiObjectiveMember.empty(length(current_population), 0);
            
            for i = 1:length(current_population)
                % Check domination between current and new solution
                current_dominates_new = obj.dominates(current_population(i), new_population(i));
                new_dominates_current = obj.dominates(new_population(i), current_population(i));
                
                if new_dominates_current
                    % New solution dominates current - keep new
                    updated_population(i) = new_population(i);
                elseif ~current_dominates_new && ~new_dominates_current
                    % Neither dominates the other - randomly choose one
                    if rand() > 0.5
                        updated_population(i) = new_population(i);
                    else
                        updated_population(i) = current_population(i);
                    end
                else
                    % Current solution dominates new - keep current
                    updated_population(i) = current_population(i);
                end
            end
        end
        
        function step = levy_flight(obj)
            %{
            levy_flight - Generate Levy flight step
            
            Returns:
                step : array
                    Levy flight step vector
            %}
            
            % Generate Levy flight step using Mantegna's algorithm
            sigma_u = (gamma(1 + obj.beta) * sin(pi * obj.beta / 2) / ...
                      (gamma((1 + obj.beta) / 2) * obj.beta * 2^((obj.beta - 1) / 2)))^(1 / obj.beta);
            
            u = randn(1, obj.dim) * sigma_u;
            v = randn(1, obj.dim);
            
            step = u ./ (abs(v).^(1 / obj.beta));
        end
    end
end
