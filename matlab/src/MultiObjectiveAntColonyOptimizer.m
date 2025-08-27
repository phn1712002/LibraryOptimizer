classdef MultiObjectiveAntColonyOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveAntColonyOptimizer - Multi-Objective Ant Colony Optimization for Continuous Domains (MO-ACOR)
    
    This algorithm extends the standard ACOR for multi-objective optimization
    using archive management and grid-based selection for solution guidance.
    
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
        - q: Intensification factor (selection pressure), default 0.5
        - zeta: Deviation-distance ratio, default 1.0

    %}
    
    properties
        q
        zeta
    end
    
    methods
        function obj = MultiObjectiveAntColonyOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveAntColonyOptimizer constructor - Initialize the MO-ACOR solver
            
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
            obj.name_solver = "Multi-Objective Ant Colony Optimizer";
            
            % Set algorithm-specific parameters with defaults
            obj.q = obj.get_kw('q', 0.5);  % Intensification factor
            obj.zeta = obj.get_kw('zeta', 1.0);  % Deviation-distance ratio
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective ACOR
            
            Inputs:
                search_agents_no : int
                    Number of search agents (population/archive size)
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
            
            % Initialize population (archive)
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
                % Calculate solution weights (Gaussian kernel weights)
                w = obj.calculate_weights(numel(population));
                
                % Calculate selection probabilities
                p = w / sum(w);
                
                % Calculate means (positions of all solutions in population)
                means = zeros(numel(population), obj.dim);
                for i = 1:numel(population)
                    means(i, :) = population(i).position;
                end
                
                % Calculate standard deviations for each solution
                sigma = obj.calculate_standard_deviations(means);
                
                % Create new population by sampling from Gaussian distributions
                new_population = obj.sample_new_population(means, sigma, p, search_agents_no);
                
                % Update archive with new population
                obj = obj.add_to_archive(new_population);
                
                % Update population for next iteration (use archive as new population)
                % This maintains diversity and focuses search on promising regions
                if numel(obj.archive) >= search_agents_no
                    % Select diverse solutions from archive using grid-based selection
                    population = obj.select_diverse_population(search_agents_no);
                else
                    % If archive is smaller than population size, use archive + random solutions
                    population = obj.archive;
                    remaining = search_agents_no - numel(population);
                    if remaining > 0
                        additional = obj.init_population(remaining);
                        population = [population, additional];
                    end
                end
                
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
        
        function w = calculate_weights(obj, n_pop)
            %{
            calculate_weights - Calculate Gaussian kernel weights for solution selection.
            
            Inputs:
                n_pop : int
                    Population size
                    
            Returns:
                w : array
                    Array of weights for each solution
            %}
            w = (1 / (sqrt(2 * pi) * obj.q * n_pop)) * ...
                exp(-0.5 * (((0:n_pop-1) / (obj.q * n_pop)) .^ 2));
        end
        
        function sigma = calculate_standard_deviations(obj, means)
            %{
            calculate_standard_deviations - Calculate standard deviations for Gaussian sampling.
            
            Inputs:
                means : array
                    Array of solution positions (means)
                    
            Returns:
                sigma : array
                    Array of standard deviations for each solution
            %}
            n_pop = size(means, 1);
            sigma = zeros(size(means));
            
            for l = 1:n_pop
                % Calculate average distance to other solutions
                D = sum(abs(means(l, :) - means), 1);
                sigma(l, :) = obj.zeta * D / (n_pop - 1);
            end
        end
        
        function new_population = sample_new_population(obj, means, sigma, probabilities, n_sample)
            %{
            sample_new_population - Sample new solutions using Gaussian distributions.
            
            Inputs:
                means : array
                    Array of solution positions (means)
                sigma : array
                    Array of standard deviations
                probabilities : array
                    Selection probabilities for each solution
                n_sample : int
                    Number of samples to generate
                    
            Returns:
                new_population : cell array
                    Cell array of newly sampled solutions
            %}
            new_population = cell(1, n_sample);
            
            for sample_idx = 1:n_sample
                % Initialize new position
                new_position = zeros(1, obj.dim);
                
                % Construct solution component by component
                for i = 1:obj.dim
                    % Select Gaussian kernel using roulette wheel selection
                    l = obj.roulette_wheel_selection(probabilities);
                    
                    % Generate Gaussian random variable
                    new_position(i) = means(l, i) + sigma(l, i) * randn();
                end
                
                % Ensure positions stay within bounds
                new_position = max(min(new_position, obj.ub), obj.lb);
                
                % Evaluate fitness
                new_fitness = obj.objective_func(new_position);
                new_fitness = new_fitness(:).';
                
                % Create new member
                new_population{sample_idx} = MultiObjectiveMember(new_position, new_fitness);
            end
            
            % Convert cell array to object array
            new_population = [new_population{:}];
        end
        
        function selected_population = select_diverse_population(obj, n_select)
            %{
            select_diverse_population - Select diverse population from archive using grid-based selection.
            
            Inputs:
                n_select : int
                    Number of solutions to select
                    
            Returns:
                selected_population : cell array
                    Selected diverse population
            %}
            if isempty(obj.archive) || n_select <= 0
                selected_population = {};
                return;
            end
            
            % Get grid indices of all archive members
            grid_indices = [obj.archive.grid_index];
            valid_indices = ~arrayfun(@isempty, {obj.archive.grid_index});
            grid_indices = grid_indices(valid_indices);
            
            if isempty(grid_indices)
                % If no grid indices, return random unique members
                n_available = min(n_select, numel(obj.archive));
                selected_indices = randperm(numel(obj.archive), n_available);
                selected_population = obj.archive(selected_indices);
                return;
            end
            
            % Get occupied cells and their counts
            [occupied_cells, ~, ic] = unique(grid_indices);
            counts = accumarray(ic, 1);
            n_cells = numel(occupied_cells);
            
            % Selection probabilities (lower density cells have higher probability)
            probabilities = exp(-obj.beta_leader * counts);
            probabilities = probabilities / sum(probabilities);
            
            selected_population = MultiObjectiveMember.empty;
            temp_probabilities = probabilities;
            temp_cells = occupied_cells;
            
            % Select solutions from different cells to maintain diversity
            for select_idx = 1:min(n_select, n_cells)
                if isempty(temp_cells)
                    break;
                end
                
                % Select a cell using roulette wheel
                r = rand();
                cum_probs = cumsum(temp_probabilities);
                selected_cell_idx = find(r <= cum_probs, 1, 'first');
                selected_cell = temp_cells(selected_cell_idx);
                
                % Get members in selected cell
                cell_members = obj.archive([obj.archive.grid_index] == selected_cell);
                
                % Select one member from the cell
                selected_member = cell_members(randi(numel(cell_members)));
                selected_population = [selected_population, selected_member];
                
                % Remove selected cell from consideration
                mask = temp_cells ~= selected_cell;
                temp_cells = temp_cells(mask);
                temp_probabilities = temp_probabilities(mask);
                if ~isempty(temp_probabilities)
                    temp_probabilities = temp_probabilities / sum(temp_probabilities);
                end
            end
            
            % If we need more solutions than available cells, fill with random selection
            if numel(selected_population) < n_select
                remaining = n_select - numel(selected_population);
                available_members = setdiff(obj.archive, selected_population);
                if ~isempty(available_members)
                    n_additional = min(remaining, numel(available_members));
                    additional_indices = randperm(numel(available_members), n_additional);
                    selected_population = [selected_population, available_members(additional_indices)];
                end
            end
        end
        
        function index = roulette_wheel_selection(~, probabilities)
            %{
            roulette_wheel_selection - Perform roulette wheel selection.
            
            Inputs:
                probabilities : array
                    Selection probabilities (should sum to 1.0)
                    
            Returns:
                index : int
                    Index of the selected individual
            %}
            r = rand();
            cumulative_sum = cumsum(probabilities);
            index = find(r <= cumulative_sum, 1, 'first');
            if isempty(index)
                index = numel(probabilities);
            end
        end
    end
end
