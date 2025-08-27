classdef MultiObjectiveBiogeographyBasedOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveBiogeographyBasedOptimizer - Multi-Objective Biogeography-Based Optimization (BBO) algorithm.
    
    BBO is a population-based optimization algorithm inspired by the migration
    of species between habitats in biogeography. This multi-objective version
    extends the algorithm to handle multiple objectives using archive management
    and grid-based selection.
    
    Parameters:
    -----------
    objective_func : function handle
        Multi-objective function that returns array of fitness values
    lb : float or array
        Lower bounds for variables
    ub : float or array
        Upper bounds for variables  
    dim : int
        Problem dimension
    maximize : bool
        Optimization direction (true for maximize, false for minimize)
    varargin : cell array
        Additional algorithm parameters including:
        - keep_rate: Rate of habitats to keep (default: 0.2)
        - migration_alpha: Migration coefficient (default: 0.9)
        - p_mutation: Mutation probability (default: 0.1)
        - sigma: Mutation step size (default: 2% of variable range)
    %}
    
    properties
        keep_rate
        migration_alpha
        p_mutation
        sigma
    end
    
    methods
        function obj = MultiObjectiveBiogeographyBasedOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveBiogeographyBasedOptimizer constructor - Initialize the MOBBO solver
            
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
            obj.name_solver = "Multi-Objective Biogeography Based Optimizer";
            
            % Set algorithm parameters with defaults
            obj.keep_rate = obj.get_kw('keep_rate', 0.2);
            obj.migration_alpha = obj.get_kw('migration_alpha', 0.9);
            obj.p_mutation = obj.get_kw('p_mutation', 0.1);
            
            % Calculate sigma as 2% of variable range if not provided
            var_range = obj.ub - obj.lb;
            obj.sigma = obj.get_kw('sigma', 0.02 * var_range);
        end
        
        function sorted_population = sort_population(obj, population)
            %{
            sort_population - Sort population based on multi-objective criteria
            
            Inputs:
                population : object array
                    Population to sort
                    
            Returns:
                sorted_population : object array
                    Sorted population (best first)
            %}
            
            % For multi-objective sorting, we use non-dominated sorting
            % This is a simplified version - in practice, you might want to implement
            % proper non-dominated sorting with crowding distance
            
            pop_size = length(population);
            fitness_values = zeros(pop_size, obj.n_objectives);
            
            for i = 1:pop_size
                fitness_values(i, :) = population(i).multi_fitness;
            end
            
            % Simple sorting based on sum of normalized objectives
            normalized_fitness = zeros(pop_size, 1);
            for obj_idx = 1:obj.n_objectives
                obj_vals = fitness_values(:, obj_idx);
                min_val = min(obj_vals);
                max_val = max(obj_vals);
                
                if max_val ~= min_val
                    normalized_obj = (obj_vals - min_val) / (max_val - min_val);
                else
                    normalized_obj = ones(pop_size, 1);
                end
                
                normalized_fitness = normalized_fitness + normalized_obj;
            end
            
            [~, sorted_indices] = sort(normalized_fitness);
            sorted_population = population(sorted_indices);
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective BBO algorithm.
            
            Inputs:
                search_agents_no : int
                    Number of habitats (population size)
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
            
            % Calculate derived parameters
            n_keep = round(obj.keep_rate * search_agents_no);  % Number of habitats to keep
            n_new = search_agents_no - n_keep;  % Number of new habitats
            
            % Initialize migration rates (emigration and immigration)
            mu = linspace(1, 0, search_agents_no);  % Emigration rates (decreasing)
            lambda_rates = 1 - mu;  % Immigration rates (increasing)
            
            % Initialize population
            population = obj.init_population(search_agents_no);
            
            % Initialize archive with non-dominated solutions
            obj.determine_domination(population);
            non_dominated = obj.get_non_dominated_particles(population);
            obj.archive = [obj.archive, non_dominated];
            
            % Initialize grid for archive management
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
                % Create new population for this iteration
                new_population = repmat(MultiObjectiveMember(0, 0), 1, search_agents_no);
                for i = 1:search_agents_no
                    new_population(i) = population(i).copy();
                end
                
                % Update each habitat
                for i = 1:search_agents_no
                    % Migration phase for each dimension
                    for k = 1:obj.dim
                        % Immigration: if random number <= immigration rate
                        if rand() <= lambda_rates(i)
                            % Calculate emigration probabilities (excluding current habitat)
                            ep = mu;
                            ep(i) = 0;  % Set current habitat probability to 0
                            ep_sum = sum(ep);
                            
                            if ep_sum > 0
                                % Select source habitat using roulette wheel selection
                                % For multi-objective, we select from archive for better diversity
                                if ~isempty(obj.archive)
                                    % Select leader from archive using grid-based selection
                                    leader = obj.select_leader();
                                    if ~isempty(leader)
                                        % Perform migration using leader guidance
                                        new_population(i).position(k) = (...
                                            population(i).position(k) + ...
                                            obj.migration_alpha * (leader.position(k) - population(i).position(k))...
                                        );
                                    end
                                else
                                    % Fallback: select from population if archive is empty
                                    j = randi(length(population));
                                    new_population(i).position(k) = (...
                                        population(i).position(k) + ...
                                        obj.migration_alpha * (population(j).position(k) - population(i).position(k))...
                                    );
                                end
                            end
                        end
                        
                        % Mutation: if random number <= mutation probability
                        if rand() <= obj.p_mutation
                            new_population(i).position(k) = new_population(i).position(k) + obj.sigma(k) * randn();
                        end
                    end
                    
                    % Apply bounds constraints
                    new_population(i).position = max(min(new_population(i).position, obj.ub), obj.lb);
                    
                    % Evaluate new fitness
                    new_population(i).multi_fitness = obj.objective_func(new_population(i).position);
                end
                
                % Update archive with new population
                obj = obj.add_to_archive(new_population);
                
                % Sort population for selection (using multi-objective sorting)
                sorted_population_new = obj.sort_population(new_population);
                sorted_population_old = obj.sort_population(population);
                
                % Select next iteration population: keep best + new solutions
                % For multi-objective, we use the sorted population
                next_population = [sorted_population_old(1:min(n_keep, length(sorted_population_old))), ...
                                  sorted_population_new(1:min(n_new, length(sorted_population_new)))];
                
                % Ensure we have exactly the population size
                if length(next_population) > search_agents_no
                    next_population = next_population(1:search_agents_no);
                elseif length(next_population) < search_agents_no
                    % If we don't have enough, fill with random solutions
                    additional = obj.init_population(search_agents_no - length(next_population));
                    next_population = [next_population, additional];
                end
                
                population = next_population;
                
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
