classdef MultiObjectiveSimulatedAnnealingOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveSimulatedAnnealingOptimizer - Multi-Objective Simulated Annealing Optimizer
    
    This algorithm extends the standard Simulated Annealing for multi-objective optimization
    using archive management and Pareto dominance criteria for solution acceptance.
    
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
        - max_temperatures: Maximum number of temperature levels (default: 100)
        - tol_fun: Function tolerance for convergence (default: 1e-4)
        - safety_threshold_df: Safety threshold calculation df (default: 1e-8)
        - equilibrium_steps: Number of steps per temperature level (default: 500)
        - initial_temperature: Starting temperature (default: 1)
        - mu_scaling: Temperature scaling parameter (default: 100.0)
    %}
    
    properties
        max_temperatures
        tol_fun
        equilibrium_steps
        initial_temperature
        safety_threshold_df
        mu_scaling
    end
    
    methods
        function obj = MultiObjectiveSimulatedAnnealingOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveSimulatedAnnealingOptimizer constructor - Initialize the MOSA solver
            
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
            obj.name_solver = "Multi-Objective Simulated Annealing Optimizer";
            
            % Set SA-specific parameters with defaults
            obj.max_temperatures = obj.get_kw('max_temperatures', 100);
            obj.tol_fun = obj.get_kw('tol_fun', 1e-4);
            obj.equilibrium_steps = obj.get_kw('equilibrium_steps', 500);
            obj.initial_temperature = obj.get_kw('initial_temperature', 1);
            obj.safety_threshold_df = obj.get_kw('safety_threshold_df', 1e-8);
            obj.mu_scaling = obj.get_kw('mu_scaling', 100.0);
        end
        
        function [history_archive, archive] = solver(obj, max_iter)
            %{
            solver - Execute the multi-objective simulated annealing optimization algorithm
            
            Inputs:
                max_iter : int
                    Maximum number of iterations (temperature levels)
                    
            Returns:
                history_archive : cell array
                    History of archive states
                archive : cell array
                    Final archive of non-dominated solutions
            %}
            
            % Override max_iter with max_temperatures if provided
            actual_max_iter = min(max_iter, obj.max_temperatures);
            
            % Initialize storage
            history_archive = {};
            
            % Initialize current solution from random position within bounds
            current_position = obj.lb + (obj.ub - obj.lb) .* rand(1, obj.dim);
            current_fitness = obj.objective_func(current_position);
            current_fitness = current_fitness(:).';
            current_solution = MultiObjectiveMember(current_position, current_fitness);
            
            % Initialize archive with current solution
            obj.archive = [current_solution.copy()];
            
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
            obj.begin_step_solver(actual_max_iter);
            
            % Main optimization loop (temperature levels)
            for iter = 1:actual_max_iter
                % Calculate temperature parameter (inverse temperature)
                T = obj.exponential_decay(obj.initial_temperature, 0, iter, actual_max_iter);
                
                % Calculate mu parameter for perturbation scaling
                mu = 10 ^ (T * obj.mu_scaling);
                
                % Simulate thermal equilibrium at this temperature
                for k = 1:obj.equilibrium_steps
                    % Generate perturbation vector
                    random_vector = 2 * rand(1, obj.dim) - 1;  % Range [-1, 1]
                    dx = obj.mu_inv(random_vector, mu) .* (obj.ub - obj.lb);
                    
                    % Generate new candidate position
                    candidate_position = current_solution.position + dx;
                    
                    % Apply bounds constraint
                    candidate_position = max(min(candidate_position, obj.ub), obj.lb);
                    
                    % Evaluate candidate fitness
                    candidate_fitness = obj.objective_func(candidate_position);
                    candidate_fitness = candidate_fitness(:).';
                    candidate_solution = MultiObjectiveMember(candidate_position, candidate_fitness);
                    
                    % Multi-objective acceptance criterion
                    accept = obj.multi_objective_acceptance(current_solution, candidate_solution, T);
                    
                    if accept
                        current_solution = candidate_solution.copy();
                    end
                    
                    % Add candidate to archive for consideration
                    obj = obj.add_to_archive([candidate_solution]);
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
                obj.callbacks(iter, actual_max_iter, best_member);
            end
            
            % Final processing
            obj.history_step_solver = history_archive;
            obj.best_solver = obj.archive;
            
            % End solver
            obj.end_step_solver();
            
            archive = obj.archive;
        end
        
        function accept = multi_objective_acceptance(obj, current, candidate, temperature)
            %{
            multi_objective_acceptance - Multi-objective acceptance criterion based on Pareto dominance and temperature
            
            Inputs:
                current : MultiObjectiveMember
                    Current solution
                candidate : MultiObjectiveMember
                    Candidate solution
                temperature : float
                    Current temperature level
                    
            Returns:
                accept : bool
                    Whether to accept the candidate solution
            %}
            
            % Check if candidate dominates current
            if obj.dominates(candidate, current)
                accept = true;
                return;
            end
            
            % Check if current dominates candidate
            if obj.dominates(current, candidate)
                accept = false;
                return;
            end
            
            % If neither dominates the other (non-dominated), use probabilistic acceptance
            % Calculate a composite fitness difference for probabilistic acceptance
            current_avg = mean(current.multi_fitness);
            candidate_avg = mean(candidate.multi_fitness);
            
            if obj.maximize
                % For maximization: accept if candidate is better on average or with probability
                df = candidate_avg - current_avg;
            else
                % For minimization: accept if candidate is better on average or with probability
                df = current_avg - candidate_avg;
            end
            
            % Calculate scale for normalization
            if ~isempty(obj.archive)
                archive_fitness = obj.get_fitness(obj.archive);
                scale = mean(std(archive_fitness, 0, 2));
            else
                scale = 1;
            end
            
            if scale > obj.safety_threshold_df
                df = df / scale;
            end
            
            accept_prob = exp(df / (temperature + eps));
            accept = rand() < accept_prob;
        end
        
        function dx = mu_inv(~, y, mu)
            %{
            mu_inv - Generate perturbation vectors according to the mu_inv function
            
            This function is used to generate new candidate points with perturbations
            that are proportional to the current temperature level.
            
            Inputs:
                y : array
                    Random vector in range [-1, 1]
                mu : float
                    Temperature scaling parameter
                    
            Returns:
                dx : array
                    Scaled perturbation vector
            %}
            
            dx = (((1 + mu) .^ abs(y) - 1) ./ mu) .* sign(y);
        end
        
        function value = exponential_decay(~, initial_value, final_value, current_iter, max_iter)
            %{
            exponential_decay - Calculate exponential decay value
            
            Inputs:
                initial_value : float
                    Initial value at iteration 0
                final_value : float
                    Final value at max_iter
                current_iter : int
                    Current iteration
                max_iter : int
                    Maximum number of iterations
                    
            Returns:
                value : float
                    Decayed value
            %}
            
            value = final_value + (initial_value - final_value) * exp(-current_iter / max_iter);
        end
    end
end
