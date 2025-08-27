classdef SimulatedAnnealingOptimizer < Solver
    %{
    Simulated Annealing Optimizer implementation based on Kirkpatrick et al. (1983).
    
    This algorithm minimizes or maximizes a function using the simulated annealing
    method, which is particularly effective for finding global optima in complex
    search spaces with many local optima.
    
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
        Additional algorithm parameters including:
        - max_temperatures: Maximum number of temperature levels (default: 100)
        - tol_fun: Function tolerance for convergence (default: 1e-4)
        - equilibrium_steps: Number of steps per temperature level (default: 500)
        - initial_temperature: Starting temperature (default: None, auto-calculated)
        - mu_scaling: Temperature scaling parameter (default: 100.0)
    %}
    
    properties
        max_temperatures    % Maximum number of temperature levels
        tol_fun             % Function tolerance for convergence
        equilibrium_steps   % Number of steps per temperature level
        initial_temperature % Starting temperature
        mu_scaling          % Temperature scaling parameter
    end
    
    methods
        function obj = SimulatedAnnealingOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            SimulatedAnnealingOptimizer constructor - Initialize the SA solver
            
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
                    Additional SA parameters:
                    - max_temperatures: Maximum number of temperature levels (default: 100)
                    - tol_fun: Function tolerance for convergence (default: 1e-4)
                    - equilibrium_steps: Number of steps per temperature level (default: 500)
                    - initial_temperature: Starting temperature (default: auto-calculated)
                    - mu_scaling: Temperature scaling parameter (default: 100.0)
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set algorithm name
            obj.name_solver = "Simulated Annealing Optimizer";
            
            % kwargs (Name-Value pairs)
            if ~isempty(varargin)
                obj.kwargs = obj.nv2struct(varargin{:});
            end
            obj.max_temperatures = obj.get_kw('max_temperatures', 100);
            obj.tol_fun = obj.get_kw('tol_fun', 1e-4);
            obj.equilibrium_steps = obj.get_kw('equilibrium_steps', 500);
            obj.initial_temperature = obj.get_kw('initial_temperature', []);
            obj.mu_scaling = obj.get_kw('mu_scaling', 100.0);
        end
        
        function [history_step_solver, best_solver] = solver(obj, max_iter)
            %{
            solver - Main optimization method for SA algorithm
            
            Inputs:
                max_iter : int
                    Maximum number of iterations (temperature levels)
                    
            Returns:
                history_step_solver : cell array
                    History of best solutions at each iteration
                best_solver : Member
                    Best solution found overall
            %}
            
            % Override max_iter with max_temperatures if provided
            actual_max_iter = min(max_iter, obj.max_temperatures);
            
            % Initialize storage variables
            history_step_solver = {};
            
            % Initialize current solution from random position within bounds
            current_solution = obj.init_population(1);
            
            % Initialize best solution
            current_solution = current_solution{1}.copy();
            best_solver = current_solution;
            
            % Call the begin function
            obj.begin_step_solver(actual_max_iter);
            
            % Main optimization loop (temperature levels)
            for iter = 1:actual_max_iter
                % Calculate temperature parameter (inverse temperature)
                T = iter / actual_max_iter;  % Ranges from 0 to 1
                
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
                    candidate_solution = Member(candidate_position, candidate_fitness);
                    
                    % Calculate fitness difference
                    df = candidate_fitness - current_solution.fitness;
                    
                    % Metropolis acceptance criterion
                    if obj.maximize
                        % For maximization: accept if better or with probability exp(df/T)
                        accept = (df > 0) || (rand() > ...
                                exp(T * df / (abs(current_solution.fitness) + eps) / obj.tol_fun));
                    else
                        % For minimization: accept if better or with probability exp(-df/T)
                        accept = (df < 0) || (rand() < ...
                                exp(-T * df / (abs(current_solution.fitness) + eps) / obj.tol_fun));
                    end
                    
                    if accept
                        current_solution = candidate_solution.copy();
                    end
                    
                    % Update best solution if current is better
                    if obj.is_better(current_solution, best_solver)
                        best_solver = current_solution.copy();
                    end
                end
                
                % Store the best solution at this temperature level
                history_step_solver{end+1} = best_solver.copy();
                
                % Call the callbacks
                obj.callbacks(iter, actual_max_iter, best_solver);
            end
            
            % Final evaluation and storage
            obj.history_step_solver = history_step_solver;
            obj.best_solver = best_solver;
            
            % Call the end function
            obj.end_step_solver();
        end
        
        function perturbation = mu_inv(obj, y, mu)
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
                perturbation : array
                    Scaled perturbation vector
            %}
            
            perturbation = (((1 + mu) .^ abs(y) - 1) ./ mu) .* sign(y);
        end
        
        function s = get_kw(obj, name, default)
            if isfield(obj.kwargs, name), s = obj.kwargs.(name);
            else, s = default; end
        end
    end
end
