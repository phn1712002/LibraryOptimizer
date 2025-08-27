classdef MultiObjectivePrairieDogsOptimizer < MultiObjectiveSolver
    %{
    MultiObjectivePrairieDogsOptimizer - Multi-Objective Prairie Dogs Optimization (PDO) algorithm
    
    Based on the MATLAB implementation from:
    Absalom E. Ezugwu, Jeffrey O. Agushaka, Laith Abualigah, Seyedali Mirjalili, Amir H Gandomi
    "Prairie Dogs Optimization: A Nature-inspired Metaheuristic"
    
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
        Additional algorithm parameters:
        - rho: float (default=0.005) - Account for individual PD difference
        - eps_pd: float (default=0.1) - Food source alarm parameter
        - eps: float (default=1e-10) - Small epsilon value for numerical stability
        - beta: float (default=1.5) - Levy flight parameter
    %}
    
    properties
        rho       % Account for individual PD difference
        eps_pd    % Food source alarm parameter
        eps       % Small epsilon for numerical stability
        beta      % Levy flight parameter
    end
    
    methods
        function obj = MultiObjectivePrairieDogsOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectivePrairieDogsOptimizer constructor - Initialize the MOPDO solver
            
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
            obj.name_solver = "Multi-Objective Prairie Dogs Optimizer";
            
            % Set PDO-specific parameters with defaults
            obj.rho = obj.get_kw('rho', 0.005);    % Account for individual PD difference
            obj.eps_pd = obj.get_kw('eps_pd', 0.1); % Food source alarm
            obj.eps = obj.get_kw('eps', 1e-10);    % Small epsilon for numerical stability
            obj.beta = obj.get_kw('beta', 1.5);    % Levy flight parameter
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective PDO
            
            Inputs:
                search_agents_no : int
                    Number of search agents (prairie dogs)
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
            population = obj.determine_domination(population);
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
                % Determine mu value based on iteration parity
                if mod(iter, 2) == 0
                    mu = -1;
                else
                    mu = 1;
                end
                
                % Calculate dynamic parameters
                DS = 1.5 * randn() * (1 - iter/max_iter) ^ (2 * iter/max_iter) * mu;  % Digging strength
                PE = 1.5 * (1 - iter/max_iter) ^ (2 * iter/max_iter) * mu;            % Predator effect
                
                % Generate Levy flight steps for all prairie dogs
                RL = zeros(search_agents_no, obj.dim);
                for i = 1:search_agents_no
                    RL(i, :) = obj.levy_flight();
                end
                
                % Select leader from archive for guidance
                leader = obj.select_leader();
                if isempty(leader)
                    % If no leader in archive, use random member from population
                    leader = population(randi(numel(population)));
                end
                
                % Create matrix of leader positions for all prairie dogs
                TPD = repmat(leader.position, search_agents_no, 1);
                
                % Update each prairie dog's position
                for i = 1:search_agents_no
                    new_position = zeros(1, obj.dim);
                    
                    for j = 1:obj.dim
                        % Choose a random prairie dog different from current one
                        k = randi(search_agents_no);
                        while k == i
                            k = randi(search_agents_no);
                        end
                        
                        % Calculate PDO-specific parameters
                        cpd = rand() * (TPD(i, j) - population(k).position(j)) / (TPD(i, j) + obj.eps);
                        P = obj.rho + (population(i).position(j) - mean(population(i).position)) / ...
                            (TPD(i, j) * (obj.ub(j) - obj.lb(j)) + obj.eps);
                        eCB = leader.position(j) * P;
                        
                        % Different position update strategies based on iteration phase
                        if iter < max_iter / 4
                            new_position(j) = leader.position(j) - eCB * obj.eps_pd - cpd * RL(i, j);
                        elseif iter < 2 * max_iter / 4
                            new_position(j) = leader.position(j) * population(k).position(j) * DS * RL(i, j);
                        elseif iter < 3 * max_iter / 4
                            new_position(j) = leader.position(j) * PE * rand();
                        else
                            new_position(j) = leader.position(j) - eCB * obj.eps - cpd * rand();
                        end
                    end
                    
                    % Apply bounds
                    new_position = max(min(new_position, obj.ub), obj.lb);
                    
                    % Update prairie dog position and fitness
                    population(i).position = new_position;
                    new_fitness = obj.objective_func(new_position);
                    new_fitness = new_fitness(:).';
                    population(i).multi_fitness = new_fitness;
                end
                
                % Update archive with current population
                obj = obj.add_to_archive(population);
                
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
        
        function step = levy_flight(obj)
            %{
            levy_flight - Generate Levy flight step
            
            Returns:
                step : array
                    Levy flight step vector
            %}
            
            % Generate Levy flight step
            sigma = (gamma(1 + obj.beta) * sin(pi * obj.beta / 2) / ...
                    (gamma((1 + obj.beta) / 2) * obj.beta * 2^((obj.beta - 1)/2)))^(1/obj.beta);
            
            u = randn(1, obj.dim) * sigma;
            v = randn(1, obj.dim);
            step = u ./ (abs(v).^(1/obj.beta));
        end
    end
end
