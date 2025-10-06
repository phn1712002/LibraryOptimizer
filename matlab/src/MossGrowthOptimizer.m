classdef MossGrowthOptimizer < Solver
    %{
    Moss Growth Optimization (MGO) algorithm.
    
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
        - w: Inertia weight parameter (default: 2.0)
        - rec_num: Number of positions to record for cryptobiosis (default: 10)
        - divide_num: Number of dimensions to divide (default: dim/4)
        - d1: Probability threshold for spore dispersal (default: 0.2)
    %}
    
    properties
        w          % Inertia weight parameter
        rec_num    % Number of positions to record
        divide_num % Number of dimensions to divide
        d1         % Probability threshold for spore dispersal
    end
    
    methods
        function obj = MossGrowthOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MossGrowthOptimizer constructor - Initialize the MGO solver
            
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
                    Additional MGO parameters:
                    - w: Inertia weight parameter (default: 2.0)
                    - rec_num: Number of positions to record (default: 10)
                    - divide_num: Number of dimensions to divide (default: dim/4)
                    - d1: Probability threshold for spore dispersal (default: 0.2)
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set algorithm name
            obj.name_solver = "Moss Growth Optimizer";
            
            % kwargs (Name-Value pairs)
            if ~isempty(varargin)
                obj.kwargs = obj.nv2struct(varargin{:});
            end
            obj.w = obj.get_kw('w', 2.0);           % Inertia weight parameter
            obj.rec_num = obj.get_kw('rec_num', 10); % Number of positions to record
            obj.divide_num = obj.get_kw('divide_num', max(1, floor(dim / 4))); % Dimensions to divide
            obj.d1 = obj.get_kw('d1', 0.2);         % Probability threshold for spore dispersal
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for MGO algorithm
            
            Inputs:
                search_agents_no : int
                    Number of search agents (population size)
                max_iter : int
                    Maximum number of iterations
                    
            Returns:
                history_step_solver : cell array
                    History of best solutions at each iteration
                best_solver : Member
                    Best solution found overall
            %}
            
            % Initialize storage variables
            history_step_solver = {};
            
            % Initialize the population
            population = obj.init_population(search_agents_no);
            
            % Initialize best solution
            sorted_population = obj.sort_population(population);
            best_solver = sorted_population{1}.copy();
            
            % Initialize cryptobiosis mechanism
            rM = zeros(search_agents_no, obj.dim, obj.rec_num);  % Record history positions
            rM_cos = zeros(search_agents_no, obj.rec_num);       % Record history costs
            
            % Initialize record counter
            rec = 0;
            
            % Call the begin function
            obj.begin_step_solver(max_iter);
            
            % Calculate maximum function evaluations
            max_fes = max_iter * search_agents_no;
            
            % Main optimization loop
            for iter = 1:max_iter
                % Calculate current function evaluations
                current_fes = iter * search_agents_no;
                
                % Calculate progress ratio for adaptive parameters
                progress_ratio = current_fes / max_fes;
                
                % Initialize new population
                new_population = cell(1, search_agents_no);
                for i = 1:search_agents_no
                    new_population{i} = population{i}.copy();
                end
                new_costs = zeros(1, search_agents_no);
                
                % Record the first generation of positions for cryptobiosis
                if rec == 0
                    for i = 1:search_agents_no
                        rM(i, :, rec + 1) = population{i}.position;
                        rM_cos(i, rec + 1) = population{i}.fitness;
                    end
                    rec = rec + 1;
                end
                
                % Process each search agent
                for i = 1:search_agents_no
                    % Select calculation positions based on majority regions
                    cal_positions = obj.get_positions(population);
                    
                    % Divide the population and select regions with more individuals
                    div_indices = randperm(obj.dim);
                    for j = 1:min(obj.divide_num, obj.dim)
                        th = best_solver.position(div_indices(j));
                        index = cal_positions(:, div_indices(j)) > th;
                        if sum(index) < size(cal_positions, 1) / 2
                            index = ~index;  % Choose the side of the majority
                        end
                        cal_positions = cal_positions(index, :);
                        if isempty(cal_positions)
                            break;
                        end
                    end
                    
                    if ~isempty(cal_positions)
                        % Compute distance from individuals to the best
                        D = repmat(best_solver.position, size(cal_positions, 1), 1) - cal_positions;
                        
                        % Calculate the mean of all distances (wind direction)
                        D_wind = mean(D, 1);
                        
                        % Calculate beta and gamma parameters
                        beta = size(cal_positions, 1) / search_agents_no;
                        gamma = 1 / sqrt(1 - beta^2);
                        if beta >= 1
                            gamma = 1.0;
                        end
                        
                        % Calculate step sizes
                        step = obj.w * (rand(1, obj.dim) - 0.5) * (1 - progress_ratio);
                        if gamma > 0
                            step2 = (0.1 * obj.w * (rand(1, obj.dim) - 0.5) * ...
                                   (1 - progress_ratio) * (1 + 0.5 * (1 + tanh(beta / gamma)) * ...
                                   (1 - progress_ratio)));
                        else
                            step2 = 0;
                        end
                        step3 = 0.1 * (rand() - 0.5) * (1 - progress_ratio);
                        
                        % Calculate activation function
                        act_input = 1 ./ (1 + (0.5 - 10 * (rand(1, obj.dim) - 0.5)));
                        act = obj.act_cal(act_input);
                        
                        % Spore dispersal search
                        if rand() > obj.d1
                            new_position = population{i}.position + step .* D_wind;
                        else
                            new_position = population{i}.position + step2 .* D_wind;
                        end
                        
                        % Dual propagation search
                        if rand() < 0.8
                            if rand() > 0.5
                                % Update specific dimension
                                if obj.dim > 0
                                    dim_idx = div_indices(1);
                                    new_position(dim_idx) = best_solver.position(dim_idx) + ...
                                                           step3 * D_wind(dim_idx);
                                end
                            else
                                % Update all dimensions with activation
                                new_position = (1 - act) .* new_position + ...
                                             act .* best_solver.position;
                            end
                        end
                        
                        % Boundary absorption
                        new_position = max(min(new_position, obj.ub), obj.lb);
                        
                        % Evaluate new fitness
                        new_fitness = obj.objective_func(new_position);
                        
                        % Update population member
                        new_population{i}.position = new_position;
                        new_population{i}.fitness = new_fitness;
                        new_costs(i) = new_fitness;
                        
                        % Record for cryptobiosis mechanism
                        if rec < obj.rec_num
                            rM(i, :, rec + 1) = new_position;
                            rM_cos(i, rec + 1) = new_fitness;
                        end
                    else
                        % If no calculation positions, keep current position
                        new_costs(i) = population{i}.fitness;
                    end
                end
                
                % Update population
                for i = 1:search_agents_no
                    population{i} = new_population{i};
                end
                
                % Cryptobiosis mechanism - apply after recording
                if rec >= obj.rec_num || iter == max_iter
                    % Find best historical position for each agent
                    for i = 1:search_agents_no
                        if obj.maximize
                            [~, best_idx] = max(rM_cos(i, 1:rec));
                        else
                            [~, best_idx] = min(rM_cos(i, 1:rec));
                        end
                        population{i}.position = rM(i, :, best_idx);
                        population{i}.fitness = rM_cos(i, best_idx);
                    end
                    
                    % Reset record counter
                    rec = 0;
                else
                    rec = rec + 1;
                end
                
                % Update best solution
                sorted_population = obj.sort_population(population);
                current_best = sorted_population{1};
                if obj.is_better(current_best, best_solver)
                    best_solver = current_best.copy();
                end
                
                % Store the best solution at this iteration
                history_step_solver{end+1} = best_solver.copy();
                
                % Call the callbacks
                obj.callbacks(iter, max_iter, best_solver);
            end
            
            % Final evaluation and storage
            obj.history_step_solver = history_step_solver;
            obj.best_solver = best_solver;
            
            % Call the end function
            obj.end_step_solver();
        end
        
        function act = act_cal(obj, X)
            %{
            act_cal - Activation function calculation
            
            Inputs:
                X : array
                    Input values
                    
            Returns:
                act : array
                    Activated values (0 or 1)
            %}
            
            act = X;
            act(X >= 0.5) = 1;
            act(X < 0.5) = 0;
        end
        
        function sorted_population = sort_population(obj, population)
            %{
            sort_population - Sort population based on fitness
            
            Inputs:
                population : cell array
                    Population to sort
                    
            Returns:
                sorted_population : cell array
                    Sorted population (best first)
            %}
            
            % Extract fitness values
            fitness_values = obj.get_fitness(population);
            
            % Sort based on optimization direction
            if obj.maximize
                [~, sorted_indices] = sort(fitness_values, 'descend');
            else
                [~, sorted_indices] = sort(fitness_values, 'ascend');
            end
            
            % Sort population
            sorted_population = cell(1, length(population));
            for i = 1:length(population)
                sorted_population{i} = population{sorted_indices(i)};
            end
        end
    end
end
