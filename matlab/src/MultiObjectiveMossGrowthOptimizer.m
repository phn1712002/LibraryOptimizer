classdef MultiObjectiveMossGrowthOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveMossGrowthOptimizer - Multi-Objective Moss Growth Optimization (MGO) algorithm
    
    This algorithm extends the standard MGO for multi-objective optimization
    using archive management and grid-based selection.
    
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
        - w: Inertia weight parameter (default: 2.0)
        - rec_num: Number of positions to record for cryptobiosis (default: 10)
        - divide_num: Number of dimensions to divide (default: dim/4)
        - d1: Probability threshold for spore dispersal (default: 0.2)
    %}
    
    properties
        w           % Inertia weight parameter
        rec_num     % Number of positions to record
        divide_num  % Number of dimensions to divide
        d1          % Probability threshold for spore dispersal
    end
    
    methods
        function obj = MultiObjectiveMossGrowthOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveMossGrowthOptimizer constructor - Initialize the MOMGO solver
            
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
            obj.name_solver = "Multi-Objective Moss Growth Optimizer";
            
            % Set MGO-specific parameters with defaults
            obj.w = obj.get_kw('w', 2.0);  % Inertia weight parameter
            obj.rec_num = obj.get_kw('rec_num', 10);  % Number of positions to record
            obj.divide_num = obj.get_kw('divide_num', max(1, floor(dim / 4)));  % Dimensions to divide
            obj.d1 = obj.get_kw('d1', 0.2);  % Probability threshold for spore dispersal
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective MGO
            
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
            
            % Initialize cryptobiosis mechanism
            rM = zeros(search_agents_no, obj.dim, obj.rec_num);  % Record history positions
            rM_cos = zeros(search_agents_no, obj.n_objectives, obj.rec_num);  % Record history multi-fitness
            
            % Initialize record counter
            rec = 0;
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Calculate maximum function evaluations
            max_fes = max_iter * search_agents_no;
            
            % Main optimization loop
            for iter = 1:max_iter
                % Calculate current function evaluations
                current_fes = iter * search_agents_no;
                
                % Calculate progress ratio for adaptive parameters
                progress_ratio = current_fes / max_fes;
                
                % Record the first generation of positions for cryptobiosis
                if rec == 0
                    for i = 1:search_agents_no
                        rM(i, :, rec + 1) = population(i).position;
                        rM_cos(i, :, rec + 1) = population(i).multi_fitness;
                    end
                    rec = rec + 1;
                end
                
                % Process each search agent
                for i = 1:search_agents_no
                    % Select leader from archive using grid-based selection
                    leader = obj.select_leader();
                    if isempty(leader)
                        % If no leader in archive, use random population member
                        leader = population(randi(numel(population)));
                    end
                    
                    % Select calculation positions based on majority regions
                    cal_positions = obj.getPositions(population);
                    
                    % Divide the population and select regions with more individuals
                    div_indices = randperm(obj.dim);
                    for j = 1:min(obj.divide_num, obj.dim)
                        th = leader.position(div_indices(j));
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
                        % Compute distance from individuals to the leader
                        D = leader.position - cal_positions;
                        
                        % Calculate the mean of all distances (wind direction)
                        D_wind = mean(D, 1);
                        
                        % Calculate beta and gamma parameters
                        beta = size(cal_positions, 1) / search_agents_no;
                        if beta < 1
                            gamma = 1 / sqrt(1 - beta^2);
                        else
                            gamma = 1.0;
                        end
                        
                        % Calculate step sizes
                        step = obj.w * (rand(1, obj.dim) - 0.5) * (1 - progress_ratio);
                        if gamma > 0
                            step2 = (0.1 * obj.w * (rand(1, obj.dim) - 0.5) * ...
                                   (1 - progress_ratio) * (1 + 0.5 * (1 + tanh(beta / gamma)) * ...
                                   (1 - progress_ratio)));
                        else
                            step2 = zeros(1, obj.dim);
                        end
                        step3 = 0.1 * (rand() - 0.5) * (1 - progress_ratio);
                        
                        % Calculate activation function
                        act_input = 1 ./ (1 + (0.5 - 10 * (rand(1, obj.dim) - 0.5)));
                        act = obj.act_cal(act_input);
                        
                        % Spore dispersal search
                        if rand() > obj.d1
                            new_position = population(i).position + step .* D_wind;
                        else
                            new_position = population(i).position + step2 .* D_wind;
                        end
                        
                        % Dual propagation search
                        if rand() < 0.8
                            if rand() > 0.5
                                % Update specific dimension
                                if obj.dim > 0
                                    dim_idx = div_indices(1);
                                    new_position(dim_idx) = leader.position(dim_idx) + ...
                                                           step3 * D_wind(dim_idx);
                                end
                            else
                                % Update all dimensions with activation
                                new_position = (1 - act) .* new_position + ...
                                             act .* leader.position;
                            end
                        end
                        
                        % Boundary absorption
                        new_position = max(min(new_position, obj.ub), obj.lb);
                        
                        % Evaluate new fitness
                        new_fitness = obj.objective_func(new_position);
                        new_fitness = new_fitness(:).';
                        
                        % Update population member
                        population(i).position = new_position;
                        population(i).multi_fitness = new_fitness;
                        
                        % Record for cryptobiosis mechanism
                        if rec < obj.rec_num
                            rM(i, :, rec + 1) = new_position;
                            rM_cos(i, :, rec + 1) = new_fitness;
                        end
                    else
                        % If no calculation positions, keep current position
                        % No action needed
                    end
                end
                
                % Cryptobiosis mechanism - apply after recording
                if rec >= obj.rec_num || iter == max_iter
                    % Find best historical position for each agent using Pareto dominance
                    for i = 1:search_agents_no
                        % Create temporary members for historical positions
                        historical_members = [];
                        for j = 1:rec
                            hist_position = squeeze(rM(i, :, j))';
                            hist_fitness = squeeze(rM_cos(i, :, j))';
                            hist_member = MultiObjectiveMember(hist_position, hist_fitness);
                            historical_members = [historical_members, hist_member];
                        end
                        
                        % Determine domination among historical positions
                        historical_members = obj.determine_domination(historical_members);
                        
                        % Get non-dominated historical positions
                        non_dominated_hist = obj.get_non_dominated_particles(historical_members);
                        
                        if ~isempty(non_dominated_hist)
                            % Randomly select one non-dominated historical position
                            selected_idx = randi(numel(non_dominated_hist));
                            selected_hist = non_dominated_hist(selected_idx);
                            population(i).position = selected_hist.position;
                            population(i).multi_fitness = selected_hist.multi_fitness;
                        end
                    end
                    
                    % Reset record counter
                    rec = 0;
                else
                    rec = rec + 1;
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
        
        function act = act_cal(~, X)
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
    end
end
