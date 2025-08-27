classdef MultiObjectiveTeachingLearningBasedOptimizer < MultiObjectiveSolver
    %{
    MultiObjectiveTeachingLearningBasedOptimizer - Multi-Objective Teaching Learning Based Optimizer
    
    This algorithm extends the standard TLBO for multi-objective optimization
    using archive management and grid-based selection for teacher selection.
    
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
        - teaching_factor_range: Range for teaching factor (default: [1, 2])
    %}
    
    properties
        teaching_factor_range
    end
    
    methods
        function obj = MultiObjectiveTeachingLearningBasedOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            MultiObjectiveTeachingLearningBasedOptimizer constructor - Initialize the MOTLBO solver
            
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
            obj.name_solver = "Multi-Objective Teaching Learning Based Optimizer";
            
            % Set TLBO-specific parameters with defaults
            obj.teaching_factor_range = obj.get_kw('teaching_factor_range', [1, 2]);
        end
        
        function [history_archive, archive] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for multi-objective TLBO
            
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
            
            % Start solver
            obj.begin_step_solver(max_iter);
            
            % Main optimization loop
            for iter = 1:max_iter
                % Partner selection for all students (shuffle indices)
                partner_indices = randperm(search_agents_no);
                
                % Process each student in the population
                for i = 1:search_agents_no
                    % ----------------- Teacher Phase ----------------- #
                    % Calculate mean of population
                    mean_position = obj.calculate_mean_position(population);
                    
                    % Find the teacher from archive using grid-based selection
                    teacher = obj.select_teacher();
                    
                    % If no teacher in archive, use best from current population
                    if isempty(teacher)
                        teacher = obj.find_best_in_population(population);
                    end
                    
                    % Generate teaching factor (TF)
                    tf = obj.generate_teaching_factor();
                    
                    % Generate new solution in teacher phase
                    new_position_teacher = obj.teacher_phase_update(...
                        population(i).position, teacher.position, mean_position, tf);
                    
                    % Apply bounds and evaluate
                    new_position_teacher = max(min(new_position_teacher, obj.ub), obj.lb);
                    new_fitness_teacher = obj.objective_func(new_position_teacher);
                    new_fitness_teacher = new_fitness_teacher(:).';
                    
                    % Create new member for comparison
                    new_member_teacher = MultiObjectiveMember(new_position_teacher, new_fitness_teacher);
                    
                    % Greedy selection for teacher phase using Pareto dominance
                    if ~obj.dominates(population(i), new_member_teacher)
                        population(i).position = new_position_teacher;
                        population(i).multi_fitness = new_fitness_teacher;
                    end
                    
                    % ----------------- Learner Phase ----------------- #
                    % Get partner index
                    partner_idx = partner_indices(i);
                    
                    % Skip if partner is the same as current student
                    if partner_idx == i
                        partner_idx = mod(partner_idx, search_agents_no) + 1;
                    end
                    
                    % Generate new solution in learner phase
                    new_position_learner = obj.learner_phase_update(...
                        population(i).position, population(partner_idx).position);
                    
                    % Apply bounds and evaluate
                    new_position_learner = max(min(new_position_learner, obj.ub), obj.lb);
                    new_fitness_learner = obj.objective_func(new_position_learner);
                    new_fitness_learner = new_fitness_learner(:).';
                    
                    % Create new member for comparison
                    new_member_learner = MultiObjectiveMember(new_position_learner, new_fitness_learner);
                    
                    % Greedy selection for learner phase using Pareto dominance
                    if ~obj.dominates(population(i), new_member_learner)
                        population(i).position = new_position_learner;
                        population(i).multi_fitness = new_fitness_learner;
                    end
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
        
        function mean_position = calculate_mean_position(obj, population)
            %{
            calculate_mean_position - Calculate the mean position of the population
            
            Inputs:
                population : array of MultiObjectiveMember
                    Population to calculate mean from
                    
            Returns:
                mean_position : array
                    Mean position vector
            %}
            
            positions = obj.getPositions(population);
            mean_position = mean(positions, 1);
        end
        
        function teacher = select_teacher(obj)
            %{
            select_teacher - Select teacher from archive using grid-based selection
            
            Returns:
                teacher : MultiObjectiveMember
                    Selected teacher from archive
            %}
            
            teacher = obj.select_leader();
        end
        
        function best_member = find_best_in_population(obj, population)
            %{
            find_best_in_population - Find a good solution in population using grid-based selection
            
            Inputs:
                population : array of MultiObjectiveMember
                    Population to select from
                    
            Returns:
                best_member : MultiObjectiveMember
                    Selected member from population
            %}
            
            if isempty(population)
                best_member = [];
                return;
            end
            
            % Use grid-based selection if archive has members
            if ~isempty(obj.archive)
                % Select multiple leaders and choose one
                leaders = obj.select_multiple_leaders(min(3, numel(obj.archive)));
                if ~isempty(leaders)
                    best_member = leaders(randi(numel(leaders)));
                    return;
                end
            end
            
            % Fallback: return random member from population
            best_member = population(randi(numel(population)));
        end
        
        function teaching_factor = generate_teaching_factor(obj)
            %{
            generate_teaching_factor - Generate a random teaching factor (TF)
            
            Returns:
                teaching_factor : int
                    Random teaching factor within the specified range
            %}
            
            tf_min = obj.teaching_factor_range(1);
            tf_max = obj.teaching_factor_range(2);
            teaching_factor = randi([tf_min, tf_max]);
        end
        
        function new_position = teacher_phase_update(obj, current_position, teacher_position, mean_position, teaching_factor)
            %{
            teacher_phase_update - Update position in teacher phase
            
            Formula: X_new = X_old + r * (Teacher - TF * Mean)
            
            Inputs:
                current_position : array
                    Current position vector
                teacher_position : array
                    Teacher position vector
                mean_position : array
                    Mean position vector of population
                teaching_factor : int
                    Teaching factor
                    
            Returns:
                new_position : array
                    Updated position vector
            %}
            
            r = rand(1, obj.dim);
            new_position = current_position + r .* (teacher_position - teaching_factor * mean_position);
        end
        
        function new_position = learner_phase_update(obj, current_position, partner_position)
            %{
            learner_phase_update - Update position in learner phase
            
            Formula:
            If current dominates partner: X_new = X_old + r * (X_old - X_partner)
            Else: X_new = X_old + r * (X_partner - X_old)
            
            Inputs:
                current_position : array
                    Current position vector
                partner_position : array
                    Partner position vector
                    
            Returns:
                new_position : array
                    Updated position vector
            %}
            
            r = rand(1, obj.dim);
            
            % Create temporary members for dominance comparison
            current_fitness = obj.objective_func(current_position);
            current_fitness = current_fitness(:).';
            partner_fitness = obj.objective_func(partner_position);
            partner_fitness = partner_fitness(:).';
            
            current_member = MultiObjectiveMember(current_position, current_fitness);
            partner_member = MultiObjectiveMember(partner_position, partner_fitness);
            
            if obj.dominates(current_member, partner_member)
                new_position = current_position + r .* (current_position - partner_position);
            else
                new_position = current_position + r .* (partner_position - current_position);
            end
        end
    end
end
