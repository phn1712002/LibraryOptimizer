classdef TeachingLearningBasedOptimizer < Solver
    %{
    Teaching Learning Based Optimization algorithm.
    
    TeachingLearningBased is a population-based optimization algorithm inspired by the teaching-learning
    process in a classroom. It consists of two main phases:
    1. Teacher Phase: Students learn from the teacher (best solution)
    2. Learner Phase: Students learn from each other through interaction
    
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
        Additional algorithm parameters:
        - teaching_factor_range: Range for teaching factor (default: [1, 2])
    %}
    
    properties
        teaching_factor_range % Teaching factor range (min, max)
    end
    
    methods
        function obj = TeachingLearningBasedOptimizer(objective_func, lb, ub, dim, maximize, varargin)
            %{
            TeachingLearningBasedOptimizer constructor - Initialize the TLBO solver
            
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
                    Additional TLBO parameters:
                    - teaching_factor_range: Range for teaching factor (default: [1, 2])
            %}
            
            % Call parent constructor
            obj@Solver(objective_func, lb, ub, dim, maximize, varargin{:});
            
            % Set algorithm name
            obj.name_solver = "Teaching Learning Based Optimizer";
            
            % kwargs (Name-Value pairs)
            if ~isempty(varargin)
                obj.kwargs = obj.nv2struct(varargin{:});
            end
            obj.teaching_factor_range = obj.get_kw('teaching_factor_range', [1, 2]);  % TF range
        end
        
        function [history_step_solver, best_solver] = solver(obj, search_agents_no, max_iter)
            %{
            solver - Main optimization method for TLBO algorithm
            
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
            
            % Call the begin function
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
                    
                    % Find the teacher (best solution in current population)
                    teacher = obj.find_teacher(population);
                    
                    % Generate teaching factor (TF)
                    tf = obj.generate_teaching_factor();
                    
                    % Generate new solution in teacher phase
                    new_position_teacher = obj.teacher_phase_update(...
                        population{i}.position, teacher.position, mean_position, tf);
                    
                    % Apply bounds and evaluate
                    new_position_teacher = max(min(new_position_teacher, obj.ub), obj.lb);
                    new_fitness_teacher = obj.objective_func(new_position_teacher);
                    
                    % Greedy selection for teacher phase
                    if obj.is_better(Member(new_position_teacher, new_fitness_teacher), population{i})
                        population{i}.position = new_position_teacher;
                        population{i}.fitness = new_fitness_teacher;
                    end
                    
                    % ----------------- Learner Phase ----------------- #
                    % Get partner index
                    partner_idx = partner_indices(i);
                    
                    % Generate new solution in learner phase
                    new_position_learner = obj.learner_phase_update(...
                        population{i}.position, population{partner_idx}.position);
                    
                    % Apply bounds and evaluate
                    new_position_learner = max(min(new_position_learner, obj.ub), obj.lb);
                    new_fitness_learner = obj.objective_func(new_position_learner);
                    
                    % Greedy selection for learner phase
                    if obj.is_better(Member(new_position_learner, new_fitness_learner), population{i})
                        population{i}.position = new_position_learner;
                        population{i}.fitness = new_fitness_learner;
                    end
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
        
        function mean_position = calculate_mean_position(obj, population)
            %{
            calculate_mean_position - Calculate the mean position of the population
            
            Inputs:
                population : cell array
                    Population to calculate mean from
                    
            Returns:
                mean_position : array
                    Mean position vector
            %}
            
            positions = obj.get_positions(population);
            mean_position = mean(positions, 1);
        end
        
        function teacher = find_teacher(obj, population)
            %{
            find_teacher - Find the teacher (best solution) in the population
            
            Inputs:
                population : cell array
                    Population to find teacher from
                    
            Returns:
                teacher : Member
                    Best solution (teacher)
            %}
            
            sorted_population = obj.sort_population(population);
            teacher = sorted_population{1};
        end
        
        function tf = generate_teaching_factor(obj)
            %{
            generate_teaching_factor - Generate a random teaching factor (TF)
            
            Returns:
                tf : int
                    Random teaching factor within the specified range
            %}
            
            tf_min = obj.teaching_factor_range(1);
            tf_max = obj.teaching_factor_range(2);
            tf = randi([tf_min, tf_max]);
        end
        
        function new_position = teacher_phase_update(obj, current_position, teacher_position, mean_position, teaching_factor)
            %{
            teacher_phase_update - Update position in teacher phase
            
            Formula: X_new = X_old + r * (Teacher - TF * Mean)
            
            Inputs:
                current_position : array
                    Current position
                teacher_position : array
                    Teacher's position
                mean_position : array
                    Mean position of population
                teaching_factor : int
                    Teaching factor
                    
            Returns:
                new_position : array
                    Updated position
            %}
            
            r = rand(1, obj.dim);
            new_position = current_position + r .* (teacher_position - teaching_factor * mean_position);
        end
        
        function new_position = learner_phase_update(obj, current_position, partner_position)
            %{
            learner_phase_update - Update position in learner phase
            
            Formula: 
            If current is better than partner: X_new = X_old + r * (X_old - X_partner)
            Else: X_new = X_old + r * (X_partner - X_old)
            
            Inputs:
                current_position : array
                    Current position
                partner_position : array
                    Partner's position
                    
            Returns:
                new_position : array
                    Updated position
            %}
            
            r = rand(1, obj.dim);
            current_fitness = obj.objective_func(current_position);
            partner_fitness = obj.objective_func(partner_position);
            
            if (obj.maximize && current_fitness > partner_fitness) || ...
               (~obj.maximize && current_fitness < partner_fitness)
                % Current individual is better than partner
                new_position = current_position + r .* (current_position - partner_position);
            else
                % Current individual is worse than partner
                new_position = current_position + r .* (partner_position - current_position);
            end
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
