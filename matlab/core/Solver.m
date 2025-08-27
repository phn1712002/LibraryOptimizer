classdef Solver
    %{
    Solver - Base class for optimization solvers
    
    This abstract base class provides common functionality for all optimization
    algorithms including population initialization, progress tracking, and
    result visualization.
    %}
    
    properties
        objective_func
        dim
        lb
        ub
        maximize
        show_chart
        history_step_solver
        best_solver
        name_solver
    end
    
    methods
        function obj = Solver(objective_func, lb, ub, dim, maximize, varargin)
            %{
            Solver constructor - Initialize the Solver base class
            
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
                    Additional solver parameters including:
                    - show_chart (bool): Whether to display convergence chart (default: true)
            %}
            
            obj.objective_func = objective_func;
            obj.dim = dim;
            
            % Convert bounds to arrays for vectorized operations
            if isscalar(lb)
                obj.lb = repmat(lb, 1, dim);
            else
                obj.lb = lb;
            end
            
            if isscalar(ub)
                obj.ub = repmat(ub, 1, dim);
            else
                obj.ub = ub;
            end
            
            obj.maximize = maximize;
            
            % Parse additional parameters
            p = inputParser;
            addParameter(p, 'show_chart', true, @islogical);
            parse(p, varargin{:});
            obj.show_chart = p.Results.show_chart;
            
            % Initialize optimization history and best solution
            obj.history_step_solver = [];
            if maximize
                initial_fitness = -inf;
            else
                initial_fitness = inf;
            end
            obj.best_solver = Member(rand(1, dim) .* (obj.ub - obj.lb) + obj.lb, initial_fitness);
            
            obj.name_solver = "";  % Solver name for display
        end
        
        function positions = get_positions(obj, population)
            %{
            get_positions - Extract positions from population
            
            Inputs:
                population : cell array of Member objects
                    Population to extract positions from
                    
            Returns:
                positions : array
                    Array of positions from all members
            %}
            
            positions = zeros(length(population), obj.dim);
            for i = 1:length(population)
                positions(i, :) = population{i}.position;
            end
        end
        
        function fitness_values = get_fitness(~, population)
            %{
            get_fitness - Extract fitness values from population
            
            Inputs:
                population : cell array of Member objects
                    Population to extract fitness values from
                    
            Returns:
                fitness_values : array
                    Array of fitness values from all members
            %}
            
            fitness_values = zeros(1, length(population));
            for i = 1:length(population)
                fitness_values(i) = population{i}.fitness;
            end
        end
        
        function result = is_better(obj, member_1, member_2)
            %{
            is_better - Compare two members to determine which is better based on optimization direction
            
            Inputs:
                member_1 : Member
                    First member to compare
                member_2 : Member
                    Second member to compare
                    
            Returns:
                result : bool
                    true if member_1 is better than member_2 according to optimization direction
            %}
            
            if obj.maximize
                result = member_1.gt(member_2);
            else
                result = member_1.lt(member_2);
            end
        end
        
        function population = init_population(obj, search_agents_no)
            %{
            init_population - Initialize a population of members with random positions
            
            Inputs:
                search_agents_no : int
                    Number of members to initialize
                    
            Returns:
                population : cell array
                    Cell array of initialized members with random positions and evaluated fitness
            %}
            
            population = cell(1, search_agents_no);
            for i = 1:search_agents_no
                position = rand(1, obj.dim) .* (obj.ub - obj.lb) + obj.lb;
                fitness = obj.objective_func(position);
                population{i} = Member(position, fitness);
            end
        end
        
        function callbacks(~, iter, max_iter, best)
            %{
            callbacks - Update progress tracking during optimization
            
            Inputs:
                iter : int
                    Current iteration number
                max_iter : int
                    Maximum number of iterations
                best : Member
                    Current best member
            %}
            
            % Update progress information (placeholder for actual progress tracking)
            % In MATLAB, we would typically use waitbar or custom progress display
            fprintf('Iteration: %d/%d, Best Fitness: %.6f\n', iter, max_iter, best.fitness);
        end
        
        function begin_step_solver(obj, max_iter)
            %{
            begin_step_solver - Initialize solver execution and display startup information
            
            Inputs:
                max_iter : int
                    Maximum number of iterations for the solver
            %}
            
            % Print algorithm start message with parameters
            fprintf('%s\n', repmat('-', 1, 50));
            fprintf('🚀 Starting %s algorithm\n', obj.name_solver);
            fprintf('📊 Parameters:\n');
            fprintf('   - Problem dimension: %d\n', obj.dim);
            
            lb_str = ['[', sprintf('%.3f ', obj.lb(:).'), ']'];
            lb_str = regexprep(lb_str, '\s+\]', ']'); 

            fprintf('   - Lower bounds: [%s]\n', strjoin(string(num2str(lb_str,'%.4g')),' '));

            ub_str = ['[', sprintf('%.3f ', obj.ub(:).'), ']'];
            ub_str = regexprep(ub_str, '\s+\]', ']'); 

            fprintf('   - Upper bounds: [%s]\n', strjoin(string(num2str(ub_str,'%.4g')),' '));
            
            fprintf('   - Optimization direction: %s\n', ternary(obj.maximize, 'Maximize', 'Minimize'));
            fprintf('   - Maximum iterations: %d\n', max_iter);
            fprintf('\n');
            
            % Initialize progress tracking
            fprintf('Starting optimization...\n');
        end
        
        function end_step_solver(obj)
            %{
            end_step_solver - Finalize solver execution and display results
            %}
            
            % Print algorithm completion message with results
            fprintf('\n');
            fprintf('✅ %s algorithm completed!\n', obj.name_solver);
            fprintf('🏆 Best solution found:\n');
            fprintf('   - Position: %s\n', mat2str(obj.best_solver.position));
            fprintf('   - Fitness: %.6f\n', obj.best_solver.fitness);
            fprintf('%s\n', repmat('-', 1, 50));
            
            if obj.show_chart
                obj.plot_history_step_solver();
            end
        end
        
        function plot_history_step_solver(obj)
            %{
            plot_history_step_solver - Plot the optimization history showing convergence over iterations
            
            Displays a line plot of the best fitness value found at each iteration.
            %}
            
            if isempty(obj.history_step_solver)
                fprintf('No optimization history available. Run the solver first.\n');
                return;
            end
            
            % Extract fitness values from history
            fitness_history = zeros(1, length(obj.history_step_solver));
            for i = 1:length(obj.history_step_solver)
                fitness_history(i) = obj.history_step_solver{i}.fitness;
            end
            
            iterations = 1:length(fitness_history);
            
            figure;
            plot(iterations, fitness_history, 'b-', 'LineWidth', 2);
            xlabel('Iteration');
            ylabel('Best Fitness');
            title('Optimization History');
            grid on;
            
            % Add horizontal line showing the best fitness achieved
            if obj.maximize
                best_fitness = max(fitness_history);
                line([0, length(fitness_history)], [best_fitness, best_fitness], ...
                     'Color', 'r', 'LineStyle', '--', ...
                     'DisplayName', sprintf('Max Fitness: %.6f', best_fitness));
            else
                best_fitness = min(fitness_history);
                line([0, length(fitness_history)], [best_fitness, best_fitness], ...
                     'Color', 'r', 'LineStyle', '--', ...
                     'DisplayName', sprintf('Min Fitness: %.6f', best_fitness));
            end
            
            legend('show');
        end
        
        function [history, best] = solver(obj)
            %{
            solver - Get the optimization results
            
            Returns:
                history : cell array
                    Cell array of best solutions at each iteration
                best : Member
                    Best solution found overall
            %}
            
            history = obj.history_step_solver;
            best = obj.best_solver;
        end
    end
end

% Helper function for ternary operation
function result = ternary(condition, true_val, false_val)
    %{
    ternary - Ternary operator helper function
    
    Inputs:
        condition : bool
            Condition to evaluate
        true_val : any
            Value to return if condition is true
        false_val : any
            Value to return if condition is false
            
    Returns:
        result : any
            true_val if condition is true, false_val otherwise
    %}
    
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
