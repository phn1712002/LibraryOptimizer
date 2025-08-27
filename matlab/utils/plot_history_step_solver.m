function plot_history_step_solver(history_step_solver)
    %{
    plot_history_step_solver - Plot the optimization history showing convergence over iterations
    
    Displays a line plot of the best fitness value found at each iteration.
    %}
    
    if isempty(history_step_solver)
        fprintf('No optimization history available. Run the solver first.\n');
        return;
    end
    
    % Extract fitness values from history
    fitness_history = zeros(1, length(history_step_solver));
    for i = 1:length(history_step_solver)
        fitness_history(i) = history_step_solver{i}.fitness;
    end
    
    iterations = 1:length(fitness_history);
    
    figure('Name', 'History best fitness');
    plot(iterations, fitness_history, 'b-', 'LineWidth', 2, 'DisplayName', 'Fitness');
    xlabel('Iteration');
    ylabel('Best Fitness');
    title('Optimization History');
    grid on;
    legend('show');
end