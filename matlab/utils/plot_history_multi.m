function plot_history_multi(history)
    % plot_history_multi - Plot all multi-objective solutions from history
    %
    % INPUT:
    %   history : cell array, where each element is an array of MultiObjectiveMember
    %
    % The function plots the multi-objective fitness values of all iterations
    % in a single static plot (supports 2 or 3 objectives).

    % Number of iterations (steps)
    numSteps = numel(history);

    % Find a valid sample to determine the number of objectives
    sample = [];
    for k = 1:numSteps
        if ~isempty(history{k})
            sample = history{k}(1);
            break;
        end
    end
    if isempty(sample)
        error('History is empty. Nothing to plot.');
    end

    % Number of objectives
    M = numel(sample{:}.multi_fitness);

    % Create figure
    figure("Name", "History Plot"); 
    hold on; grid on;
    cmap = jet(numSteps); % Colormap gradient by iteration

    if M == 2
        xlabel('Objective f1'); ylabel('Objective f2');
    elseif M == 3
        xlabel('Objective f1'); ylabel('Objective f2'); zlabel('Objective f3');
    else
        error('Only 2 or 3 objectives are supported for plotting.');
    end

    % Plot all iterations
    for k = 1:numSteps
        members = history{k};
        if isempty(members), continue; end

        % Collect fitness values
        F = cell2mat(arrayfun(@(m) m{:}.multi_fitness(:), members, 'UniformOutput', false));

        if M == 2
            scatter(F(1,:), F(2,:), 25, repmat(cmap(k,:), size(F,2),1), ...
                'filled', 'MarkerEdgeColor','k', 'MarkerFaceAlpha',0.6);
        elseif M == 3
            scatter3(F(1,:), F(2,:), F(3,:), 25, repmat(cmap(k,:), size(F,2),1), ...
                'filled', 'MarkerEdgeColor','k', 'MarkerFaceAlpha',0.6);
        end
    end

    % 3D view
    if M == 3
        view(45,30);
    end

    title('All Iterations');
    colormap(cmap);
    cb = colorbar; 
    cb.Label.String = 'Iteration';
    caxis([1 numSteps]); % scale colorbar to number of generations
    hold off;
end
