function plot_history_animation(history)
    % plotHistoryAnimation - Animate the evolution of multi-objective solutions
    %
    % INPUT:
    %   history : cell array, where each element is an array of MultiObjectiveMember
    %
    % The function plots the multi-objective fitness values over generations
    % as an animation. Supports 2 or 3 objectives only.

    % Number of generations (steps)
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
    figure;
    hold on;
    grid on;
    if M == 2
        xlabel('Objective f1'); ylabel('Objective f2');
    elseif M == 3
        xlabel('Objective f1'); ylabel('Objective f2'); zlabel('Objective f3');
    else
        error('Only 2 or 3 objectives are supported for plotting.');
    end

    % Animation loop
    for k = 1:numSteps
        cla; % clear previous frame
        members = history{k};

        if isempty(members)
            title(sprintf('Generation %d/%d (empty)', k, numSteps));
            drawnow;
            pause(0.3);
            continue;
        end

        % Collect fitness values into matrix [M x N]
        F = cell2mat(arrayfun(@(m) m{:}.multi_fitness(:), members, 'UniformOutput', false));

        if M == 2
            plot(F(1,:), F(2,:), 'o', 'MarkerFaceColor', 'b');
        elseif M == 3
            plot3(F(1,:), F(2,:), F(3,:), 'o', 'MarkerFaceColor', 'b');
            view(45,30);
        end

        title(sprintf('Generation %d/%d', k, numSteps));
        drawnow;
        pause(0.3); % pause for animation effect
    end
end
