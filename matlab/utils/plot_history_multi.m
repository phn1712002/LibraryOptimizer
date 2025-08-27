function plot_history_multi(history)
    % plot_history_multi - Plot all multi-objective solutions from history
    %
    % INPUT:
    %   history : cell array, where each element is an array of MultiObjectiveMember
    %
    % The function plots the multi-objective fitness values of all generations
    % in a single static plot (supports 2 or 3 objectives).

    % Number of generations (steps)
    numSteps = numel(history);

    % Find a valid sample để biết số objective
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

    % Số objective
    M = numel(sample{:}.multi_fitness);

    % Tạo figure
    figure("Name", "History Plot"); hold on; grid on;
    cmap = jet(numSteps); % Gradient màu theo generation

    if M == 2
        xlabel('Objective f1'); ylabel('Objective f2');
    elseif M == 3
        xlabel('Objective f1'); ylabel('Objective f2'); zlabel('Objective f3');
    else
        error('Only 2 or 3 objectives are supported for plotting.');
    end

    % Vẽ tất cả generations
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

    if M == 3
        view(45,30);
    end

    title('All Generations');
    colormap(cmap);
    cb = colorbar; 
    cb.Label.String = 'Generation';
    caxis([1 numSteps]); % đánh số thế hệ trên colorbar
    hold off;
end
