classdef MultiObjectiveSolver
    % Multi-objective optimizer with Pareto archive & grid-based selection
    properties
        % Problem
        objective_func   % function handle: f(x) -> 1xM (row vector)
        lb               % 1xD or scalar
        ub               % 1xD or scalar
        dim              % D
        maximize logical = true

        % MO params
        n_objectives
        archive_size = 100
        archive

        % Grid selection
        alpha_grid = 0.1
        n_grid = 7
        beta_leader = 2
        gamma_archive = 2
        grid % cell array {1..M} of struct('lower',vec,'upper',vec)

        % Misc
        show_chart logical = true
        name_solver = 'MultiObjectiveSolver'
        kwargs struct = struct()

        best_solver = []

        % History
        history_step_solver = {}
    end

    methods
        function obj = MultiObjectiveSolver(objective_func, lb, ub, dim, maximize, varargin)
            if nargin < 5, maximize = true; end
            obj.objective_func = objective_func;
            obj.lb = obj.expand_bounds(lb, dim);
            obj.ub = obj.expand_bounds(ub, dim);
            obj.dim = dim;
            obj.maximize = maximize;

            % kwargs (Name-Value pairs)
            if ~isempty(varargin)
                obj.kwargs = obj.nv2struct(varargin{:});
            end
            obj.show_chart   = obj.get_kw('show_chart', true);
            obj.archive_size = obj.get_kw('archive_size', 100);
            obj.alpha_grid   = obj.get_kw('alpha_grid', 0.1);
            obj.n_grid       = obj.get_kw('n_grid', 7);
            obj.beta_leader  = obj.get_kw('beta', 2);
            obj.gamma_archive= obj.get_kw('gamma_archive', 2);

            % Suy ra số mục tiêu từ một đánh giá ngẫu nhiên
            probe = obj.objective_func(obj.lb + (obj.ub-obj.lb).*rand(1, obj.dim));
            probe = probe(:).';
            obj.n_objectives = numel(probe);
        end

        %================ Core helpers ================%
        function flag = dominates(obj, x, y)
            xf = x.multi_fitness; yf = y.multi_fitness;
            if obj.maximize
                not_worse = all(xf >= yf);
                better    = any(xf >  yf);
            else
                not_worse = all(xf <= yf);
                better    = any(xf <  yf);
            end
            flag = not_worse && better;
        end

        function population = determine_domination(obj, population)
            n = numel(population);
            for i = 1:n
                population(i).dominated = false;
                for j = 1:n
                    if i~=j && ~population(j).dominated
                        if obj.dominates(population(j), population(i))
                            population(i).dominated = true;
                            break;
                        end
                    end
                end
            end
        end

        function nd = get_non_dominated_particles(~, population)
            nd = population(~[population.dominated]);
        end

        function C = get_fitness(~, population)
            if isempty(population)
                C = [];
            else
                F = cellfun(@(v) v(:).', {population.multi_fitness}, 'UniformOutput', false);
                C = cat(1, F{:}).'; % M x N
            end
        end

        function pop = get_random_population(~, population, sz)
            if isempty(population) || sz <= 0
                pop = MultiObjectiveMember.empty;
                return;
            end
            if sz >= numel(population)
                pop = arrayfun(@(p) p.copy(), population);
                return;
            end
            idx = randperm(numel(population), sz);
            pop = arrayfun(@(k) population(k).copy(), idx);
        end

        function grid = create_hypercubes(obj, costs)
            % costs: M x N
            if isempty(costs), grid = {}; return; end
            M = size(costs,1);
            grid = cell(1,M);
            for j = 1:M
                mincj = min(costs(j,:));
                maxcj = max(costs(j,:));
                dcj = obj.alpha_grid * (maxcj - mincj);
                mincj = mincj - dcj;
                maxcj = maxcj + dcj;
                gx = linspace(mincj, maxcj, obj.n_grid-1);
                grid{j} = struct( ...
                    'lower', [-inf, gx], ...
                    'upper', [gx, inf]  ...
                );
            end
        end

        function [linIdx, subIdx] = get_grid_index(obj, particle)
            if isempty(obj.grid)
                linIdx = []; subIdx = [];
                return;
            end
            c = particle.multi_fitness(:).';
            M = numel(c);
            subIdx = zeros(1,M);
            dims = zeros(1,M);
            for j = 1:M
                Ub = obj.grid{j}.upper;
                k = find(c(j) < Ub, 1, 'first'); % 1-based
                if isempty(k), k = numel(Ub); end
                subIdx(j) = k;
                dims(j) = numel(Ub);
            end
            linIdx = obj.sub2ind_vec(dims, subIdx);
        end

        function leader = select_leader(obj)
            if isempty(obj.archive)
                leader = [];
                return;
            end
            gridIdx = [obj.archive.grid_index];
            gridIdx = gridIdx(~arrayfun(@isempty, {obj.archive.grid_index}));
            if isempty(gridIdx)
                leader = obj.archive(randi(numel(obj.archive)));
                return;
            end
            [cells,~,ic] = unique(gridIdx);
            counts = accumarray(ic, 1);
            probs = exp(-obj.beta_leader .* counts);
            probs = probs./sum(probs);
            cellChosen = obj.roulette_select(cells, probs);
            members = obj.archive([obj.archive.grid_index] == cellChosen);
            leader = members(randi(numel(members)));
        end

        function leaders = select_multiple_leaders(obj, n_leaders)
            leaders = MultiObjectiveMember.empty;
            if isempty(obj.archive) || n_leaders <= 0, return; end

            gridIdx = [obj.archive.grid_index];
            gridIdx = gridIdx(~arrayfun(@isempty, {obj.archive.grid_index}));

            if isempty(gridIdx)
                n_avail = min(n_leaders, numel(obj.archive));
                leaders = obj.archive(randperm(numel(obj.archive), n_avail));
                return;
            end

            [cells, ~, ic] = unique(gridIdx);
            counts = accumarray(ic, 1);
            if n_leaders > numel(cells)
                % lấy mỗi ô 1 leader trước
                for k = 1:numel(cells)
                    mem = obj.archive([obj.archive.grid_index] == cells(k));
                    leaders(end+1) = mem(randi(numel(mem))); %#ok<AGROW>
                end
                remain = n_leaders - numel(cells);
                if remain > 0
                    avail = setdiff(obj.archive, leaders);
                    if ~isempty(avail)
                        pick = min(remain, numel(avail));
                        leaders = [leaders, avail(randperm(numel(avail), pick))];
                    end
                end
                return;
            end

            probs = exp(-obj.beta_leader .* counts);
            probs = probs./sum(probs);
            chosenCells = obj.roulette_select_unique(cells, probs, n_leaders);
            for k = 1:numel(chosenCells)
                mem = obj.archive([obj.archive.grid_index] == chosenCells(k));
                leaders(end+1) = mem(randi(numel(mem))); %#ok<AGROW>
            end
        end

        function pop_sorted = sort_population(obj, population)
            if isempty(population)
                pop_sorted = MultiObjectiveMember.empty; return;
            end
            n = numel(population);
            leaders = obj.select_multiple_leaders(n);

            % Giữ lại các leader thực sự có mặt trong population (so sánh theo position)
            popPos = obj.getPositions(population);
            isValid = false(1, numel(leaders));
            for i = 1:numel(leaders)
                isValid(i) = obj.row_in_matrix(leaders(i).position, popPos);
            end
            leaders = leaders(isValid);

            % loại khỏi population những phần tử có position trùng với leader
            leadPos = obj.getPositions(leaders);
            non_leaders = population(~obj.rows_member_of(popPos, leadPos));

            % sắp thứ tự non-leaders theo random fitness (phụ thuộc hướng tối ưu)
            rf = arrayfun(@(m) obj.get_random_fitness(m), non_leaders);
            if obj.maximize
                [~, idx] = sort(rf, 'descend');
            else
                [~, idx] = sort(rf, 'ascend');
            end
            non_leaders = non_leaders(idx);

            pop_sorted = [leaders, non_leaders];
            if numel(pop_sorted) > n
                pop_sorted = pop_sorted(1:n);
            end
        end

        function val = get_random_fitness(~, member)
            if rand > 0.5
                val = mean(member.multi_fitness);
            else
                val = member.multi_fitness(randi(numel(member.multi_fitness)));
            end
        end

        function obj = add_to_archive(obj, newSolutions)
            newSolutions = obj.determine_domination(newSolutions);
            obj.archive = [obj.archive, obj.get_non_dominated_particles(newSolutions)];
            obj.archive = obj.determine_domination(obj.archive);
            obj.archive = obj.get_non_dominated_particles(obj.archive);

            C = obj.get_fitness(obj.archive);
            if ~isempty(C)
                obj.grid = obj.create_hypercubes(C);
                for k = 1:numel(obj.archive)
                    [gi, gs] = obj.get_grid_index(obj.archive(k));
                    obj.archive(k).grid_index = gi;
                    obj.archive(k).grid_sub_index = gs;
                end
            end

            if numel(obj.archive) > obj.archive_size
                obj = obj.trim_archive();
            end
        end

        function obj = trim_archive(obj)
            extra = numel(obj.archive) - obj.archive_size;
            for t = 1:extra
                gridIdx = [obj.archive.grid_index];
                gridIdx = gridIdx(~arrayfun(@isempty, {obj.archive.grid_index}));
                if isempty(gridIdx)
                    obj.archive(randi(numel(obj.archive))) = [];
                    continue;
                end
                [cells, ~, ic] = unique(gridIdx);
                counts = accumarray(ic, 1);
                probs = (counts .^ obj.gamma_archive);
                probs = probs ./ sum(probs); % ô đông bị loại nhiều hơn
                cellChosen = obj.roulette_select(cells, probs);
                mem = find([obj.archive.grid_index] == cellChosen);
                obj.archive(mem(randi(numel(mem)))) = [];
            end
        end

        function population = init_population(obj, N)
            population = repmat(MultiObjectiveMember, 1, N);
            for i = 1:N
                pos = obj.lb + (obj.ub - obj.lb).*rand(1, obj.dim);
                fit = obj.objective_func(pos); fit = fit(:).';
                population(i) = MultiObjectiveMember(pos, fit);
            end
        end

        function begin_step_solver(obj, max_iter)
            fprintf('%s\n', repmat('-',1,50));
            fprintf('🚀 Starting %s algorithm\n', obj.name_solver);
            fprintf('📊 Parameters:\n');
            fprintf('   - Objectives dimension: %d\n', obj.n_objectives);
            fprintf('   - Problem dimension: %d\n', obj.dim);

            lb_str = ['[', sprintf('%.3f ', obj.lb(:).'), ']'];
            lb_str = regexprep(lb_str, '\s+\]', ']'); 

            fprintf('   - Lower bounds: [%s]\n', strjoin(string(num2str(lb_str,'%.4g')),' '));

            ub_str = ['[', sprintf('%.3f ', obj.ub(:).'), ']'];
            ub_str = regexprep(ub_str, '\s+\]', ']'); 

            fprintf('   - Upper bounds: [%s]\n', strjoin(string(num2str(ub_str,'%.4g')),' '));
            
            fprintf('   - Optimization direction: %s\n', tern(obj.maximize,'Maximize','Minimize'));
            fprintf('   - Maximum iterations: %d\n\n', max_iter);
        end

        function obj = callbacks(obj, iter, max_iter, best)
            arch_size = numel(obj.archive);
            if ~isempty(best)
                fitness_str = ['[', sprintf('%.3f ', best.multi_fitness), ']'];
                fitness_str = regexprep(fitness_str, '\s+\]', ']'); 
            else
                fitness_str = 'N/A';
            end
            fprintf('Iter %4d/%4d | Archive: %4d | Best: %s\n', iter, max_iter, arch_size, fitness_str);
        end

        function end_step_solver(obj)
            fprintf('\n✅ %s algorithm completed!\n', obj.name_solver);
            fprintf('🏆 Archive contains %d non-dominated solutions\n', numel(obj.archive));
            if ~isempty(obj.archive)
                C = obj.get_fitness(obj.archive);
                for i = 1:size(C,1)
                    if obj.maximize
                        fprintf('Objective %d: worst=%.6f, best=%.6f\n', i, min(C(i,:)), max(C(i,:)));
                    else
                        fprintf('Objective %d: best=%.6f, worst=%.6f\n', i, min(C(i,:)), max(C(i,:)));
                    end
                end
            end
            fprintf('%s\n', repmat('-',1,50));
            if obj.show_chart && ~isempty(obj.archive) && numel(obj.archive(1).multi_fitness) >= 2
                obj.plot_pareto_front();
            end
        end

        function m = tournament_selection_multi(~, population, tournament_size)
            if numel(population) < tournament_size
                m = population(randi(numel(population))); return;
            end
            idx = randperm(numel(population), tournament_size);
            cand = population(idx);

            % ưu tiên phần tử không bị dominated; nếu nhiều -> dùng độ đa dạng ô lưới
            nd = cand(~[cand.dominated]);
            if ~isempty(nd)
                gridIdx = [nd.grid_index];
                gridIdx = gridIdx(~arrayfun(@isempty, {nd.grid_index}));
                if ~isempty(gridIdx)
                    [cells, ~, ic] = unique(gridIdx);
                    counts = accumarray(ic, 1);
                    leastCell = cells(counts == min(counts));
                    sel = nd([nd.grid_index] == leastCell(randi(numel(leastCell))));
                    m = sel(randi(numel(sel)));
                    return;
                end
                m = nd(randi(numel(nd))); return;
            end
            m = cand(randi(numel(cand)));
        end

        function obj = plot_pareto_front(obj)
            if isempty(obj.archive)
                disp('No solutions in obj.archive to plot.'); return;
            end
            C = obj.get_fitness(obj.archive); % M x N
            M = obj.n_objectives;
            if M == 2
                figure; scatter(C(1,:), C(2,:), 36, 'filled');
                xlabel('Objective 1'); ylabel('Objective 2'); title('Pareto Front (2D)'); grid on;
            elseif M == 3
                figure; plot3(C(1,:), C(2,:), C(3,:), '.', 'MarkerSize', 18);
                xlabel('Objective 1'); ylabel('Objective 2'); zlabel('Objective 3');
                title('Pareto Front (3D)'); grid on;
            else
                warning('Cannot plot Pareto front for %d objectives (only up to 3 supported).', M);
            end
        end

        function [history, archive] = solver(obj)
            history = obj.history_step_solver;
            archive = obj.archive;
        end

        function bounds = expand_bounds(~, b, dim)
            if isscalar(b), bounds = repmat(b, 1, dim);
            else, bounds = b(:).'; end
        end

        function s = get_kw(obj, name, default)
            if isfield(obj.kwargs, name), s = obj.kwargs.(name);
            else, s = default; end
        end

        function S = nv2struct(~, varargin)
            S = struct();
            if mod(numel(varargin),2) ~= 0, return; end
            for k = 1:2:numel(varargin)
                key = varargin{k};
                val = varargin{k+1};
                if ischar(key) || isstring(key)
                    S.(matlab.lang.makeValidName(char(key))) = val;
                end
            end
        end

        function pos = getPositions(~, population)
            if isempty(population)
                pos = zeros(0,0); return;
            end
            P = arrayfun(@(m) m.position(:).', population, 'UniformOutput', false);
            pos = cat(1, P{:}); % N x D
        end

        function tf = row_in_matrix(~, row, M)
            if isempty(M), tf = false; return; end
            tf = ismember(row, M, 'rows');
        end

        function mask = rows_member_of(~, A, B)
            % mask(i)=true nếu hàng A(i,:) thuộc B (dùng cho loại leaders ra khỏi population)
            if isempty(B) || isempty(A)
                mask = false(size(A,1),1); return;
            end
            mask = ismember(A, B, 'rows');
        end

        function idx = sub2ind_vec(~, dims, subs)
            % dims & subs là hàng (1xM), subs 1-based
            idx = 1;
            mult = 1;
            for k = 1:numel(dims)
                idx = idx + (subs(k)-1)*mult;
                mult = mult * dims(k);
            end
        end

        function val = roulette_select(~, labels, probs)
            cp = cumsum(probs(:)./sum(probs));
            r = rand;
            k = find(r <= cp, 1, 'first');
            if isempty(k), k = numel(cp); end
            val = labels(k);
        end

        function chosen = roulette_select_unique(obj, labels, probs, K)
            chosen = [];
            L = labels(:);
            P = probs(:)./sum(probs);
            for i = 1:K
                if isempty(L), break; end
                v = obj.roulette_select(L, P);
                chosen(end+1,1) = v; %#ok<AGROW>
                keep = L ~= v;
                L = L(keep);
                P = P(keep);
                if ~isempty(P), P = P./sum(P); end
            end
        end
    end
end

function s = tern(cond, a, b)
if cond, s = a; else, s = b; end
end
