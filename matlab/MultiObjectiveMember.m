classdef MultiObjectiveMember < handle
    % A particle/individual for multi-objective optimization
    properties
        position        % 1 x dim double
        multi_fitness   % 1 x M double (M = số mục tiêu)
        dominated logical = false
        grid_index      % scalar linear index của hypercube
        grid_sub_index  % 1 x M chỉ số con theo từng mục tiêu
    end

    methods
        function obj = MultiObjectiveMember(position, fitness)
            if nargin > 0
                obj.position      = position(:).';      % đảm bảo dạng hàng
                obj.multi_fitness = fitness(:).';       % đảm bảo dạng hàng
            end
        end

        function newm = copy(obj)
            newm = MultiObjectiveMember(obj.position, obj.multi_fitness);
            newm.dominated     = obj.dominated;
            newm.grid_index    = obj.grid_index;
            newm.grid_sub_index= obj.grid_sub_index;
        end

        function disp(obj)
            fprintf('Position: [%s] - Fitness: [%s] - Dominated: %d\n', ...
                strjoin(string(num2str(obj.position(:), '%.6g')),' '), ...
                strjoin(string(num2str(obj.multi_fitness(:), '%.6g')),' '), ...
                obj.dominated);
        end
    end
end
