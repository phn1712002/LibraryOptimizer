function plot_2D(x, y, varargin)
% plot2D - Vẽ đồ thị 2D với chế độ line hoặc point
% Cú pháp:
%   plot2D(x, y, 'Title', 'My Plot', 'XLabel', 'X', 'YLabel', 'Y', 'Mode', 'line')
%
% Đầu vào:
%   x, y: vector dữ liệu
%   Các cặp tên-giá trị tùy chọn:
%       'Title'  - Chuỗi tiêu đề (mặc định: '')
%       'XLabel' - Nhãn trục X (mặc định: 'x')
%       'YLabel' - Nhãn trục Y (mặc định: 'y')
%       'Mode'   - 'line' hoặc 'point' (mặc định: 'line')
%       'Color'  - Màu sắc (mặc định: 'b')
%
% Ví dụ:
%   plot2D(1:10, (1:10).^2, 'Mode', 'point', 'Title', 'Parabola')

    % Thiết lập các giá trị mặc định
    p.Title = '';
    p.XLabel = 'x';
    p.YLabel = 'y';
    p.Mode = 'line';  % hoặc 'point'
    p.Color = 'b';

    % Đọc các tham số tên-giá trị
    for i = 1:2:length(varargin)
        key = upper(varargin{i});
        if i+1 <= length(varargin)
            value = varargin{i+1};
        else
            error('Thiếu giá trị cho tham số: %s', varargin{i});
        end
        
        switch key
            case 'TITLE'
                p.Title = value;
            case 'XLABEL'
                p.XLabel = value;
            case 'YLABEL'
                p.YLabel = value;
            case 'MODE'
                if ~ismember(lower(value), {'line', 'point'})
                    error('Mode phải là ''line'' hoặc ''point''');
                end
                p.Mode = lower(value);
            case 'COLOR'
                p.Color = value;
            otherwise
                warning('Bỏ qua tham số không hỗ trợ: %s', varargin{i});
        end
    end

    % Vẽ đồ thị
    figure('Name',p.Title);
    if strcmp(p.Mode, 'line')
        plot(x, y, 'Color', p.Color, 'LineWidth', 1.5);
    else
        scatter(x, y, 'filled', 'MarkerFaceColor', p.Color);
    end

    % Thêm nhãn và tiêu đề
    title(p.Title, 'FontSize', 12);
    xlabel(p.XLabel);
    ylabel(p.YLabel);
    grid on;

end