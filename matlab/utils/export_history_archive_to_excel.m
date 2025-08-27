function export_history_archive_to_excel(history_archive, filename)
        %{
        export_history_to_excel - Xuất lịch sử archive ra file Excel
        
        Inputs:
            history_archive : cell array
                Lịch sử archive từ quá trình tối ưu hóa
            filename : string
                Tên file Excel để xuất dữ liệu
        %}
        
        % Kiểm tra xem history_archive có dữ liệu không
        if isempty(history_archive)
            warning('History archive is empty. No data to export.');
            return;
        end
        
        % Tạo bảng dữ liệu
        data = {};
        header = {'Iteration', 'Solution_Index', 'Position', 'Fitness'};
        
        % Duyệt qua từng iteration
        for iter = 1:length(history_archive)
            archive_iter = history_archive{iter};
            
            % Duyệt qua từng solution trong archive của iteration hiện tại
            for sol_idx = 1:length(archive_iter)
                solution = archive_iter{sol_idx};
                
                % Thêm dòng dữ liệu
                row_data = {
                    iter, ...
                    sol_idx, ...
                    mat2str(solution.position), ...
                    mat2str(solution.multi_fitness)
                };
                
                data = [data; row_data];
            end
        end
        
        % Chuyển thành table
        T = cell2table(data, 'VariableNames', header);
        
        % Ghi ra file Excel
        writetable(T, filename, 'Sheet', 'Archive_History');
        
        fprintf('History archive exported to: %s\n', filename);
    end