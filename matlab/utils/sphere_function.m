function y = sphere_function(x)
    if isrow(x)
        x = x(:)';  % Đảm bảo x là hàng
    end
    y = sum(x.^2, 2);
end