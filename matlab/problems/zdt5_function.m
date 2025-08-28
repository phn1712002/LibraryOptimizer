function fitness = zdt5_function(x)
    % ZDT5 test function (modified, 3 objectives, real-coded version)
    % 
    % f1 = 1 + sin(pi * x1)^2
    % f2 = g * (1 - sqrt(f1/g))
    % f3 = h * (1 - (f1/g)^2)
    %
    % where:
    %   g = 1 + 9 * sum(x2...xn)/(n-1)
    %   h = 1 + sum(sin(2*pi*xi))/(n-1), i=2..n
    %
    % Input:
    %   x : vector of decision variables (real-valued, normalized [0,1])
    %
    % Output:
    %   fitness : [f1, f2, f3]

    n = length(x);

    % f1: thay thế cho binary ones-count
    f1 = 1 + (sin(pi * x(1)))^2;

    % g(x)
    g = 1 + 9 * sum(x(2:end)) / (n - 1);

    % f2
    f2 = g * (1 - sqrt(f1 / g));

    % h(x) cho f3
    h = 1 + sum(sin(2 * pi * x(2:end))) / (n - 1);

    % f3
    f3 = h * (1 - (f1 / g)^2);

    % output vector
    fitness = [f1, f2, f3];
end
