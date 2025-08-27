% Test multi-objective function - ZDT1 benchmark problem
function fitness = zdt1_function(x)
    % ZDT1 benchmark problem (minimization)
    % x should be a vector of length 30 (standard ZDT1 dimension)
    n = length(x);
    
    % Objective 1: f1(x) = x1
    f1 = x(1);
    
    % Objective 2: f2(x) = g(x) * (1 - sqrt(x1/g(x)))
    g = 1 + 9 * sum(x(2:end)) / (n - 1);
    f2 = g * (1 - sqrt(x(1) / g));
    
    fitness = [f1, f2];
end
