classdef AntColonyOptimizer < Solver
    %{
    Ant Colony Optimization for Continuous Domains (ACOR).
    
    ACOR is a population-based metaheuristic algorithm inspired by the foraging
    behavior of ants. It uses a solution archive and Gaussian sampling to
    explore the search space.
    
    References:
        Socha, K., & Dorigo, M. (2008). Ant colony optimization for continuous domains.
