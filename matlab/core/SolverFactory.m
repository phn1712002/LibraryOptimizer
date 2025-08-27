classdef SolverFactory
    % SolverFactory - A factory class for creating optimization solvers
    %
    % This class provides functionality to:
    % - Automatically detect and instantiate single or multi-objective solvers
    % - List available solvers by type
    % - Retrieve registered solver classes by name

    methods (Static)
        function solver_instance = create_solver(solver_name, objective_func, lb, ub, dim, maximize, varargin)
            %CREATE_SOLVER Create a solver instance with automatic detection of objective function type
            %
            %   Automatically detects whether to use single-objective or multi-objective version
            %   based on objective function output for algorithms that have both versions.
            %
            %   Inputs:
            %       solver_name (char): Name of the solver to create
            %       objective_func (function_handle): Objective function to optimize
            %       lb (double or array): Lower bounds of search space
            %       ub (double or array): Upper bounds of search space  
            %       dim (int): Number of dimensions in the problem
            %       maximize (logical): Whether to maximize (true) or minimize (false) objective
            %       varargin: Additional solver parameters as name-value pairs
            %
            %   Returns:
            %       solver_instance: Instance of the appropriate solver class
            %
            %   Example:
            %       solver = SolverFactory.create_solver('WhaleOptimizer', @zdt1_function, [0 0], [1 1], 2, false);

            solver_name = strtrim(solver_name);

            % Get registry and mapping
            [SOLVER_REGISTRY, MULTI_OBJECTIVE_MAPPING] = get_solver_registry();

            % Check if this solver has a multi-objective counterpart
            if isKey(MULTI_OBJECTIVE_MAPPING, solver_name)
                try
                    % Create a test point within bounds
                    if isscalar(lb)
                        test_lb = repmat(lb, 1, dim);
                    else
                        test_lb = lb;
                    end
                    if isscalar(ub)
                        test_ub = repmat(ub, 1, dim);
                    else
                        test_ub = ub;
                    end

                    test_point = test_lb + (test_ub - test_lb) .* rand(1, dim);
                    result = objective_func(test_point);

                    % Check if result is a vector (multi-objective) or scalar (single-objective)
                    if numel(result) > 1
                        % Multi-objective function detected
                        n_objectives = numel(result);
                        multi_solver_name = MULTI_OBJECTIVE_MAPPING(solver_name);
                        fprintf('Detected multi-objective function with %d objectives. Using %s.\n', n_objectives, multi_solver_name);

                        if isKey(SOLVER_REGISTRY, multi_solver_name)
                            solver_constructor = SOLVER_REGISTRY(multi_solver_name);
                            solver_instance = solver_constructor(objective_func, lb, ub, dim, maximize, varargin{:});
                        else
                            error('Solver ''%s'' not found in registry.', multi_solver_name);
                        end
                    else
                        % Single-objective function detected
                        fprintf('Detected single-objective function. Using %s.\n', solver_name);

                        if isKey(SOLVER_REGISTRY, solver_name)
                            solver_constructor = SOLVER_REGISTRY(solver_name);
                            solver_instance = solver_constructor(objective_func, lb, ub, dim, maximize, varargin{:});
                        else
                            error('Solver ''%s'' not found in registry.', solver_name);
                        end
                    end

                catch e
                    fprintf('Warning: Could not auto-detect objective function type: %s\n', e.message);
                    fprintf('Falling back to single-objective %s.\n', solver_name);

                    if isKey(SOLVER_REGISTRY, solver_name)
                        solver_constructor = SOLVER_REGISTRY(solver_name);
                        solver_instance = solver_constructor(objective_func, lb, ub, dim, maximize, varargin{:});
                    else
                        error('Solver ''%s'' not found in registry.', solver_name);
                    end
                end

            else
                % For solvers without multi-objective counterparts, use the standard approach
                if isKey(SOLVER_REGISTRY, solver_name)
                    solver_constructor = SOLVER_REGISTRY(solver_name);
                    solver_instance = solver_constructor(objective_func, lb, ub, dim, maximize, varargin{:});
                else
                    error('Solver ''%s'' not found in registry.', solver_name);
                end
            end
        end

        function solver_class = find_solver(solver_name)
            %FIND_SOLVER Find and return a solver class by its registered name
            %
            %   Inputs:
            %       solver_name (char): Name of the solver to find (case-sensitive)
            %
            %   Returns:
            %       solver_class: The solver class corresponding to the given name

            solver_name = strtrim(solver_name);
            [SOLVER_REGISTRY, ~] = get_solver_registry();
            
            if ~isKey(SOLVER_REGISTRY, solver_name)
                available_solvers = keys(SOLVER_REGISTRY);
                error('Solver ''%s'' not found. Available solvers: %s', ...
                    solver_name, strjoin(available_solvers, ', '));
            end

            solver_class = SOLVER_REGISTRY(solver_name);
        end

        function show_solvers(mode)
            %SHOW_SOLVERS Display list of solvers by mode
            %
            %   Inputs:
            %       mode (char): 
            %           - 'single': Show only single-objective solvers.
            %           - 'multi': Show only multi-objective solvers.
            %           - 'all': Show all solvers (default).

            if nargin < 1
                mode = 'all';
            end

            mode = lower(strtrim(mode));

            if ~ismember(mode, {'single', 'multi', 'all'})
                error('mode must be one of: ''single'', ''multi'', ''all''');
            end

            [SOLVER_REGISTRY, ~] = get_solver_registry();
            solver_names = keys(SOLVER_REGISTRY);

            single_solvers = {};
            multi_solvers = {};

            % Categorize solvers
            for i = 1:length(solver_names)
                name = solver_names{i};
                if startsWith(name, 'MultiObjective')
                    multi_solvers{end+1} = name;
                else
                    single_solvers{end+1} = name;
                end
            end

            single_solvers = sort(single_solvers);
            multi_solvers = sort(multi_solvers);
            fprintf('%s\n', repmat('-',1,50));
            if strcmp(mode, 'single')
                fprintf('Single-objective Solvers:\n');
                for i = 1:length(single_solvers)
                    fprintf('  - %s\n', single_solvers{i});
                end
            elseif strcmp(mode, 'multi')
                fprintf('Multi-objective Solvers:\n');
                for i = 1:length(multi_solvers)
                    fprintf('  - %s\n', multi_solvers{i});
                end
            else  % all
                fprintf('All Solvers:\n');
                fprintf('Single-objective:\n');
                for i = 1:length(single_solvers)
                    fprintf('  - %s\n', single_solvers{i});
                end
                fprintf('\nMulti-objective:\n');
                for i = 1:length(multi_solvers)
                    fprintf('  - %s\n', multi_solvers{i});
                end
            end
            fprintf('%s\n', repmat('-',1,50));
        end
    end
end

function [SOLVER_REGISTRY, MULTI_OBJECTIVE_MAPPING] = get_solver_registry()
    %GET_SOLVER_REGISTRY Initialize and return solver registry and mapping
    SOLVER_REGISTRY = containers.Map();
    MULTI_OBJECTIVE_MAPPING = containers.Map();
    
    % Register solvers
    % Multi Objective
    SOLVER_REGISTRY('MultiObjectiveGreyWolfOptimizer') = @(varargin) MultiObjectiveGreyWolfOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveParticleSwarmOptimizer') = @(varargin) MultiObjectiveParticleSwarmOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveFireflyOptimizer') = @(varargin) MultiObjectiveFireflyOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveHarmonySearchOptimizer') = @(varargin) MultiObjectiveHarmonySearchOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveArtificialBeeColonyOptimizer') = @(varargin) MultiObjectiveArtificialBeeColonyOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveWhaleOptimizer') = @(varargin) MultiObjectiveWhaleOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveBatOptimizer') = @(varargin) MultiObjectiveBatOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveCuckooSearchOptimizer') = @(varargin) MultiObjectiveCuckooSearchOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveAntColonyOptimizer') = @(varargin) MultiObjectiveAntColonyOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveArtificialEcosystemOptimizer') = @(varargin) MultiObjectiveArtificialEcosystemOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveDingoOptimizer') = @(varargin) MultiObjectiveDingoOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveModifiedSocialGroupOptimizer') = @(varargin) MultiObjectiveModifiedSocialGroupOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveShuffledFrogLeapingOptimizer') = @(varargin) MultiObjectiveShuffledFrogLeapingOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveGeneticAlgorithmOptimizer') = @(varargin) MultiObjectiveGeneticAlgorithmOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveTeachingLearningBasedOptimizer') = @(varargin) MultiObjectiveTeachingLearningBasedOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectivePrairieDogsOptimizer') = @(varargin) MultiObjectivePrairieDogsOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveMossGrowthOptimizer') = @(varargin) MultiObjectiveMossGrowthOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveSimulatedAnnealingOptimizer') = @(varargin) MultiObjectiveSimulatedAnnealingOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveGravitationalSearchOptimizer') = @(varargin) MultiObjectiveGravitationalSearchOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveBiogeographyBasedOptimizer') = @(varargin) MultiObjectiveBiogeographyBasedOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveSeagullOptimizer') = @(varargin) MultiObjectiveSeagullOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveGlowwormSwarmOptimizer') = @(varargin) MultiObjectiveGlowwormSwarmOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveElectromagneticChargedParticlesOptimizer') = @(varargin) MultiObjectiveElectromagneticChargedParticlesOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveHenryGasSolubilityOptimizer') = @(varargin) MultiObjectiveHenryGasSolubilityOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveMarinePredatorsOptimizer') = @(varargin) MultiObjectiveMarinePredatorsOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveEquilibriumOptimizer') = @(varargin) MultiObjectiveEquilibriumOptimizer(varargin{:});
    SOLVER_REGISTRY('MultiObjectiveJAYAOptimizer') = @(varargin) MultiObjectiveJAYAOptimizer(varargin{:});
    
    % Signle Objective
    SOLVER_REGISTRY('GreyWolfOptimizer') = @(varargin) GreyWolfOptimizer(varargin{:});
    SOLVER_REGISTRY('ArtificialBeeColonyOptimizer') = @(varargin) ArtificialBeeColonyOptimizer(varargin{:});
    SOLVER_REGISTRY('ParticleSwarmOptimizer') = @(varargin) ParticleSwarmOptimizer(varargin{:});
    SOLVER_REGISTRY('WhaleOptimizer') = @(varargin) WhaleOptimizer(varargin{:});
    SOLVER_REGISTRY('FireflyOptimizer') = @(varargin) FireflyOptimizer(varargin{:});
    SOLVER_REGISTRY('HarmonySearchOptimizer') = @(varargin) HarmonySearchOptimizer(varargin{:});
    SOLVER_REGISTRY('BatOptimizer') = @(varargin) BatOptimizer(varargin{:});
    SOLVER_REGISTRY('CuckooSearchOptimizer') = @(varargin) CuckooSearchOptimizer(varargin{:});
    SOLVER_REGISTRY('AntColonyOptimizer') = @(varargin) AntColonyOptimizer(varargin{:});
    SOLVER_REGISTRY('ArtificialEcosystemOptimizer') = @(varargin) ArtificialEcosystemOptimizer(varargin{:});
    SOLVER_REGISTRY('BiogeographyBasedOptimizer') = @(varargin) BiogeographyBasedOptimizer(varargin{:});
    SOLVER_REGISTRY('DingoOptimizer') = @(varargin) DingoOptimizer(varargin{:});
    SOLVER_REGISTRY('ElectromagneticChargedParticlesOptimizer') = @(varargin) ElectromagneticChargedParticlesOptimizer(varargin{:});
    SOLVER_REGISTRY('EquilibriumOptimizer') = @(varargin) EquilibriumOptimizer(varargin{:});
    SOLVER_REGISTRY('GeneticAlgorithmOptimizer') = @(varargin) GeneticAlgorithmOptimizer(varargin{:});
    SOLVER_REGISTRY('GlowwormSwarmOptimizer') = @(varargin) GlowwormSwarmOptimizer(varargin{:});
    SOLVER_REGISTRY('GravitationalSearchOptimizer') = @(varargin) GravitationalSearchOptimizer(varargin{:});
    SOLVER_REGISTRY('HenryGasSolubilityOptimizer') = @(varargin) HenryGasSolubilityOptimizer(varargin{:});
    SOLVER_REGISTRY('JAYAOptimizer') = @(varargin) JAYAOptimizer(varargin{:});
    SOLVER_REGISTRY('MarinePredatorsOptimizer') = @(varargin) GlowwormSwarmOptimizer(varargin{:});
    SOLVER_REGISTRY('ModifiedSocialGroupOptimizer') = @(varargin) ModifiedSocialGroupOptimizer(varargin{:});
    SOLVER_REGISTRY('MossGrowthOptimizer') = @(varargin) MossGrowthOptimizer(varargin{:});
    SOLVER_REGISTRY('SeagullOptimizer') = @(varargin) SeagullOptimizer(varargin{:});
    SOLVER_REGISTRY('PrairieDogsOptimizer') = @(varargin) PrairieDogsOptimizer(varargin{:});
    SOLVER_REGISTRY('ShuffledFrogLeapingOptimizer') = @(varargin) ShuffledFrogLeapingOptimizer(varargin{:});
    SOLVER_REGISTRY('SimulatedAnnealingOptimizer') = @(varargin) SimulatedAnnealingOptimizer(varargin{:});
    SOLVER_REGISTRY('TeachingLearningBasedOptimizer') = @(varargin) TeachingLearningBasedOptimizer(varargin{:});

    % Add multi-objective mappings
    MULTI_OBJECTIVE_MAPPING('GreyWolfOptimizer') = 'MultiObjectiveGreyWolfOptimizer';
    MULTI_OBJECTIVE_MAPPING('ParticleSwarmOptimizer') = 'MultiObjectiveParticleSwarmOptimizer';
    MULTI_OBJECTIVE_MAPPING('FireflyOptimizer') = 'MultiObjectiveFireflyOptimizer';
    MULTI_OBJECTIVE_MAPPING('HarmonySearchOptimizer') = 'MultiObjectiveHarmonySearchOptimizer';
    MULTI_OBJECTIVE_MAPPING('ArtificialBeeColonyOptimizer') = 'MultiObjectiveArtificialBeeColonyOptimizer';
    MULTI_OBJECTIVE_MAPPING('WhaleOptimizer') = 'MultiObjectiveWhaleOptimizer';
    MULTI_OBJECTIVE_MAPPING('BatOptimizer') = 'MultiObjectiveBatOptimizer';
    MULTI_OBJECTIVE_MAPPING('CuckooSearchOptimizer') = 'MultiObjectiveCuckooSearchOptimizer';
    MULTI_OBJECTIVE_MAPPING('AntColonyOptimizer') = 'MultiObjectiveAntColonyOptimizer';
    MULTI_OBJECTIVE_MAPPING('ArtificialEcosystemOptimizer') = 'MultiObjectiveArtificialEcosystemOptimizer';
    MULTI_OBJECTIVE_MAPPING('DingoOptimizer') = 'MultiObjectiveDingoOptimizer';
    MULTI_OBJECTIVE_MAPPING('ModifiedSocialGroupOptimizer') = 'MultiObjectiveModifiedSocialGroupOptimizer';
    MULTI_OBJECTIVE_MAPPING('ShuffledFrogLeapingOptimizer') = 'MultiObjectiveShuffledFrogLeapingOptimizer';
    MULTI_OBJECTIVE_MAPPING('GeneticAlgorithmOptimizer') = 'MultiObjectiveGeneticAlgorithmOptimizer';
    MULTI_OBJECTIVE_MAPPING('TeachingLearningBasedOptimizer') = 'MultiObjectiveTeachingLearningBasedOptimizer';
    MULTI_OBJECTIVE_MAPPING('PrairieDogsOptimizer') = 'MultiObjectivePrairieDogsOptimizer';
    MULTI_OBJECTIVE_MAPPING('MossGrowthOptimizer') = 'MultiObjectiveMossGrowthOptimizer';
    MULTI_OBJECTIVE_MAPPING('SimulatedAnnealingOptimizer') = 'MultiObjectiveSimulatedAnnealingOptimizer';
    MULTI_OBJECTIVE_MAPPING('GravitationalSearchOptimizer') = 'MultiObjectiveGravitationalSearchOptimizer';
    MULTI_OBJECTIVE_MAPPING('BiogeographyBasedOptimizer') = 'MultiObjectiveBiogeographyBasedOptimizer';
    MULTI_OBJECTIVE_MAPPING('SeagullOptimizer') = 'MultiObjectiveSeagullOptimizer';
    MULTI_OBJECTIVE_MAPPING('GlowwormSwarmOptimizer') = 'MultiObjectiveGlowwormSwarmOptimizer';
    MULTI_OBJECTIVE_MAPPING('ElectromagneticChargedParticlesOptimizer') = 'MultiObjectiveElectromagneticChargedParticlesOptimizer';
    MULTI_OBJECTIVE_MAPPING('HenryGasSolubilityOptimizer') = 'MultiObjectiveHenryGasSolubilityOptimizer';
    MULTI_OBJECTIVE_MAPPING('MarinePredatorsOptimizer') = 'MultiObjectiveMarinePredatorsOptimizer';
    MULTI_OBJECTIVE_MAPPING('EquilibriumOptimizer') = 'MultiObjectiveEquilibriumOptimizer';
    MULTI_OBJECTIVE_MAPPING('JAYAOptimizer') = 'MultiObjectiveJAYAOptimizer';
    
    % Add more solvers here as they are implemented
    % Example: SOLVER_REGISTRY('NSGA2') = @(varargin) NSGA2(varargin{:});
end
