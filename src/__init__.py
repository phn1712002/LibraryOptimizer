
from typing import Dict, Type, Callable, Union
import numpy as np
from ._core import *
from .greywolf_optimizer import *
from .whale_optimizer import *
from .particleswarm_optimizer import *
from .artificialbeecolony_optimizer import *
from .antcolony_optimizer import *
from .bat_optimizer import *
from .artificialecosystem_optimizer import *
from .cuckoosearch_optimizer import *
from .dingo_optimizer import *
from .firefly_optimizer import *
from .jaya_optimizer import *
from .modifiedsocialgroup_optimizer import *
from .mossgrowth_optimizer import *
from .shuffledfrogleaping_optimizer import *
from .teachinglearningbased_optimizer import *
from .prairiedogs_optimizer import *
from .simulatedannealing_optimizer import *
from .geneticalgorithm_optimizer import *

# Multi-objective optimizers
from .multiobjective.artificialbeecolony_optimizer import *
from .multiobjective.greywolf_optimizer import *
from .multiobjective.whale_optimizer import *
from .multiobjective.particleswarm_optimizer import *
from .multiobjective.antcolony_optimizer import MultiObjectiveAntColonyOptimizer
from .multiobjective.bat_optimizer import MultiObjectiveBatOptimizer
from .multiobjective.artificialecosystem_optimizer import MultiObjectiveArtificialEcosystemOptimizer
from .multiobjective.cuckoosearch_optimizer import MultiObjectiveCuckooSearchOptimizer
from .multiobjective.dingo_optimizer import MultiObjectiveDingoOptimizer
from .multiobjective.firefly_optimizer import MultiObjectiveFireflyOptimizer
from .multiobjective.modifiedsocialgroup_optimizer import MultiObjectiveModifiedSocialGroupOptimizer
from .multiobjective.shuffledfrogleaping_optimizer import MultiObjectiveShuffledFrogLeapingOptimizer
from .multiobjective.geneticalgorithm_optimizer import MultiObjectiveGeneticAlgorithmOptimizer
from .multiobjective.teachinglearningbased_optimizer import MultiObjectiveTeachingLearningBasedOptimizer

__version__ = "0.1.0"
__author__ = "HoangggNam"
__email__ = "phn1712002@gmai.com"

# Registry of available solvers
_SOLVER_REGISTRY: Dict[str, Type[Solver]] = {
    "GreyWolfOptimizer": GreyWolfOptimizer,
    "WhaleOptimizer": WhaleOptimizer,
    "ParticleSwarmOptimizer": ParticleSwarmOptimizer,
    "ArtificialBeeColonyOptimizer": ArtificialBeeColonyOptimizer,
    "AntColonyOptimizer": AntColonyOptimizer,
    "BatOptimizer": BatOptimizer,
    "ArtificialEcosystemOptimizer": ArtificialEcosystemOptimizer,
    "CuckooSearchOptimizer": CuckooSearchOptimizer,
    "DingoOptimizer": DingoOptimizer,
    "FireflyOptimizer": FireflyOptimizer,
    "JAYAOptimizer": JAYAOptimizer,
    "ModifiedSocialGroupOptimizer": ModifiedSocialGroupOptimizer,
    "MossGrowthOptimizer": MossGrowthOptimizer,
    "ShuffledFrogLeapingOptimizer": ShuffledFrogLeapingOptimizer,
    "TeachingLearningBasedOptimizer": TeachingLearningBasedOptimizer,
    "PrairieDogsOptimizer": PrairieDogsOptimizer,
    "SimulatedAnnealingOptimizer": SimulatedAnnealingOptimizer,
    "GeneticAlgorithmOptimizer": GeneticAlgorithmOptimizer,
    "MultiObjectiveArtificialBeeColonyOptimizer": MultiObjectiveArtificialBeeColonyOptimizer,
    "MultiObjectiveGreyWolfOptimizer": MultiObjectiveGreyWolfOptimizer,
    "MultiObjectiveWhaleOptimizer": MultiObjectiveWhaleOptimizer,
    "MultiObjectiveParticleSwarmOptimizer": MultiObjectiveParticleSwarmOptimizer,
    "MultiObjectiveAntColonyOptimizer": MultiObjectiveAntColonyOptimizer,
    "MultiObjectiveBatOptimizer": MultiObjectiveBatOptimizer,
    "MultiObjectiveArtificialEcosystemOptimizer": MultiObjectiveArtificialEcosystemOptimizer,
    "MultiObjectiveCuckooSearchOptimizer": MultiObjectiveCuckooSearchOptimizer,
    "MultiObjectiveDingoOptimizer": MultiObjectiveDingoOptimizer,
    "MultiObjectiveFireflyOptimizer": MultiObjectiveFireflyOptimizer,
    "MultiObjectiveModifiedSocialGroupOptimizer": MultiObjectiveModifiedSocialGroupOptimizer,
    "MultiObjectiveShuffledFrogLeapingOptimizer": MultiObjectiveShuffledFrogLeapingOptimizer,
    "MultiObjectiveGeneticAlgorithmOptimizer": MultiObjectiveGeneticAlgorithmOptimizer,
    "MultiObjectiveTeachingLearningBasedOptimizer": MultiObjectiveTeachingLearningBasedOptimizer
}

# Mapping of single-objective solvers to their multi-objective counterparts
_MULTI_OBJECTIVE_MAPPING: Dict[str, str] = {
    "ArtificialBeeColonyOptimizer": "MultiObjectiveArtificialBeeColonyOptimizer",
    "GreyWolfOptimizer": "MultiObjectiveGreyWolfOptimizer",
    "WhaleOptimizer": "MultiObjectiveWhaleOptimizer",
    "ParticleSwarmOptimizer": "MultiObjectiveParticleSwarmOptimizer",
    "AntColonyOptimizer": "MultiObjectiveAntColonyOptimizer",
    "BatOptimizer": "MultiObjectiveBatOptimizer",
    "ArtificialEcosystemOptimizer": "MultiObjectiveArtificialEcosystemOptimizer",
    "CuckooSearchOptimizer": "MultiObjectiveCuckooSearchOptimizer",
    "DingoOptimizer": "MultiObjectiveDingoOptimizer",
    "FireflyOptimizer": "MultiObjectiveFireflyOptimizer",
    "ModifiedSocialGroupOptimizer": "MultiObjectiveModifiedSocialGroupOptimizer",
    "ShuffledFrogLeapingOptimizer": "MultiObjectiveShuffledFrogLeapingOptimizer",
    "GeneticAlgorithmOptimizer": "MultiObjectiveGeneticAlgorithmOptimizer",
    "TeachingLearningBasedOptimizer": "MultiObjectiveTeachingLearningBasedOptimizer",
}


def find_solver(solver_name: str) -> Type[Solver]:
    solver_name = solver_name.strip()
    if solver_name not in _SOLVER_REGISTRY:
        available_solvers = list(_SOLVER_REGISTRY.keys())
        raise ValueError(
            f"Solver '{solver_name}' not found. "
            f"Available solvers: {available_solvers}"
        )
    return _SOLVER_REGISTRY[solver_name]


def register_solver(name: str, solver_class: Type[Solver]) -> None:
    _SOLVER_REGISTRY[name] = solver_class


def create_solver(
    solver_name: str, 
    objective_func: Callable, 
    lb: Union[float, np.ndarray], 
    ub: Union[float, np.ndarray], 
    dim: int, 
    maximize: bool = True,
    **kwargs
) -> Solver:
    """
    Create a solver instance with automatic detection of objective function type.
    
    Automatically detects whether to use single-objective or multi-objective version
    based on objective function output for algorithms that have both versions.
    """
    solver_name = solver_name.strip()
    
    # Check if this solver has a multi-objective counterpart
    if solver_name in _MULTI_OBJECTIVE_MAPPING:
        # Test the objective function to determine its output type
        try:
            # Create a test point within bounds
            test_point = np.random.uniform(lb, ub, dim)
            result = objective_func(test_point)
            
            # Check if result is a list/array (multi-objective) or scalar (single-objective)
            if hasattr(result, '__len__') and len(result) > 1:
                # Multi-objective function detected
                n_objectives = len(result)
                multi_solver_name = _MULTI_OBJECTIVE_MAPPING[solver_name]
                print(f"Detected multi-objective function with {n_objectives} objectives. Using {multi_solver_name}.")
                solver_class = find_solver(multi_solver_name)
                return solver_class(objective_func, lb, ub, dim, maximize, **kwargs)
            else:
                # Single-objective function detected
                print(f"Detected single-objective function. Using {solver_name}.")
                solver_class = find_solver(solver_name)
                return solver_class(objective_func, lb, ub, dim, maximize, **kwargs)
                
        except Exception as e:
            print(f"Warning: Could not auto-detect objective function type: {e}")
            print(f"Falling back to single-objective {solver_name}.")
            solver_class = find_solver(solver_name)
            return solver_class(objective_func, lb, ub, dim, maximize, **kwargs)
    
    # For solvers without multi-objective counterparts, use the standard approach
    solver_class = find_solver(solver_name)
    return solver_class(objective_func, lb, ub, dim, maximize, **kwargs)


def show_solvers(mode: str = "all") -> None:
    """
    Display list of solvers by mode:
    - 'single': Show only single-objective solvers.
    - 'multi': Show only multi-objective solvers.
    - 'all': Show all solvers (default).
    """
    mode = mode.lower().strip()
    
    if mode not in ["single", "multi", "all"]:
        raise ValueError("mode must be one of: 'single', 'multi', 'all'")

    single_solvers = []
    multi_solvers = []

    # Categorize solvers
    for name in _SOLVER_REGISTRY:
        if name.startswith("MultiObjective"):
            multi_solvers.append(name)
        else:
            single_solvers.append(name)

    if mode == "single":
        print("Single-objective Solvers:")
        for solver in sorted(single_solvers):
            print(f"  - {solver}")
    elif mode == "multi":
        print("Multi-objective Solvers:")
        for solver in sorted(multi_solvers):
            print(f"  - {solver}")
    else:  # all
        print("All Solvers:")
        print("Single-objective:")
        for solver in sorted(single_solvers):
            print(f"  - {solver}")
        print("\nMulti-objective:")
        for solver in sorted(multi_solvers):
            print(f"  - {solver}")
