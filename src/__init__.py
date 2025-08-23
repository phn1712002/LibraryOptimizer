
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
    "ModifiedSocialGroupOptimizer": ModifiedSocialGroupOptimizer
}


def find_solver(solver_name: str) -> Type[Solver]:
    solver_name_lower = solver_name.strip()
    if solver_name_lower not in _SOLVER_REGISTRY:
        available_solvers = list(_SOLVER_REGISTRY.keys())
        raise ValueError(
            f"Solver '{solver_name}' not found. "
            f"Available solvers: {available_solvers}"
        )
    return _SOLVER_REGISTRY[solver_name_lower]


def register_solver(name: str, solver_class: Type[Solver]) -> None:
    _SOLVER_REGISTRY[name.lower()] = solver_class


def create_solver(
    solver_name: str, 
    objective_func: Callable, 
    lb: Union[float, np.ndarray], 
    ub: Union[float, np.ndarray], 
    dim: int, 
    maximize: bool = True,
    **kwargs
) -> Solver:
    solver_class = find_solver(solver_name)
    return solver_class(objective_func, lb, ub, dim, maximize, **kwargs)
