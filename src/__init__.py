"""
Grey Wolf Optimizer - A Python implementation of the Grey Wolf Optimizer algorithm.

This package provides an implementation of the Grey Wolf Optimizer (GWO) algorithm
for solving optimization problems with bound constraints.
"""

from typing import Dict, Type, Callable, Union
import numpy as np
from .core import Solver
from .greywolf_optimizer import GreyWolfOptimizer
from .whale_optimizer import WhaleOptimizer
from .particleswarm_optimizer import ParticleSwarmOptimizer
from .artificialbeecolony_optimizer import ArtificialBeeColonyOptimizer
__version__ = "0.1.0"
__author__ = "HoangggNam"
__email__ = "phn1712002@gmai.com"
__all__ = [
    "Solver",
    "GreyWolfOptimizer",
    "create_solver",
]
# Registry of available solvers
_SOLVER_REGISTRY: Dict[str, Type[Solver]] = {
    "GreyWolfOptimizer": GreyWolfOptimizer,
    "WhaleOptimizer": WhaleOptimizer,
    "ParticleSwarmOptimizer": ParticleSwarmOptimizer,
    "ArtificialBeeColonyOptimizer": ArtificialBeeColonyOptimizer
}


def find_solver(solver_name: str) -> Type[Solver]:
    """
    Find and return a solver class by name.
    
    Args:
        solver_name: Name of the solver to find (e.g., 'gwo', 'greywolf')
        
    Returns:
        The solver class
        
    Raises:
        ValueError: If the solver name is not found in the registry
    """
    solver_name_lower = solver_name.strip()
    if solver_name_lower not in _SOLVER_REGISTRY:
        available_solvers = list(_SOLVER_REGISTRY.keys())
        raise ValueError(
            f"Solver '{solver_name}' not found. "
            f"Available solvers: {available_solvers}"
        )
    return _SOLVER_REGISTRY[solver_name_lower]


def register_solver(name: str, solver_class: Type[Solver]) -> None:
    """
    Register a new solver class in the registry.
    
    Args:
        name: Name to register the solver under
        solver_class: The solver class to register
    """
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
    """
    Create a solver instance by name with the given parameters.
    
    Args:
        solver_name: Name of the solver to create
        objective_func: Objective function to optimize
        lb: Lower bounds for variables
        ub: Upper bounds for variables  
        dim: Dimension of the problem
        maximize: Whether to maximize (True) or minimize (False)
        **kwargs: Additional parameters specific to the solver
        
    Returns:
        An instance of the requested solver
    """
    solver_class = find_solver(solver_name)
    return solver_class(objective_func, lb, ub, dim, maximize, **kwargs)
