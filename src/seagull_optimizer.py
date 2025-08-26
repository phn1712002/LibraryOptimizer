import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from ._general import sort_population

class SeagullOptimizer(Solver):
    """
    Seagull Optimization Algorithm (SOA).
    
    SOA is a nature-inspired metaheuristic optimization algorithm that mimics
    the migration and attacking behavior of seagulls in nature. The algorithm
    simulates the migration and attacking behaviors of seagulls, which include:
    
    1. Migration (exploration): Seagulls move towards the best position
    2. Attacking (exploitation): Seagulls attack prey using spiral movements
    
    The algorithm uses a control parameter Fc that decreases linearly to balance
    exploration and exploitation phases.
    
    References:
        Dhiman, G., & Kumar, V. (2019). Seagull optimization algorithm: 
        Theory and its applications for large-scale industrial engineering problems. 
        Knowledge-Based Systems, 165, 169-196.
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        """
        Initialize the Seagull Optimization Algorithm.
        
        Args:
            objective_func (Callable): Objective function to optimize
            lb (Union[float, np.ndarray]): Lower bounds of search space
            ub (Union[float, np.ndarray]): Upper bounds of search space
            dim (int): Number of dimensions in the problem
            maximize (bool): Whether to maximize (True) or minimize (False) objective
            **kwargs: Additional algorithm parameters
        """
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        # Store additional parameters for later use
        self.kwargs = kwargs
        self.name_solver = "Seagull Optimization Algorithm"
        
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Execute the Seagull Optimization Algorithm.
        
        The algorithm simulates the migration and attacking behavior of seagulls:
        1. Migration phase: Seagulls move towards the best position (exploration)
        2. Attacking phase: Seagulls attack prey using spiral movements (exploitation)
        
        Args:
            search_agents_no (int): Number of seagulls in the population
            max_iter (int): Maximum number of iterations for optimization
            
        Returns:
            Tuple[List, Member]: A tuple containing:
                - history_step_solver: List of best solutions at each iteration
                - best_solver: Best solution found overall
        """
        # Initialize the population of search agents
        population = self._init_population(search_agents_no)

        # Initialize storage variables
        history_step_solver = []
        best_solver = self.best_solver
        
        # Call the begin function
        self._begin_step_solver(max_iter)

        # Main optimization loop
        for iter in range(max_iter):
            # Evaluate all search agents and update best solution
            for i, member in enumerate(population):
                # Ensure positions stay within bounds
                population[i].position = np.clip(population[i].position, self.lb, self.ub)
                
                # Update fitness
                population[i].fitness = self.objective_func(population[i].position)
                
                # Update best solution if better solution found
                if self._is_better(population[i], best_solver):
                    best_solver = population[i].copy()
            
            # Update control parameter Fc (decreases linearly from 2 to 0)
            Fc = 2 - iter * (2 / max_iter)
            
            # Update all search agents
            for i, member in enumerate(population):
                new_position = np.zeros(self.dim)
                
                for j in range(self.dim):
                    # Generate random numbers
                    r1 = np.random.random()
                    r2 = np.random.random()
                    
                    # Calculate A1 and C1 parameters
                    A1 = 2 * Fc * r1 - Fc
                    C1 = 2 * r2
                    
                    # Calculate ll parameter
                    ll = (Fc - 1) * np.random.random() + 1
                    
                    # Calculate D_alphs (direction towards best solution)
                    D_alphs = Fc * member.position[j] + A1 * (best_solver.position[j] - member.position[j])
                    
                    # Update position using spiral movement (attacking behavior)
                    X1 = D_alphs * np.exp(ll) * np.cos(ll * 2 * np.pi) + best_solver.position[j]
                    new_position[j] = X1
                
                # Ensure positions stay within bounds
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Update member position
                population[i].position = new_position
            
            # Store the best solution at this iteration
            history_step_solver.append(best_solver.copy())
            
            # Call the callbacks 
            self._callbacks(iter, max_iter, best_solver) 
            
        # Final evaluation of all positions to find the best solution
        self.history_step_solver = history_step_solver
        self.best_solver = best_solver
        
        # Call the end function
        self._end_step_solver()
        return history_step_solver, best_solver
    
    def _sort_population(self, population):
        """
        Sort the population based on fitness.
        """
        return sort_population(population, self.maximize)
