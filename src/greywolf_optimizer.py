import numpy as np
from typing import Callable, Union, Tuple, List
from tqdm import tqdm
from ._core import Solver, Member
from utils.general import sort_population, sort_population

class GreyWolfOptimizer(Solver):
    """
    Grey Wolf Optimizer (GWO) Algorithm.
    
    GWO is a nature-inspired metaheuristic optimization algorithm that mimics
    the leadership hierarchy and hunting behavior of grey wolves in nature.
    The algorithm considers the social hierarchy of wolves and simulates three
    main steps of hunting: searching for prey, encircling prey, and attacking prey.
    
    The social hierarchy consists of:
    - Alpha (α): Best solution
    - Beta (β): Second best solution  
    - Delta (δ): Third best solution
    - Omega (ω): Other candidate solutions
    
    References:
        Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer.
        Advances in engineering software, 69, 46-61.
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        """
        Initialize the Grey Wolf Optimizer.
        
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
        self.name_solver = "Grey Wolf Optimizer"
        
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Execute the Grey Wolf Optimization Algorithm.
        
        The algorithm simulates the hunting behavior of grey wolves through
        three main phases controlled by coefficient vectors:
        1. Searching for prey: Exploration phase with random search agents
        2. Encircling prey: Exploitation phase moving towards alpha, beta, delta
        3. Attacking prey: Convergence phase as coefficients approach zero
        
        Args:
            search_agents_no (int): Number of grey wolves in the pack
            max_iter (int): Maximum number of iterations for optimization
            
        Returns:
            Tuple[List, Member]: A tuple containing:
                - history_step_solver: List of best solutions at each iteration
                - best_member: Best solution (alpha wolf) found overall
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
            # Update alpha, beta, delta based on current population
            _, idx = self._sort_population(population)
            alpha = population[idx[0]].copy()
            beta = population[idx[1]].copy()
            delta = population[idx[2]].copy()

            # Update a parameter (decreases linearly from 2 to 0)
            a = 2 - iter * (2 / max_iter)
            
            # Update all search agents
            for i, member in enumerate(population):
                new_position = np.zeros(self.dim)
                
                for j in range(self.dim):
                    # Update position using alpha, beta, and delta wolves
                    r1 = np.random.random()
                    r2 = np.random.random()
                    
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    
                    D_alpha = abs(C1 * alpha.position[j] - member.position[j])
                    X1 = alpha.position[j] - A1 * D_alpha
                    
                    r1 = np.random.random()
                    r2 = np.random.random()
                    
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    
                    D_beta = abs(C2 * beta.position[j] - member.position[j])
                    X2 = beta.position[j] - A2 * D_beta
                    
                    r1 = np.random.random()
                    r2 = np.random.random()
                    
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    
                    D_delta = abs(C3 * delta.position[j] - member.position[j])
                    X3 = delta.position[j] - A3 * D_delta
                    
                    # Update position component
                    new_position[j] = (X1 + X2 + X3) / 3
                
                # Ensure positions stay within bounds
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Update member position and fitness
                population[i].position = new_position
                population[i].fitness = self.objective_func(new_position)
                
                # Update best immediately if better solution found
                if self._is_better(population[i], best_solver):
                    best_solver = population[i].copy()
            
            # Store the best solution at this iteration
            history_step_solver.append(best_solver)
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