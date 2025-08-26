import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from ._general import sort_population

class HarmonySearchOptimizer(Solver):
    """
    Harmony Search Algorithm.
    
    Harmony Search is a music-inspired metaheuristic optimization algorithm that
    mimics the process of musicians improvising harmonies to find the perfect state
    of harmony. The algorithm maintains a harmony memory (HM) of candidate solutions
    and generates new harmonies through three operations: memory consideration,
    pitch adjustment, and random selection.
    
    Parameters:
    -----------
    objective_func : Callable
        Objective function to optimize
    lb : Union[float, np.ndarray]
        Lower bounds for variables
    ub : Union[float, np.ndarray]
        Upper bounds for variables
    dim : int
        Problem dimension
    maximize : bool, optional
        Optimization direction, default is True (maximize)
    **kwargs
        Additional algorithm parameters:
        - hms: Harmony Memory Size (default: 100)
        - hmcr: Harmony Memory Considering Rate (default: 0.95)
        - par: Pitch Adjustment Rate (default: 0.3)
        - bw: Bandwidth (default: 0.2)
    
    References:
        Geem, Z. W., Kim, J. H., & Loganathan, G. V. (2001). 
        A new heuristic optimization algorithm: harmony search. 
        Simulation, 76(2), 60-68.
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set solver name
        self.name_solver = "Harmony Search Optimizer"
        
        # Algorithm-specific parameters with defaults
        self.hms = kwargs.get('hms', 100)  # Harmony Memory Size
        self.hmcr = kwargs.get('hmcr', 0.95)  # Harmony Memory Considering Rate
        self.par = kwargs.get('par', 0.3)  # Pitch Adjustment Rate
        self.bw = kwargs.get('bw', 0.2)  # Bandwidth
        
        # Initialize harmony memory
        self.harmony_memory = None
        self.harmony_fitness = None
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Execute the Harmony Search Algorithm.
        
        The algorithm maintains a harmony memory of candidate solutions and
        generates new harmonies through a combination of memory consideration,
        pitch adjustment, and random selection.
        
        Args:
            search_agents_no (int): Number of search agents (population size)
            max_iter (int): Maximum number of iterations
            
        Returns:
            Tuple[List, Member]: A tuple containing:
                - history_step_solver: List of best solutions at each iteration
                - best_solver: Best solution found overall
        """
        # Initialize harmony memory
        self.harmony_memory = np.zeros((self.hms, self.dim))
        self.harmony_fitness = np.zeros(self.hms)
        
        # Initialize harmony memory with random solutions
        for i in range(self.hms):
            self.harmony_memory[i] = np.random.uniform(self.lb, self.ub, self.dim)
            self.harmony_fitness[i] = self.objective_func(self.harmony_memory[i])
        
        # Initialize storage variables
        history_step_solver = []
        
        # Find initial best and worst solutions
        if self.maximize:
            best_idx = np.argmax(self.harmony_fitness)
            worst_idx = np.argmin(self.harmony_fitness)
        else:
            best_idx = np.argmin(self.harmony_fitness)
            worst_idx = np.argmax(self.harmony_fitness)
        
        best_solver = Member(self.harmony_memory[best_idx].copy(), self.harmony_fitness[best_idx])
        worst_fitness = self.harmony_fitness[worst_idx]
        worst_idx_current = worst_idx
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Create a new harmony
            new_harmony = np.zeros(self.dim)
            
            for j in range(self.dim):
                if np.random.random() < self.hmcr:
                    # Memory consideration: select from harmony memory
                    harmony_idx = np.random.randint(0, self.hms)
                    new_harmony[j] = self.harmony_memory[harmony_idx, j]
                    
                    # Pitch adjustment
                    if np.random.random() < self.par:
                        new_harmony[j] += self.bw * (2 * np.random.random() - 1)
                else:
                    # Random selection
                    new_harmony[j] = np.random.uniform(self.lb[j], self.ub[j])
            
            # Ensure positions stay within bounds
            new_harmony = np.clip(new_harmony, self.lb, self.ub)
            
            # Evaluate new harmony
            new_fitness = self.objective_func(new_harmony)
            
            # Update harmony memory if new harmony is better than worst
            if self._is_better(new_fitness, worst_fitness):
                self.harmony_memory[worst_idx_current] = new_harmony
                self.harmony_fitness[worst_idx_current] = new_fitness
                
                # Update best and worst
                if self.maximize:
                    best_idx = np.argmax(self.harmony_fitness)
                    worst_idx_current = np.argmin(self.harmony_fitness)
                else:
                    best_idx = np.argmin(self.harmony_fitness)
                    worst_idx_current = np.argmax(self.harmony_fitness)
                
                current_best = Member(self.harmony_memory[best_idx].copy(), self.harmony_fitness[best_idx])
                worst_fitness = self.harmony_fitness[worst_idx_current]
                
                # Update best solution if improved
                if self._is_better(current_best, best_solver):
                    best_solver = current_best.copy()
            
            # Store the best solution at this iteration
            history_step_solver.append(best_solver.copy())
            
            # Update progress
            self._callbacks(iter, max_iter, best_solver)
        
        # Final processing
        self.history_step_solver = history_step_solver
        self.best_solver = best_solver
        
        # End solver
        self._end_step_solver()
        
        return history_step_solver, best_solver
    
    def _sort_population(self, population):
        """
        Sort the population based on fitness.
        """
        return sort_population(population, self.maximize)
