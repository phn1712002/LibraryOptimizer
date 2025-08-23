import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from utils.general import sort_population


class JAYAOptimizer(Solver):
    """
    JAYA (To Win) Optimizer
    
    A simple and powerful optimization algorithm that always tries to get closer 
    to success (best solution) and away from failure (worst solution).
    
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
        Additional algorithm parameters
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize)
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set solver name
        self.name_solver = "JAYA Optimizer"
        
        # JAYA algorithm doesn't have specific parameters beyond the base ones
        # The algorithm uses random coefficients in the update equation

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Main optimization method for JAYA algorithm.
        
        Parameters:
        -----------
        search_agents_no : int
            Number of search agents (population size)
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List, Member]
            History of best solutions and the best solution found
        """
        # Initialize storage variables
        history_step_solver = []
        
        # Initialize the population
        population = self._init_population(search_agents_no)
        
        # Initialize best solution
        sorted_population, _ = self._sort_population(population)
        best_solution = sorted_population[0].copy()
        
        # Call the begin function
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Find best and worst solutions in current population
            sorted_population, _ = self._sort_population(population)
            best_member = sorted_population[0]
            worst_member = sorted_population[-1]
            
            # Update each search agent
            for i in range(search_agents_no):
                # Generate new candidate solution using JAYA update equation
                # xnew = x + rand*(best - |x|) - rand*(worst - |x|)
                # Note: The MATLAB code uses abs(x(i,j)) but this seems unusual
                # We'll implement the exact formula from the MATLAB code
                
                # Create new position using JAYA formula
                new_position = np.zeros(self.dim)
                for j in range(self.dim):
                    rand1 = np.random.random()
                    rand2 = np.random.random()
                    new_position[j] = (
                        population[i].position[j] + 
                        rand1 * (best_member.position[j] - abs(population[i].position[j])) - 
                        rand2 * (worst_member.position[j] - abs(population[i].position[j]))
                    )
                
                # Apply bounds
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Evaluate new fitness
                new_fitness = self.objective_func(new_position)
                
                # Greedy selection: accept if better
                if self._is_better(Member(new_position, new_fitness), population[i]):
                    population[i].position = new_position
                    population[i].fitness = new_fitness
            
            # Update best solution
            sorted_population, _ = self._sort_population(population)
            current_best = sorted_population[0]
            if self._is_better(current_best, best_solution):
                best_solution = current_best.copy()
            
            # Store the best solution at this iteration
            history_step_solver.append(best_solution.copy())
            
            # Call the callbacks
            self._callbacks(iter, max_iter, best_solution)
        
        # Final evaluation and storage
        self.history_step_solver = history_step_solver
        self.best_solver = best_solution
        
        # Call the end function
        self._end_step_solver()
        return history_step_solver, best_solution
    
    def _sort_population(self, population):
        """Sort population using the utility function."""
        return sort_population(population, self.maximize)
