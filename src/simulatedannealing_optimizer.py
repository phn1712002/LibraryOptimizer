import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member


class SimulatedAnnealingOptimizer(Solver):
    """
    Simulated Annealing Optimizer implementation based on Kirkpatrick et al. (1983).
    
    This algorithm minimizes or maximizes a function using the simulated annealing
    method, which is particularly effective for finding global optima in complex
    search spaces with many local optima.
    
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
        Additional algorithm parameters including:
        - max_temperatures: Maximum number of temperature levels (default: 100)
        - tol_fun: Function tolerance for convergence (default: 1e-4)
        - equilibrium_steps: Number of steps per temperature level (default: 500)
        - initial_temperature: Starting temperature (default: None, auto-calculated)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set solver name
        self.name_solver = "Simulated Annealing Optimizer"
        
        # Set default SA parameters
        self.max_temperatures = kwargs.get('max_temperatures', 100)
        self.tol_fun = kwargs.get('tol_fun', 1e-4)
        self.equilibrium_steps = kwargs.get('equilibrium_steps', 500)
        self.initial_temperature = kwargs.get('initial_temperature', None)
        
        # Initialize temperature scaling parameter
        self.mu_scaling = kwargs.get('mu_scaling', 100.0)

    def solver(self, max_iter: int) -> Tuple[List, Member]:
        """
        Execute the simulated annealing optimization algorithm.
        
        Parameters:
        -----------
        max_iter : int
            Maximum number of iterations (temperature levels)
            
        Returns:
        --------
        Tuple[List, Member]
            History of best solutions and the final best solution
        """
        # Override max_iter with max_temperatures if provided
        actual_max_iter = min(max_iter, self.max_temperatures)
        
        # Initialize storage variables
        history_step_solver = []
        
        # Initialize current solution from random position within bounds
        current_position = np.random.uniform(self.lb, self.ub, self.dim)
        current_fitness = self.objective_func(current_position)
        current_solution = Member(current_position, current_fitness)
        
        # Initialize best solution
        best_solution = current_solution.copy()
        
        # Call the begin function
        self._begin_step_solver(actual_max_iter)
        
        # Main optimization loop (temperature levels)
        for iter in range(actual_max_iter):
            # Calculate temperature parameter (inverse temperature)
            T = iter / actual_max_iter  # Ranges from 0 to 1
            
            # Calculate mu parameter for perturbation scaling
            mu = 10 ** (T * self.mu_scaling)
            
            # Simulate thermal equilibrium at this temperature
            for k in range(self.equilibrium_steps):
                # Generate perturbation vector
                random_vector = 2 * np.random.random(self.dim) - 1  # Range [-1, 1]
                dx = self._mu_inv(random_vector, mu) * (self.ub - self.lb)
                
                # Generate new candidate position
                candidate_position = current_solution.position + dx
                
                # Apply bounds constraint
                candidate_position = np.clip(candidate_position, self.lb, self.ub)
                
                # Evaluate candidate fitness
                candidate_fitness = self.objective_func(candidate_position)
                candidate_solution = Member(candidate_position, candidate_fitness)
                
                # Calculate fitness difference
                df = candidate_fitness - current_solution.fitness
                
                # Metropolis acceptance criterion
                if self.maximize:
                    # For maximization: accept if better or with probability exp(df/T)
                    accept = (df > 0) or (np.random.random() > 
                            np.exp(T * df / (abs(current_solution.fitness) + np.finfo(float).eps) / self.tol_fun))
                else:
                    # For minimization: accept if better or with probability exp(-df/T)
                    accept = (df < 0) or (np.random.random() < 
                            np.exp(-T * df / (abs(current_solution.fitness) + np.finfo(float).eps) / self.tol_fun))
                
                if accept:
                    current_solution = candidate_solution.copy()
                
                # Update best solution if current is better
                if self._is_better(current_solution, best_solution):
                    best_solution = current_solution.copy()
            
            # Store the best solution at this temperature level
            history_step_solver.append(best_solution.copy())
            
            # Call the callbacks
            self._callbacks(iter, actual_max_iter, best_solution)
        
        # Final evaluation and storage
        self.history_step_solver = history_step_solver
        self.best_solver = best_solution
        
        # Call the end function
        self._end_step_solver()
        
        return history_step_solver, best_solution

    def _mu_inv(self, y: np.ndarray, mu: float) -> np.ndarray:
        """
        Generate perturbation vectors according to the mu_inv function.
        
        This function is used to generate new candidate points with perturbations
        that are proportional to the current temperature level.
        
        Parameters:
        -----------
        y : np.ndarray
            Random vector in range [-1, 1]
        mu : float
            Temperature scaling parameter
            
        Returns:
        --------
        np.ndarray
            Scaled perturbation vector
        """
        return (((1 + mu) ** np.abs(y) - 1) / mu) * np.sign(y)