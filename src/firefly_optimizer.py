import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from utils.general import sort_population

class FireflyOptimizer(Solver):
    """
    Firefly Algorithm Optimizer
    
    Implementation based on the MATLAB Firefly Algorithm by Xin-She Yang.
    Fireflies are attracted to each other based on their brightness (fitness),
    with attractiveness decreasing with distance.
    
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
        - alpha: Randomness parameter (default: 0.5)
        - betamin: Minimum attractiveness (default: 0.2)
        - gamma: Absorption coefficient (default: 1.0)
        - alpha_reduction: Whether to reduce alpha over iterations (default: True)
        - alpha_delta: Alpha reduction factor (default: 0.97)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set default Firefly Algorithm parameters
        self.name_solver = "Firefly Optimizer"
        self.alpha = kwargs.get('alpha', 0.5)  # Randomness parameter
        self.betamin = kwargs.get('betamin', 0.2)  # Minimum attractiveness
        self.gamma = kwargs.get('gamma', 1.0)  # Absorption coefficient
        self.alpha_reduction = kwargs.get('alpha_reduction', True)  # Reduce alpha over time
        self.alpha_delta = kwargs.get('alpha_delta', 0.97)  # Alpha reduction factor
        
        # Store initial alpha for reference
        self.alpha_initial = self.alpha

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Main optimization loop for the Firefly Algorithm.
        
        Parameters:
        -----------
        search_agents_no : int
            Number of fireflies in the population
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List[Member], Member]
            History of best solutions and the final best solution
        """
        # Initialize storage variables
        history_step_solver = []
        
        # Initialize the population of fireflies
        population = self._init_population(search_agents_no)
        
        # Initialize best solution
        sorted_population, _ = self._sort_population(population)
        best_solution = sorted_population[0].copy()
        
        # Call the begin function
        self._begin_step_solver(max_iter)
        
        # Calculate scale for random movement
        scale = np.abs(self.ub - self.lb)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Evaluate all fireflies
            for i in range(search_agents_no):
                population[i].fitness = self.objective_func(population[i].position)
            
            # Sort fireflies by brightness (fitness)
            sorted_population, sorted_indices = self._sort_population(population)
            
            # Update best solution
            current_best = sorted_population[0]
            if self._is_better(current_best, best_solution):
                best_solution = current_best.copy()
            
            # Move all fireflies towards brighter ones
            for i in range(search_agents_no):
                for j in range(search_agents_no):
                    # Firefly i moves towards firefly j if j is brighter
                    if self._is_better(population[j], population[i]):
                        # Calculate distance between fireflies
                        r = np.sqrt(np.sum((population[i].position - population[j].position)**2))
                        
                        # Calculate attractiveness
                        beta = self._calculate_attractiveness(r)
                        
                        # Generate random movement
                        random_move = self.alpha * (np.random.random(self.dim) - 0.5) * scale
                        
                        # Update position
                        new_position = (population[i].position * (1 - beta) + 
                                       population[j].position * beta + 
                                       random_move)
                        
                        # Apply bounds
                        new_position = np.clip(new_position, self.lb, self.ub)
                        
                        # Update firefly position
                        population[i].position = new_position
            
            # Store the best solution at this iteration
            history_step_solver.append(best_solution.copy())
            
            # Reduce alpha (randomness) over iterations if enabled
            if self.alpha_reduction:
                self.alpha = self._reduce_alpha(self.alpha, self.alpha_delta)
            
            # Call the callbacks
            self._callbacks(iter, max_iter, best_solution)
        
        # Final evaluation and storage
        self.history_step_solver = history_step_solver
        self.best_solver = best_solution
        
        # Call the end function
        self._end_step_solver()
        
        return history_step_solver, best_solution
    
    def _sort_population(self, population):
        """
        Sort population based on fitness (brightness).
        
        Parameters:
        -----------
        population : List[Firefly]
            Population to sort
            
        Returns:
        --------
        Tuple[List[Firefly], List[int]]
            Sorted population and sorted indices
        """
        return sort_population(population, self.maximize)

    def _calculate_attractiveness(self, distance: float) -> float:
        """
        Calculate attractiveness based on distance between fireflies.
        
        Parameters:
        -----------
        distance : float
            Euclidean distance between two fireflies
            
        Returns:
        --------
        float
            Attractiveness value (beta)
        """
        beta0 = 1.0  # Attractiveness at distance 0
        return (beta0 - self.betamin) * np.exp(-self.gamma * distance**2) + self.betamin

    def _reduce_alpha(self, current_alpha: float, delta: float) -> float:
        """
        Reduce the randomness parameter alpha over iterations.
        
        Parameters:
        -----------
        current_alpha : float
            Current alpha value
        delta : float
            Reduction factor
            
        Returns:
        --------
        float
            Reduced alpha value
        """
        return current_alpha * delta