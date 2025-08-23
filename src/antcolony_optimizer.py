import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from utils.general import roulette_wheel_selection, sort_population


class AntColonyOptimizer(Solver):
    """
    Ant Colony Optimization for Continuous Domains (ACOR).
    
    ACOR is a population-based metaheuristic algorithm inspired by the foraging
    behavior of ants. It uses a solution archive and Gaussian sampling to
    explore the search space.
    
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
        - q: Intensification factor (selection pressure), default 0.5
        - zeta: Deviation-distance ratio, default 1.0
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize)
        
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set solver name
        self.name_solver = "Ant Colony Optimizer for Continuous Domains"
        
        # Set algorithm parameters with defaults
        self.q = kwargs.get('q', 0.5)  # Intensification factor
        self.zeta = kwargs.get('zeta', 1.0)  # Deviation-distance ratio
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Main optimization method for ACOR.
        
        Parameters:
        -----------
        search_agents_no : int
            Number of search agents (population/archive size)
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List, Member]
            History of best solutions and the best solution found
        """
        # Initialize storage variables
        history_step_solver = []
        

        # Initialize population (archive)
        population = self._init_population(search_agents_no)
        
        # Sort initial population
        sorted_population, _ = self._sort_population(population)
        best_solution = sorted_population[0].copy()

        # Start solver (show progress bar)
        self._begin_step_solver(max_iter)
        
        # Calculate solution weights (Gaussian kernel weights)
        w = self._calculate_weights(search_agents_no)
        
        # Calculate selection probabilities
        p = w / np.sum(w)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Calculate means (positions of all solutions in archive)
            means = np.array([member.position for member in population])
            
            # Calculate standard deviations for each solution
            sigma = self._calculate_standard_deviations(means)
            
            # Create new population by sampling from Gaussian distributions
            new_population = self._sample_new_population(means, sigma, p, search_agents_no)
            
            # Merge archive and new population
            merged_population = population + new_population
            
            # Sort merged population and keep only the best solutions
            sorted_merged, _ = self._sort_population(merged_population)
            population = sorted_merged[:search_agents_no]
            
            # Update best solution
            current_best = population[0]
            if self._is_better(current_best, best_solution):
                best_solution = current_best.copy()
            
            # Save history
            history_step_solver.append(best_solution.copy())
            
            # Call callback
            self._callbacks(iter, max_iter, best_solution)
        
        # End solver
        self.history_step_solver = history_step_solver
        self.best_solver = best_solution
        self._end_step_solver()
        
        return history_step_solver, best_solution
    
    def _sort_population(self, population):
        return sort_population(population, self.maximize)

    def _calculate_weights(self, n_pop: int) -> np.ndarray:
        """
        Calculate Gaussian kernel weights for solution selection.
        
        Parameters:
        -----------
        n_pop : int
            Population size
            
        Returns:
        --------
        np.ndarray
            Array of weights for each solution
        """
        w = (1 / (np.sqrt(2 * np.pi) * self.q * n_pop)) * \
            np.exp(-0.5 * (((np.arange(n_pop)) / (self.q * n_pop)) ** 2))
        return w
    
    def _calculate_standard_deviations(self, means: np.ndarray) -> np.ndarray:
        """
        Calculate standard deviations for Gaussian sampling.
        
        Parameters:
        -----------
        means : np.ndarray
            Array of solution positions (means)
            
        Returns:
        --------
        np.ndarray
            Array of standard deviations for each solution
        """
        n_pop = means.shape[0]
        sigma = np.zeros_like(means)
        
        for l in range(n_pop):
            # Calculate average distance to other solutions
            D = np.sum(np.abs(means[l] - means), axis=0)
            sigma[l] = self.zeta * D / (n_pop - 1)
        
        return sigma
    
    def _sample_new_population(self, means: np.ndarray, sigma: np.ndarray, 
                              probabilities: np.ndarray, n_sample: int) -> List[Member]:
        """
        Sample new solutions using Gaussian distributions.
        
        Parameters:
        -----------
        means : np.ndarray
            Array of solution positions (means)
        sigma : np.ndarray
            Array of standard deviations
        probabilities : np.ndarray
            Selection probabilities for each solution
        n_sample : int
            Number of samples to generate
            
        Returns:
        --------
        List[Member]
            List of newly sampled solutions
        """
        new_population = []
        
        for _ in range(n_sample):
            # Initialize new position
            new_position = np.zeros(self.dim)
            
            # Construct solution component by component
            for i in range(self.dim):
                # Select Gaussian kernel using roulette wheel selection
                l = roulette_wheel_selection(probabilities)
                
                # Generate Gaussian random variable
                new_position[i] = means[l, i] + sigma[l, i] * np.random.randn()
            
            # Ensure positions stay within bounds
            new_position = np.clip(new_position, self.lb, self.ub)
            
            # Evaluate fitness
            new_fitness = self.objective_func(new_position)
            
            # Create new member
            new_population.append(Member(new_position, new_fitness))
        
        return new_population
