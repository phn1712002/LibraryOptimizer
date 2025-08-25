import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember

class MultiObjectiveFireflyOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Firefly Algorithm Optimizer
    
    This algorithm extends the standard Firefly Algorithm for multi-objective optimization
    using archive management and grid-based selection for leader selection.
    
    Parameters:
    -----------
    objective_func : Callable
        Objective function that returns a list of fitness values
    lb : Union[float, np.ndarray]
        Lower bounds for variables
    ub : Union[float, np.ndarray]
        Upper bounds for variables
    dim : int
        Problem dimension
    **kwargs
        Additional parameters:
        - archive_size: Size of the external archive (default: 100)
        - alpha: Randomness parameter (default: 0.5)
        - betamin: Minimum attractiveness (default: 0.2)
        - gamma: Absorption coefficient (default: 1.0)
        - alpha_reduction: Whether to reduce alpha over iterations (default: True)
        - alpha_delta: Alpha reduction factor (default: 0.97)
        - grid_alpha: Grid inflation parameter (default: 0.1)
        - n_grid: Number of grids per dimension (default: 7)
        - beta: Leader selection pressure (default: 2)
        - gamma_removal: Archive removal pressure (default: 2)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Firefly Optimizer"
        
        # Firefly algorithm specific parameters
        self.alpha = kwargs.get('alpha', 0.5)  # Randomness parameter
        self.betamin = kwargs.get('betamin', 0.2)  # Minimum attractiveness
        self.gamma = kwargs.get('gamma', 1.0)  # Absorption coefficient
        self.alpha_reduction = kwargs.get('alpha_reduction', True)  # Reduce alpha over time
        self.alpha_delta = kwargs.get('alpha_delta', 0.97)  # Alpha reduction factor
        
        # Store initial alpha for reference
        self.alpha_initial = self.alpha
        
        # Grid parameters (override parent defaults if provided)
        self.alpha_grid = kwargs.get('grid_alpha', 0.1)
        self.n_grid = kwargs.get('n_grid', 7)
        self.beta_selection = kwargs.get('beta', 2)
        self.gamma_removal = kwargs.get('gamma_removal', 2)

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Main optimization method for multi-objective Firefly Algorithm
        
        Parameters:
        -----------
        search_agents_no : int
            Number of search agents (population size)
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List, List[MultiObjectiveMember]]
            History of archive states and the final archive
        """
        # Initialize storage
        history_archive = []
        
        # Initialize population
        population = self._init_population(search_agents_no)
        
        # Initialize archive with non-dominated solutions
        self._determine_domination(population)
        non_dominated = self._get_non_dominated_particles(population)
        self.archive.extend(non_dominated)
        
        # Initialize grid for archive
        costs = self._get_fitness(self.archive)
        if costs.size > 0:
            self.grid = self._create_hypercubes(costs)
            for particle in self.archive:
                particle.grid_index, particle.grid_sub_index = self._get_grid_index(particle)
        
        # Calculate scale for random movement
        scale = np.abs(self.ub - self.lb)
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Evaluate all fireflies
            for i in range(search_agents_no):
                population[i].multi_fitness = self.objective_func(population[i].position)
            
            # Move all fireflies towards brighter ones in the archive
            for i in range(search_agents_no):
                # Select a leader from archive using grid-based selection
                leader = self._select_leader()
                
                if leader is not None:
                    # Calculate distance between firefly and leader
                    r = np.sqrt(np.sum((population[i].position - leader.position)**2))
                    
                    # Calculate attractiveness
                    beta = self._calculate_attractiveness(r)
                    
                    # Generate random movement
                    random_move = self.alpha * (np.random.random(self.dim) - 0.5) * scale
                    
                    # Update position
                    new_position = (population[i].position * (1 - beta) + 
                                   leader.position * beta + 
                                   random_move)
                    
                    # Apply bounds
                    new_position = np.clip(new_position, self.lb, self.ub)
                    
                    # Update firefly position and fitness
                    population[i].position = new_position
                    population[i].multi_fitness = self.objective_func(new_position)
            
            # Update archive with current population
            self._add_to_archive(population)
            
            # Store archive state for history
            history_archive.append([firefly.copy() for firefly in self.archive])
            
            # Reduce alpha (randomness) over iterations if enabled
            if self.alpha_reduction:
                self.alpha = self._reduce_alpha(self.alpha, self.alpha_delta)
            
            # Update progress
            self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
        
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver
        self._end_step_solver()
        
        return history_archive, self.archive
    
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
    