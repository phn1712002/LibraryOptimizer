import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from ._general import sort_population


class BatMember(Member):
    """
    Custom Member class for Bat Algorithm with bat-specific attributes.
    
    Attributes:
    -----------
    position : np.ndarray
        Position of the bat in search space
    fitness : float
        Fitness value of the position
    frequency : float
        Frequency used for velocity update
    velocity : np.ndarray
        Velocity vector of the bat
    loudness : float
        Loudness parameter (controls acceptance of new solutions)
    pulse_rate : float
        Pulse emission rate (controls random walk probability)
    """
    
    def __init__(self, position: np.ndarray, fitness: float, frequency: float = 0.0, 
                 velocity: np.ndarray = None, loudness: float = 1.0, pulse_rate: float = 0.5):
        super().__init__(position, fitness)
        self.frequency = frequency
        self.velocity = velocity if velocity is not None else np.zeros_like(position)
        self.loudness = loudness
        self.pulse_rate = pulse_rate
    
    def copy(self):
        return BatMember(
            self.position.copy(), 
            self.fitness, 
            self.frequency,
            self.velocity.copy(),
            self.loudness,
            self.pulse_rate
        )
    
    def __str__(self):
        return (f"Position: {self.position} - Fitness: {self.fitness} - "
                f"Frequency: {self.frequency:.3f} - Loudness: {self.loudness:.3f} - "
                f"Pulse Rate: {self.pulse_rate:.3f}")


class BatOptimizer(Solver):
    """
    Bat Algorithm Optimizer implementation.
    
    The Bat Algorithm is a metaheuristic optimization algorithm inspired by 
    the echolocation behavior of microbats. It uses frequency tuning, loudness,
    and pulse emission rate to control the search process.
    
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
        - fmin: Minimum frequency (default: 0)
        - fmax: Maximum frequency (default: 2)
        - alpha: Loudness decay constant (default: 0.9)
        - gamma: Pulse rate increase constant (default: 0.9)
        - ro: Initial pulse emission rate (default: 0.5)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set solver name
        self.name_solver = "Bat Optimizer"
        
        # Set default BAT parameters
        self.fmin = kwargs.get('fmin', 0.0)          # Minimum frequency
        self.fmax = kwargs.get('fmax', 2.0)          # Maximum frequency
        self.alpha = kwargs.get('alpha', 0.9)        # Loudness decay constant
        self.gamma = kwargs.get('gamma', 0.9)        # Pulse rate increase constant
        self.ro = kwargs.get('ro', 0.5)              # Initial pulse emission rate
    
    def _init_population(self, search_agents_no) -> List:
        """
        Initialize population of bats with bat-specific parameters.
        
        Parameters:
        -----------
        search_agents_no : int
            Number of bats to initialize
            
        Returns:
        --------
        List[BatMember]
            List of initialized bat members
        """
        population = []
        for _ in range(search_agents_no):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            fitness = self.objective_func(position)
            # Initialize with default bat parameters
            population.append(BatMember(
                position, 
                fitness, 
                frequency=0.0,
                velocity=np.zeros(self.dim),
                loudness=1.0,
                pulse_rate=self.ro
            ))
        return population

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, BatMember]:
        """
        Main optimization method for the Bat Algorithm.
        
        Parameters:
        -----------
        search_agents_no : int
            Number of bats (search agents)
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List, BatMember]
            History of best solutions and the best solution found
        """
        # Initialize population
        population = self._init_population(search_agents_no)
        
        # Find initial best solution
        sorted_population, _ = self._sort_population(population)
        best_solver = sorted_population[0].copy()
        
        # Initialize history
        history_step_solver = []
        
        # Start solver (show progress bar)
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Update each bat
            for i in range(search_agents_no):
                # Update frequency
                population[i].frequency = self.fmin + (self.fmax - self.fmin) * np.random.random()
                
                # Update velocity
                population[i].velocity = population[i].velocity + (population[i].position - best_solver.position) * population[i].frequency
                
                # Update position
                new_position = population[i].position + population[i].velocity
                
                # Apply boundary constraints
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Random walk with probability (1 - pulse_rate)
                if np.random.random() > population[i].pulse_rate:
                    # Generate random walk step
                    epsilon = -1 + 2 * np.random.random()
                    # Calculate mean loudness of all bats
                    mean_loudness = np.mean([bat.loudness for bat in population])
                    new_position = best_solver.position + epsilon * mean_loudness
                    new_position = np.clip(new_position, self.lb, self.ub)
                
                # Evaluate new fitness
                new_fitness = self.objective_func(new_position)
                
                # Create temporary BatMember for comparison
                new_bat = BatMember(new_position, new_fitness, population[i].frequency, 
                                   population[i].velocity.copy(), population[i].loudness, population[i].pulse_rate)
                
                # Update if solution improves and meets loudness criteria
                if (self._is_better(new_bat, population[i]) and 
                    np.random.random() < population[i].loudness):
                    
                    # Update position and fitness
                    population[i].position = new_position
                    population[i].fitness = new_fitness
                    
                    # Update loudness and pulse rate
                    population[i].loudness = self.alpha * population[i].loudness
                    population[i].pulse_rate = self.ro * (1 - np.exp(-self.gamma * iter))
                
                # Update best solution if improved
                if self._is_better(population[i], best_solver):
                    best_solver = population[i].copy()
            
            # Save history
            history_step_solver.append(best_solver.copy())
            
            # Call callback for progress tracking
            self._callbacks(iter, max_iter, best_solver)
        
        # Store final results
        self.history_step_solver = history_step_solver
        self.best_solver = best_solver
        
        # End solver
        self._end_step_solver()
        
        return history_step_solver, best_solver

    def _sort_population(self, population):
        return sort_population(population, self.maximize)