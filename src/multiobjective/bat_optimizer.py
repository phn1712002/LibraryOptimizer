import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember
from .._general import linear_decay

class BatMultiMember(MultiObjectiveMember):
    """
    Custom MultiObjectiveMember class for Bat Algorithm with bat-specific attributes.
    
    Attributes:
    -----------
    position : np.ndarray
        Position of the bat in search space
    multi_fitness : np.ndarray
        Multi-objective fitness values
    frequency : float
        Frequency used for velocity update
    velocity : np.ndarray
        Velocity vector of the bat
    loudness : float
        Loudness parameter (controls acceptance of new solutions)
    pulse_rate : float
        Pulse emission rate (controls random walk probability)
    dominated : bool
        Whether this solution is dominated by others
    grid_index : int
        Grid index for archive management
    grid_sub_index : np.ndarray
        Sub-grid indices for each objective
    """
    
    def __init__(self, position: np.ndarray, fitness: np.ndarray, frequency: float = 0.0, 
                 velocity: np.ndarray = None, loudness: float = 1.0, pulse_rate: float = 0.5):
        super().__init__(position, fitness)
        self.frequency = frequency
        self.velocity = velocity if velocity is not None else np.zeros_like(position)
        self.loudness = loudness
        self.pulse_rate = pulse_rate
    
    def copy(self):
        new_member = BatMultiMember(
            self.position.copy(), 
            self.multi_fitness.copy(),
            self.frequency,
            self.velocity.copy(),
            self.loudness,
            self.pulse_rate
        )
        new_member.dominated = self.dominated
        new_member.grid_index = self.grid_index
        new_member.grid_sub_index = self.grid_sub_index.copy() if self.grid_sub_index is not None else None
        return new_member
    
    def __str__(self):
        return (f"Position: {self.position} - Fitness: {self.multi_fitness} - "
                f"Frequency: {self.frequency:.3f} - Loudness: {self.loudness:.3f} - "
                f"Pulse Rate: {self.pulse_rate:.3f} - Dominated: {self.dominated}")


class MultiObjectiveBatOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Bat Optimizer
    
    This algorithm extends the standard Bat Algorithm for multi-objective optimization
    using archive management and grid-based selection for solution evaluation.
    
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
        - alpha: Grid inflation parameter (default: 0.1)
        - n_grid: Number of grids per dimension (default: 7)
        - beta: Leader selection pressure (default: 2)
        - gamma: Archive removal pressure (default: 2)
        - fmin: Minimum frequency (default: 0)
        - fmax: Maximum frequency (default: 2)
        - alpha_loud: Loudness decay constant (default: 0.9)
        - gamma_pulse: Pulse rate increase constant (default: 0.9)
        - ro: Initial pulse emission rate (default: 0.5)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Bat Optimizer"
        
        # Set default BAT parameters
        self.fmin = kwargs.get('fmin', 0.0)          # Minimum frequency
        self.fmax = kwargs.get('fmax', 2.0)          # Maximum frequency
        self.alpha_loud = kwargs.get('alpha_loud', 0.9)  # Loudness decay constant
        self.gamma_pulse = kwargs.get('gamma_pulse', 0.9)  # Pulse rate increase constant
        self.ro = kwargs.get('ro', 0.5)              # Initial pulse emission rate

    def _init_population(self, search_agents_no) -> List[BatMultiMember]:
        """Initialize multi-objective bat population"""
        population = []
        for _ in range(search_agents_no):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            fitness = self.objective_func(position)
            population.append(BatMultiMember(
                position, 
                fitness, 
                frequency=0.0,
                velocity=np.zeros(self.dim),
                loudness=1.0,
                pulse_rate=self.ro
            ))
        return population

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[BatMultiMember]]:
        """
        Main optimization method for multi-objective Bat Algorithm
        
        Parameters:
        -----------
        search_agents_no : int
            Number of bats (search agents)
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List, List[BatMultiMember]]
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
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Update each bat
            for i, bat in enumerate(population):
                # Select a leader from archive using grid-based selection
                leader = self._select_leader()
                
                # If no leader in archive, use random bat from population
                if leader is None:
                    leader = np.random.choice(population)
                
                # Update frequency
                bat.frequency = self.fmin + (self.fmax - self.fmin) * np.random.random()
                
                # Update velocity towards leader
                bat.velocity = bat.velocity + (bat.position - leader.position) * bat.frequency
                
                # Update position
                new_position = bat.position + bat.velocity
                
                # Apply boundary constraints
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Random walk with probability (1 - pulse_rate)
                if np.random.random() > bat.pulse_rate:
                    # Generate random walk step
                    epsilon = -1 + 2 * np.random.random()
                    # Calculate mean loudness of all bats
                    mean_loudness = np.mean([b.loudness for b in population])
                    new_position = leader.position + epsilon * mean_loudness
                    new_position = np.clip(new_position, self.lb, self.ub)
                
                # Evaluate new fitness
                new_fitness = self.objective_func(new_position)
                
                # Create temporary BatMultiMember for comparison
                new_bat = BatMultiMember(
                    new_position, 
                    new_fitness, 
                    bat.frequency,
                    bat.velocity.copy(),
                    bat.loudness,
                    bat.pulse_rate
                )
                
                # Check if new solution is non-dominated compared to current bat
                current_dominates_new = self._dominates(bat, new_bat)
                
                # Update if new solution is better and meets loudness criteria
                if (not current_dominates_new and np.random.random() < bat.loudness):
                    # Update position and fitness
                    population[i].position = new_position
                    population[i].multi_fitness = new_fitness
                    
                    # Update loudness and pulse rate
                    population[i].loudness = self.alpha_loud * bat.loudness
                    population[i].pulse_rate = self.ro * (1 - np.exp(-self.gamma_pulse * iter))
            
            # Update archive with current population
            self._add_to_archive(population)
            
            # Store archive state for history
            history_archive.append([bat.copy() for bat in self.archive])
            
            # Update progress
            self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
        
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver
        self._end_step_solver()
        
        return history_archive, self.archive
