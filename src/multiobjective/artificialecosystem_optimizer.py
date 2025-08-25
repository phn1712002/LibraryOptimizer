import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember
from utils.general import levy_flight

class MultiObjectiveArtificialEcosystemOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Artificial Ecosystem-based Optimization (AEO) algorithm.
    
    This algorithm extends the standard AEO for multi-objective optimization
    using archive management and grid-based selection.
    
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
        - production_weight: Weight for production phase (default: 1.0)
        - consumption_weight: Weight for consumption phase (default: 1.0)
        - decomposition_weight: Weight for decomposition phase (default: 1.0)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Artificial Ecosystem Optimizer"
        
        # Algorithm-specific parameters
        self.production_weight = kwargs.get('production_weight', 1.0)
        self.consumption_weight = kwargs.get('consumption_weight', 1.0)
        self.decomposition_weight = kwargs.get('decomposition_weight', 1.0)

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Main optimization method for multi-objective AEO
        
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
        costs = self._get_costs(self.archive)
        if costs.size > 0:
            self.grid = self._create_hypercubes(costs)
            for particle in self.archive:
                particle.grid_index, particle.grid_sub_index = self._get_grid_index(particle)
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Production phase: Create new organism based on archive leaders
            new_population = self._production_phase(population, iter, max_iter)
            
            # Consumption phase: Update organisms based on consumption behavior
            new_population = self._consumption_phase(new_population, population)
            
            # Decomposition phase: Update organisms based on decomposition behavior
            new_population = self._decomposition_phase(new_population)
            
            # Evaluate new population and update archive
            self._add_to_archive(new_population)
            
            # Store archive state for history
            history_archive.append([member.copy() for member in self.archive])
            
            # Update progress
            self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
        
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver
        self._end_step_solver()
        
        return history_archive, self.archive
    
    def _production_phase(self, population: List, iter: int, max_iter: int) -> List[MultiObjectiveMember]:
        """
        Production phase: Create new organism based on archive leaders and random position.
        
        Parameters:
        -----------
        population : List[MultiObjectiveMember]
            Current population
        iter : int
            Current iteration
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        List[MultiObjectiveMember]
            New population after production phase
        """
        new_population = []
        
        # Select leader from archive using grid-based selection
        leader = self._select_leader()
        
        # If no leader in archive, use random member from population
        if leader is None:
            leader = np.random.choice(population)
        
        # Create random position in search space
        random_position = np.random.uniform(self.lb, self.ub, self.dim)
        
        # Calculate production weight (decreases linearly)
        r1 = np.random.random()
        a = (1 - iter / max_iter) * r1
        
        # Create first organism: combination of leader and random position
        new_position = (1 - a) * leader.position + a * random_position
        new_position = np.clip(new_position, self.lb, self.ub)
        new_fitness = self.objective_func(new_position)
        new_population.append(MultiObjectiveMember(new_position, new_fitness))
        
        return new_population
    
    def _consumption_phase(self, new_population: List, old_population: List) -> List[MultiObjectiveMember]:
        """
        Consumption phase: Update organisms based on consumption behavior.
        
        Parameters:
        -----------
        new_population : List[MultiObjectiveMember]
            Population from production phase
        old_population : List[MultiObjectiveMember]
            Original population
            
        Returns:
        --------
        List[MultiObjectiveMember]
            Population after consumption phase
        """
        # Handle second organism (special case)
        if len(old_population) >= 2:
            # Generate consumption factor C using Levy flight
            C = 0.5 * self._levy_flight(self.dim)
            
            # Second organism consumes from producer (first organism)
            new_position = old_population[1].position + C * (
                old_population[1].position - new_population[0].position
            )
            
            # Apply bounds
            new_position = np.clip(new_position, self.lb, self.ub)
            new_fitness = self.objective_func(new_position)
            new_population.append(MultiObjectiveMember(new_position, new_fitness))
        
        # For remaining organisms (starting from third one)
        for i in range(2, len(old_population)):
            # Generate consumption factor C using Levy flight
            C = 0.5 * self._levy_flight(self.dim)
            
            r = np.random.random()
            
            if r < 1/3:
                # Consume from producer (first organism)
                new_position = old_population[i].position + C * (
                    old_population[i].position - new_population[0].position
                )
            elif 1/3 <= r < 2/3:
                # Consume from random consumer (between 1 and i-1)
                random_idx = np.random.randint(1, i)
                new_position = old_population[i].position + C * (
                    old_population[i].position - old_population[random_idx].position
                )
            else:
                # Consume from both producer and random consumer
                r2 = np.random.random()
                random_idx = np.random.randint(1, i)
                new_position = old_population[i].position + C * (
                    r2 * (old_population[i].position - new_population[0].position) +
                    (1 - r2) * (old_population[i].position - old_population[random_idx].position)
                )
            
            # Apply bounds
            new_position = np.clip(new_position, self.lb, self.ub)
            new_fitness = self.objective_func(new_position)
            new_population.append(MultiObjectiveMember(new_position, new_fitness))
        
        return new_population
    
    def _decomposition_phase(self, population: List) -> List[MultiObjectiveMember]:
        """
        Decomposition phase: Update organisms based on decomposition behavior.
        
        Parameters:
        -----------
        population : List[MultiObjectiveMember]
            Current population
            
        Returns:
        --------
        List[MultiObjectiveMember]
            Population after decomposition phase
        """
        new_population = []
        
        # Select leader from archive for decomposition guidance
        leader = self._select_leader()
        
        # If no leader in archive, use random member from population
        if leader is None:
            leader = np.random.choice(population)
        
        for i in range(len(population)):
            # Generate decomposition factors
            r3 = np.random.random()
            weight_factor = 3 * np.random.normal(0, 1)
            
            # Calculate new position using decomposition equation
            random_multiplier = np.random.randint(1, 3)  # This gives 1 or 2
            new_position = leader.position + weight_factor * (
                (r3 * random_multiplier - 1) * leader.position -
                (2 * r3 - 1) * population[i].position
            )
            
            # Apply bounds
            new_position = np.clip(new_position, self.lb, self.ub)
            new_fitness = self.objective_func(new_position)
            new_population.append(MultiObjectiveMember(new_position, new_fitness))
        
        return new_population

    def _levy_flight(self):
        """Generate Levy flight step using utility function"""
        return levy_flight(self.dim, self.beta)