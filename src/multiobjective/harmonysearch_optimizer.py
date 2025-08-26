import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember

class MultiObjectiveHarmonySearchOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Harmony Search Algorithm.
    
    This algorithm extends the standard Harmony Search for multi-objective optimization
    using archive management and grid-based selection for maintaining diversity.
    
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
    maximize : bool
        Optimization direction
    **kwargs
        Additional algorithm parameters:
        - hms: Harmony Memory Size (default: 100)
        - hmcr: Harmony Memory Considering Rate (default: 0.95)
        - par: Pitch Adjustment Rate (default: 0.3)
        - bw: Bandwidth (default: 0.2)
        - archive_size: Size of the external archive (default: 100)
        - alpha: Grid inflation parameter (default: 0.1)
        - n_grid: Number of grids per dimension (default: 7)
        - beta: Leader selection pressure (default: 2)
        - gamma: Archive removal pressure (default: 2)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Harmony Search Optimizer"
        
        # Algorithm-specific parameters with defaults
        self.hms = kwargs.get('hms', 100)  # Harmony Memory Size
        self.hmcr = kwargs.get('hmcr', 0.95)  # Harmony Memory Considering Rate
        self.par = kwargs.get('par', 0.3)  # Pitch Adjustment Rate
        self.bw = kwargs.get('bw', 0.2)  # Bandwidth
        
        # Initialize harmony memory
        self.harmony_memory = None
        self.harmony_fitness = None
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Main optimization method for multi-objective Harmony Search.
        
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
        
        # Initialize harmony memory
        self.harmony_memory = np.zeros((self.hms, self.dim))
        self.harmony_fitness = []
        
        # Initialize harmony memory with random solutions
        for i in range(self.hms):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            fitness = self.objective_func(position)
            self.harmony_memory[i] = position
            self.harmony_fitness.append(fitness)
        
        # Convert harmony memory to MultiObjectiveMember objects
        population = []
        for i in range(self.hms):
            member = MultiObjectiveMember(self.harmony_memory[i].copy(), np.array(self.harmony_fitness[i]))
            population.append(member)
        
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
            
            # Create new member
            new_member = MultiObjectiveMember(new_harmony.copy(), np.array(new_fitness))
            
            # Update harmony memory if new harmony is non-dominated
            # For multi-objective, we use archive-based replacement
            # Find the worst harmony in memory (most crowded in grid)
            if len(self.archive) > 0:
                # Select a leader from archive to compare
                leader = self._select_leader()
                
                # Create temporary population including new member
                temp_population = population + [new_member]
                
                # Determine domination
                self._determine_domination(temp_population)
                
                # Get non-dominated solutions
                non_dominated_temp = self._get_non_dominated_particles(temp_population)
                
                # If new member is non-dominated, add to archive
                if new_member in non_dominated_temp:
                    # Add to archive and trim if necessary
                    self.archive.append(new_member.copy())
                    self._trim_archive()
                    
                    # Update harmony memory: replace worst harmony with new one
                    # For simplicity, replace a random harmony
                    replace_idx = np.random.randint(0, self.hms)
                    self.harmony_memory[replace_idx] = new_harmony.copy()
                    self.harmony_fitness[replace_idx] = new_fitness
                    population[replace_idx] = new_member
            
            # Update archive with current population
            self._add_to_archive(population)
            
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
