import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember

class MultiObjectiveWhaleOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Whale Optimization Algorithm
    
    This algorithm extends the standard WOA for multi-objective optimization
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
        - alpha: Grid inflation parameter (default: 0.1)
        - n_grid: Number of grids per dimension (default: 7)
        - beta: Leader selection pressure (default: 2)
        - gamma: Archive removal pressure (default: 2)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Whale Optimizer"

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Main optimization method for multi-objective WOA
        
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
            # Update a parameters (decreases linearly)
            a = 2 - iter * (2 / max_iter)
            a2 = -1 + iter * ((-1) / max_iter)
            
            # Update all search agents
            for i, whale in enumerate(population):
                new_position = np.zeros(self.dim)
                
                # Select leader from archive using grid-based selection
                leader = self._select_leader()
                
                # If no leader in archive, use random whale from population
                if leader is None:
                    leader = np.random.choice(population)
                
                # Update position for each dimension
                for j in range(self.dim):
                    r1 = np.random.random()
                    r2 = np.random.random()
                    
                    A = 2 * a * r1 - a  # Eq. (2.3) in the paper
                    C = 2 * r2          # Eq. (2.4) in the paper
                    
                    b = 1               # parameters in Eq. (2.5)
                    l = (a2 - 1) * np.random.random() + 1  # parameters in Eq. (2.5)
                    
                    p = np.random.random()  # p in Eq. (2.6)
                    
                    if p < 0.5:
                        if abs(A) >= 1:
                            # Search for prey (exploration phase)
                            # Select random leader from archive for exploration
                            if self.archive:
                                rand_leader = np.random.choice(self.archive)
                                D_X_rand = abs(C * rand_leader.position[j] - whale.position[j])  # Eq. (2.7)
                                new_position[j] = rand_leader.position[j] - A * D_X_rand  # Eq. (2.8)
                            else:
                                # If archive is empty, use random whale from population
                                rand_whale = np.random.choice(population)
                                D_X_rand = abs(C * rand_whale.position[j] - whale.position[j])
                                new_position[j] = rand_whale.position[j] - A * D_X_rand
                        else:
                            # Encircling prey (exploitation phase)
                            D_leader = abs(C * leader.position[j] - whale.position[j])  # Eq. (2.1)
                            new_position[j] = leader.position[j] - A * D_leader  # Eq. (2.2)
                    else:
                        # Bubble-net attacking method (spiral updating position)
                        distance_to_leader = abs(leader.position[j] - whale.position[j])
                        # Eq. (2.5) - spiral movement
                        new_position[j] = distance_to_leader * np.exp(b * l) * np.cos(l * 2 * np.pi) + leader.position[j]
                
                # Ensure positions stay within bounds
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Update whale position and fitness
                population[i].position = new_position
                population[i].multi_fitness = self.objective_func(new_position)
            
            # Update archive with current population
            self._add_to_archive(population)
            
            # Store archive state for history
            history_archive.append([whale.copy() for whale in self.archive])
            
            # Update progress
            self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
        
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver
        self._end_step_solver()
        
        return history_archive, self.archive
