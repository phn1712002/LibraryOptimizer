import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember

class MultiObjectiveGreyWolfOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Grey Wolf Optimizer
    
    This algorithm extends the standard GWO for multi-objective optimization
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
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Grey Wolf Optimizer"

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Main optimization method for multi-objective GWO
        
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
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Update a parameter (decreases linearly from 2 to 0)
            a = 2 - iter * (2 / max_iter)
            
            # Update all search agents
            for i, wolf in enumerate(population):
                new_position = np.zeros(self.dim)
                
                # Select alpha, beta, and delta wolves from archive using grid-based selection
                leaders = self._select_multiple_leaders(3)
                
                # If we don't have enough leaders, use random wolves from population
                if len(leaders) < 3:
                    # Get additional random wolves from population to make 3 leaders
                    available_wolves = [w for w in population if w not in leaders]
                    needed = 3 - len(leaders)
                    if available_wolves:
                        additional = list(np.random.choice(available_wolves, 
                                                         size=min(needed, len(available_wolves)), 
                                                         replace=False))
                        leaders.extend(additional)
                
                # Ensure we have exactly 3 leaders
                if len(leaders) > 3:
                    leaders = leaders[:3]
                
                # Update position using alpha, beta, and delta wolves
                for j in range(self.dim):
                    # Update position using each leader
                    for leader_idx, leader in enumerate(leaders):
                        r1 = np.random.random()
                        r2 = np.random.random()
                        
                        A = 2 * a * r1 - a
                        C = 2 * r2
                        
                        D = abs(C * leader.position[j] - wolf.position[j])
                        X = leader.position[j] - A * D
                        
                        new_position[j] += X
                    # Average the contributions from all leaders
                    new_position[j] /= len(leaders)
                
                # Ensure positions stay within bounds
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Update wolf position and fitness
                population[i].position = new_position
                population[i].multi_fitness = self.objective_func(new_position)
            
            # Update archive with current population
            self._add_to_archive(population)
            
            # Store archive state for history
            history_archive.append([wolf.copy() for wolf in self.archive])
            
            # Update progress
            self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
        
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver
        self._end_step_solver()
        
        return history_archive, self.archive
