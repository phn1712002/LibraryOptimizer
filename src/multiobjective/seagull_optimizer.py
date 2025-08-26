import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember
from .._general import sort_population

class MultiObjectiveSeagullOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Seagull Optimization Algorithm (SOA).
    
    Multi-objective version of the Seagull Optimization Algorithm that mimics
    the migration and attacking behavior of seagulls for multi-objective optimization.
    
    References:
        Dhiman, G., & Kumar, V. (2019). Seagull optimization algorithm: 
        Theory and its applications for large-scale industrial engineering problems. 
        Knowledge-Based Systems, 165, 169-196.
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        """
        Initialize the Multi-Objective Seagull Optimization Algorithm.
        
        Args:
            objective_func (Callable): Objective function to optimize
            lb (Union[float, np.ndarray]): Lower bounds of search space
            ub (Union[float, np.ndarray]): Upper bounds of search space
            dim (int): Number of dimensions in the problem
            maximize (bool): Whether to maximize (True) or minimize (False) objective
            **kwargs: Additional algorithm parameters
        """
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        # Store additional parameters for later use
        self.kwargs = kwargs
        self.name_solver = "Multi-Objective Seagull Optimization Algorithm"
        
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Execute the Multi-Objective Seagull Optimization Algorithm.
        
        The algorithm simulates the migration and attacking behavior of seagulls:
        1. Migration phase: Seagulls move towards the best position (exploration)
        2. Attacking phase: Seagulls attack prey using spiral movements (exploitation)
        
        Args:
            search_agents_no (int): Number of seagulls in the population
            max_iter (int): Maximum number of iterations for optimization
            
        Returns:
            Tuple[List, List[MultiObjectiveMember]]: A tuple containing:
                - history_archive: List of archive states at each iteration
                - final_archive: Final archive of non-dominated solutions
        """
        # Initialize the population of search agents
        population = self._init_population(search_agents_no)

        # Initialize archive with non-dominated solutions
        self._determine_domination(population)
        non_dominated = self._get_non_dominated_particles(population)
        self.archive.extend(non_dominated)
        
        # Build grid
        costs = self._get_fitness(self.archive)
        if costs.size > 0:
            self.grid = self._create_hypercubes(costs)
            for particle in self.archive:
                particle.grid_index, particle.grid_sub_index = self._get_grid_index(particle)
        
        # Initialize storage variables
        history_archive = []
        
        # Call the begin function
        self._begin_step_solver(max_iter)

        # Main optimization loop
        for iter in range(max_iter):
            # Update control parameter Fc (decreases linearly from 2 to 0)
            Fc = 2 - iter * (2 / max_iter)
            
            # Update all search agents
            for i, member in enumerate(population):
                new_position = np.zeros(self.dim)
                
                # Select leader from archive
                leader = self._select_leader()
                if leader is None:
                    # If no leader available, use random member
                    leader = np.random.choice(population)
                
                for j in range(self.dim):
                    # Generate random numbers
                    r1 = np.random.random()
                    r2 = np.random.random()
                    
                    # Calculate A1 and C1 parameters
                    A1 = 2 * Fc * r1 - Fc
                    C1 = 2 * r2
                    
                    # Calculate ll parameter
                    ll = (Fc - 1) * np.random.random() + 1
                    
                    # Calculate D_alphs (direction towards leader)
                    D_alphs = Fc * member.position[j] + A1 * (leader.position[j] - member.position[j])
                    
                    # Update position using spiral movement (attacking behavior)
                    X1 = D_alphs * np.exp(ll) * np.cos(ll * 2 * np.pi) + leader.position[j]
                    new_position[j] = X1
                
                # Ensure positions stay within bounds
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Update member position and fitness
                population[i].position = new_position
                population[i].multi_fitness = self.objective_func(new_position)
                # For compatibility, set scalar fitness to first objective
                population[i].fitness = population[i].multi_fitness[0]
            
            # Add non-dominated solutions to archive
            self._add_to_archive(population)
            
            # Store archive history
            history_archive.append([member.copy() for member in self.archive])
            
            # Call the callbacks 
            self._callbacks(iter, max_iter, self.archive[0] if self.archive else None) 
            
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # Call the end function
        self._end_step_solver()
        return history_archive, self.archive
    
    def _sort_population(self, population):
        """
        Sort the population based on fitness.
        """
        return sort_population(population, self.maximize)
