import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember

class MultiObjectiveJAYAOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective JAYA (To Win) Optimizer
    
    This algorithm extends the standard JAYA for multi-objective optimization
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
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective JAYA Optimizer"

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Main optimization method for multi-objective JAYA
        
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
            # Find best and worst solutions in current population
            self._determine_domination(population)
            non_dominated_pop = self._get_non_dominated_particles(population)
            
            # If we have non-dominated solutions, use them as leaders
            if non_dominated_pop:
                # Select best and worst from non-dominated using grid-based selection
                leaders = self._select_multiple_leaders(2)  # Get 2 leaders
                
                if len(leaders) >= 2:
                    best_member = leaders[0]
                    worst_member = leaders[1]
                else:
                    # Fallback: use random selection if not enough leaders
                    if len(non_dominated_pop) >= 2:
                        best_member, worst_member = np.random.choice(non_dominated_pop, 2, replace=False)
                    else:
                        # If only one non-dominated, use it as best and random as worst
                        best_member = non_dominated_pop[0]
                        worst_member = np.random.choice([m for m in population if m not in non_dominated_pop])
            else:
                # If no non-dominated solutions, use random selection
                best_member, worst_member = np.random.choice(population, 2, replace=False)
            
            # Update each search agent
            for i in range(search_agents_no):
                # Create new position using JAYA formula
                new_position = np.zeros(self.dim)
                for j in range(self.dim):
                    rand1 = np.random.random()
                    rand2 = np.random.random()
                    new_position[j] = (
                        population[i].position[j] + 
                        rand1 * (best_member.position[j] - abs(population[i].position[j])) - 
                        rand2 * (worst_member.position[j] - abs(population[i].position[j]))
                    )
                
                # Apply bounds
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Evaluate new fitness
                new_fitness = self.objective_func(new_position)
                
                # Create new member
                new_member = MultiObjectiveMember(new_position, new_fitness)
                
                # Check if new solution dominates current solution
                if self._dominates(new_member, population[i]):
                    population[i].position = new_position
                    population[i].multi_fitness = new_fitness
                    population[i].dominated = False  # Reset domination status
            
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
