import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember


class MultiObjectiveBiogeographyBasedOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Biogeography-Based Optimization (BBO) algorithm.
    
    BBO is a population-based optimization algorithm inspired by the migration
    of species between habitats in biogeography. This multi-objective version
    extends the algorithm to handle multiple objectives using archive management
    and grid-based selection.
    
    Parameters:
    -----------
    objective_func : Callable
        Multi-objective function that returns array of fitness values
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
        - archive_size: Size of external archive (default: 100)
        - alpha: Grid inflation parameter (default: 0.1)
        - n_grid: Number of grids per dimension (default: 7)
        - beta: Leader selection pressure (default: 2)
        - gamma: Archive removal pressure (default: 2)
        - keep_rate: Rate of habitats to keep (default: 0.2)
        - migration_alpha: Migration coefficient (default: 0.9)
        - p_mutation: Mutation probability (default: 0.1)
        - sigma: Mutation step size (default: 2% of variable range)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Biogeography Based Optimizer"
        
        # Set algorithm parameters with defaults
        self.keep_rate = kwargs.get('keep_rate', 0.2)
        self.migration_alpha = kwargs.get('migration_alpha', 0.9)
        self.p_mutation = kwargs.get('p_mutation', 0.1)
        
        # Calculate sigma as 2% of variable range if not provided
        var_range = self.ub - self.lb
        self.sigma = kwargs.get('sigma', 0.02 * var_range)
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Main optimization method for multi-objective BBO algorithm.
        
        Parameters:
        -----------
        search_agents_no : int
            Number of habitats (population size)
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List, List[MultiObjectiveMember]]
            History of archive states and the final archive
        """
        # Initialize storage
        history_archive = []
        
        # Calculate derived parameters
        n_keep = round(self.keep_rate * search_agents_no)  # Number of habitats to keep
        n_new = search_agents_no - n_keep  # Number of new habitats
        
        # Initialize migration rates (emigration and immigration)
        mu = np.linspace(1, 0, search_agents_no)  # Emigration rates (decreasing)
        lambda_rates = 1 - mu  # Immigration rates (increasing)
        
        # Initialize population
        population = self._init_population(search_agents_no)
        
        # Initialize archive with non-dominated solutions
        self._determine_domination(population)
        non_dominated = self._get_non_dominated_particles(population)
        self.archive.extend(non_dominated)
        
        # Initialize grid
        costs = self._get_fitness(self.archive)
        if costs.size > 0:
            self.grid = self._create_hypercubes(costs)
            for particle in self.archive:
                particle.grid_index, particle.grid_sub_index = self._get_grid_index(particle)
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Create new population for this iteration
            new_population = [member.copy() for member in population]
            
            # Update each habitat
            for i in range(search_agents_no):
                # Migration phase for each dimension
                for k in range(self.dim):
                    # Immigration: if random number <= immigration rate
                    if np.random.random() <= lambda_rates[i]:
                        # Calculate emigration probabilities (excluding current habitat)
                        ep = mu.copy()
                        ep[i] = 0  # Set current habitat probability to 0
                        ep_sum = np.sum(ep)
                        
                        if ep_sum > 0:
                            ep = ep / ep_sum  # Normalize probabilities
                            
                            # Select source habitat using roulette wheel selection
                            # For multi-objective, we select from archive for better diversity
                            if self.archive:
                                # Select leader from archive using grid-based selection
                                leader = self._select_leader()
                                if leader:
                                    # Perform migration using leader guidance
                                    new_population[i].position[k] = (
                                        population[i].position[k] + 
                                        self.migration_alpha * (leader.position[k] - population[i].position[k])
                                    )
                            else:
                                # Fallback: select from population if archive is empty
                                j = np.random.choice(len(population))
                                new_population[i].position[k] = (
                                    population[i].position[k] + 
                                    self.migration_alpha * (population[j].position[k] - population[i].position[k])
                                )
                    
                    # Mutation: if random number <= mutation probability
                    if np.random.random() <= self.p_mutation:
                        new_population[i].position[k] += self.sigma[k] * np.random.randn()
                
                # Apply bounds constraints
                new_population[i].position = np.clip(new_population[i].position, self.lb, self.ub)
                
                # Evaluate new fitness
                new_population[i].multi_fitness = self.objective_func(new_population[i].position)
            
            # Update archive with new population
            self._add_to_archive(new_population)
            
            # Sort population for selection (using multi-objective sorting)
            sorted_population_new = self._sort_population(new_population)
            sorted_population = self._sort_population(population)
            
            # Select next iteration population: keep best + new solutions
            # For multi-objective, we use the sorted population
            next_population = sorted_population[:n_keep] + sorted_population_new[:n_new]
            
            # Fix size population
            size_fix = search_agents_no - len(next_population)
            if size_fix > 0:
                next_population += self._get_random_population(population, size_fix)
            
            # Ensure we have exactly the population size
            population = next_population[:search_agents_no]
            
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
