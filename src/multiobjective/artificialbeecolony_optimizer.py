import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember
from utils.general import roulette_wheel_selection, normalized_values

class BeeMulti(MultiObjectiveMember):
    def __init__(self, position: np.ndarray, fitness: np.ndarray, trial: int = 0):
        super().__init__(position, fitness)
        self.trial = trial 
    
    def copy(self):
        new_member = BeeMulti(self.position.copy(), self.multi_fitness.copy())
        new_member.dominated = self.dominated
        new_member.grid_index = self.grid_index
        new_member.grid_sub_index = self.grid_sub_index
        new_member.trial = self.trial
        return new_member
    
    def __str__(self):
        return f"Position: {self.position} - Fitness: {self.fitness} - Dominated: {self.dominated} - Trial: {self.trial}"
    
class MultiObjectiveArtificialBeeColonyOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Artificial Bee Colony Optimizer
    
    This algorithm extends the standard ABC for multi-objective optimization
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
        - abandonment_limit: Trial limit for scout bees (default: calculated as 0.6 * dim * population_size)
        - n_onlooker: Number of onlooker bees (default: same as population size)
        - alpha: Grid inflation parameter (default: 0.1)
        - n_grid: Number of grids per dimension (default: 7)
        - beta: Leader selection pressure (default: 2)
        - gamma: Archive removal pressure (default: 2)
        - acceleration_coef: Acceleration coefficient (default: 1.0)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Artificial Bee Colony Optimizer"
        
        # ABC-specific parameters
        self.acceleration_coef = kwargs.get('acceleration_coef', 1.0)
        self.n_onlooker = kwargs.get('n_onlooker', None)
        self.abandonment_limit = kwargs.get('abandonment_limit', None)
    
    
    def _init_population(self, search_agents_no) -> List[BeeMulti]:
        population = []
        for _ in range(search_agents_no):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            fitness = self.objective_func(position)
            population.append(BeeMulti(position, fitness, 0))
        return population
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[BeeMulti]]:
        """
        Main optimization method for multi-objective ABC
        
        Parameters:
        -----------
        search_agents_no : int
            Number of search agents (population size)
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List, List[BeeMulti]]
            History of archive states and the final archive
        """
        # Initialize storage
        history_archive = []
        
        # Set default number of onlooker bees
        if self.n_onlooker is None:
            self.n_onlooker = search_agents_no
        
        # Set default abandonment limit
        if self.abandonment_limit is None:
            # Default: 60% of variable dimension * population size (as in MATLAB code)
            self.abandonment_limit = int(0.6 * self.dim * search_agents_no)
        
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
            # Phase 1: Employed Bees
            bee_population = [bee.copy() for bee in population]
            
            for i in range(search_agents_no):
                # Choose a random neighbor different from current bee
                neighbors = [j for j in range(search_agents_no) if j != i]
                k = np.random.choice(neighbors)
                
                # Define acceleration coefficient
                phi = self.acceleration_coef * np.random.uniform(-1, 1, self.dim)
                
                # Generate new candidate solution using neighbor guidance
                new_position = population[i].position + phi * (population[i].position - population[k].position)
                
                # Apply bounds
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Evaluate new fitness
                new_fitness = self.objective_func(new_position)
                new_bee = BeeMulti(new_position, new_fitness)
                
                # Check if new solution dominates current solution
                if self._dominates(new_bee, population[i]):
                    bee_population[i] = new_bee
                    bee_population[i].trial = 0
                else:
                    bee_population[i].trial = population[i].trial + 1
            
            # Update population
            population = bee_population
            
            # Phase 2: Onlooker Bees
            # Calculate fitness values for selection probabilities
            # For multi-objective, we use a simple aggregation approach for selection
            # Sum of normalized objectives (assuming minimization for all objectives)
            # Normalize costs and sum them (lower sum is better)
            normalized_costs = self._get_normalized_costs(population)
            fitness_values = 1.0 / (np.sum(normalized_costs, axis=1) + 1e-10)
            
            # Normalize to get probabilities
            probabilities = fitness_values / np.sum(fitness_values)
            
            for _ in range(self.n_onlooker):
                # Select source site using roulette wheel selection
                i = self._roulette_wheel_selection(probabilities)
                
                # Choose a random neighbor different from current bee
                neighbors = [j for j in range(search_agents_no) if j != i]
                k = np.random.choice(neighbors)
                
                # Define acceleration coefficient
                phi = self.acceleration_coef * np.random.uniform(-1, 1, self.dim)
                
                # Generate new candidate solution using neighbor guidance
                new_position = population[i].position + phi * (population[i].position - population[k].position)
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Evaluate new fitness
                new_fitness = self.objective_func(new_position)
                new_bee = BeeMulti(new_position, new_fitness)
                
                # Greedy selection
                if self._dominates(new_bee, population[i]):
                    population[i] = new_bee
                    population[i].trial = 0
                else:
                    population[i].trial += 1
            
            # Phase 3: Scout Bees
            for i in range(search_agents_no):
                if population[i].trial >= self.abandonment_limit:
                    # Replace abandoned solution
                    position = np.random.uniform(self.lb, self.ub, self.dim)
                    fitness = self.objective_func(position)
                    population[i] = BeeMulti(position, fitness)
                    population[i].trial = 0
            
            # Update archive with current population
            self._add_to_archive(population)
            
            # Store archive state for history
            history_archive.append([bee.copy() for bee in self.archive])
            
            # Update progress
            self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
        
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver
        self._end_step_solver()
        
        return history_archive, self.archive
    
    def _get_normalized_costs(self, population: List[MultiObjectiveMember]) -> np.ndarray:
        """Get normalized cost matrix from population for aggregation methods"""
        costs = self._get_costs(population)
        return normalized_values(costs)
    
    def _roulette_wheel_selection(self, probabilities):
        """Perform roulette wheel selection (fitness proportionate selection)."""
        return roulette_wheel_selection(probabilities)