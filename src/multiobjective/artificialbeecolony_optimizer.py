import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember
from utils.general import roulette_wheel_selection

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
        - n_objectives: Number of objectives (default: 2)
        - limit_trial: Trial limit for scout bees (default: 100)
        - alpha: Grid inflation parameter (default: 0.1)
        - n_grid: Number of grids per dimension (default: 7)
        - beta: Leader selection pressure (default: 2)
        - gamma: Archive removal pressure (default: 2)
        - acceleration_coef: Acceleration coefficient (default: 1.0)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, **kwargs):
        super().__init__(objective_func, lb, ub, dim, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Artificial Bee Colony Optimizer"
        
        # ABC-specific parameters
        self.limit_trial = kwargs.get('limit_trial', 100)
        self.acceleration_coef = kwargs.get('acceleration_coef', 1.0)
        self.n_onlooker = kwargs.get('n_onlooker', None)
    
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
            
            # Select leader from archive
            leader = self._select_leader()
            if leader is None:
                # If no leader in archive, use random bee
                leader = np.random.choice(population)
            
            for i in range(search_agents_no):
                # Generate new candidate solution using leader guidance
                phi = self.acceleration_coef * np.random.uniform(-1, 1, self.dim)
                new_position = population[i].position + phi * (leader.position - population[i].position)
                
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
            # Calculate selection probabilities based on non-domination
            self._determine_domination(population)
            non_dominated_count = sum(1 for p in population if not p.dominated)
            
            if non_dominated_count > 0:
                # Use roulette wheel selection based on non-domination status
                fitness_values = np.array([0 if p.dominated else 1 for p in population])
                probabilities = fitness_values / np.sum(fitness_values)
                
                for m in range(self.n_onlooker):
                    # Select source site
                    i = roulette_wheel_selection(probabilities)
                    
                    # Select leader
                    leader = self._select_leader()
                    if leader is None:
                        leader = np.random.choice(population)
                    
                    # Generate new candidate
                    phi = self.acceleration_coef * np.random.uniform(-1, 1, self.dim)
                    new_position = population[i].position + phi * (leader.position - population[i].position)
                    new_position = np.clip(new_position, self.lb, self.ub)
                    
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
                if population[i].trial >= self.limit_trial:
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
