import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember
from .._general import roulette_wheel_selection, normalized_values

class BacteriaMultiMember(MultiObjectiveMember):
    """
    Multi-objective member class for Bacteria Foraging Optimization.
    
    Extends MultiObjectiveMember with additional attributes needed for BFO.
    """
    def __init__(self, position: np.ndarray, fitness: np.ndarray, health: float = 0.0):
        super().__init__(position, fitness)
        self.health = health  # Sum of fitness values across chemotaxis loops
    
    def copy(self):
        new_member = BacteriaMultiMember(self.position.copy(), self.multi_fitness.copy())
        new_member.dominated = self.dominated
        new_member.grid_index = self.grid_index
        new_member.grid_sub_index = self.grid_sub_index
        new_member.health = self.health
        return new_member
    
    def __str__(self):
        return f"Position: {self.position} - Fitness: {self.fitness} - Dominated: {self.dominated} - Health: {self.health}"

class MultiObjectiveBacteriaForagingOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Bacteria Foraging Optimization (BFO) Algorithm.
    
    This algorithm extends the standard BFO for multi-objective optimization
    using archive management and grid-based selection. The algorithm simulates
    three main processes in bacterial foraging adapted for multi-objective problems:
    1. Chemotaxis: Movement towards better solutions using Pareto dominance
    2. Reproduction: Reproduction of successful bacteria based on health
    3. Elimination-dispersal: Random elimination and dispersal to maintain diversity
    
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
        Whether to maximize (True) or minimize (False) objectives
    **kwargs
        Additional algorithm parameters:
        - n_reproduction: Number of reproduction steps (default: 4)
        - n_chemotaxis: Number of chemotaxis steps (default: 10)
        - n_swim: Number of swim steps (default: 4)
        - step_size: Step size for movement (default: 0.1)
        - elimination_prob: Probability of elimination-dispersal (default: 0.25)
    
    References:
        Passino, K. M. (2002). Biomimicry of bacterial foraging for distributed 
        optimization and control. IEEE Control Systems Magazine, 22(3), 52-67.
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Bacteria Foraging Optimizer"
        
        # BFO-specific parameters with defaults
        self.n_reproduction = kwargs.get('n_reproduction', 4)
        self.n_chemotaxis = kwargs.get('n_chemotaxis', 10)
        self.n_swim = kwargs.get('n_swim', 4)
        self.step_size = kwargs.get('step_size', 0.1)
        self.elimination_prob = kwargs.get('elimination_prob', 0.25)
    
    def _init_population(self, search_agents_no) -> List[BacteriaMultiMember]:
        """
        Initialize population of bacteria.
        
        Args:
            search_agents_no (int): Number of bacteria in population
            
        Returns:
            List[BacteriaMultiMember]: Initial population
        """
        population = []
        for _ in range(search_agents_no):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            fitness = self.objective_func(position)
            population.append(BacteriaMultiMember(position, fitness, 0.0))
        return population
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[BacteriaMultiMember]]:
        """
        Main optimization method for multi-objective BFO.
        
        The algorithm extends the single-objective BFO by using Pareto dominance
        for comparisons and archive management for storing non-dominated solutions.
        
        Args:
            search_agents_no (int): Number of bacteria in the population
            max_iter (int): Maximum number of iterations for optimization
            
        Returns:
            Tuple[List, List[BacteriaMultiMember]]: 
                - history_archive: History of archive states at each iteration
                - final_archive: Final archive of non-dominated solutions
        """
        # Initialize storage
        history_archive = []
        
        # Initialize population
        population = self._init_population(search_agents_no)
        
        # Initialize archive with non-dominated solutions
        self._determine_domination(population)
        non_dominated = self._get_non_dominated_particles(population)
        self.archive.extend(non_dominated)
        
        # Build grid for archive management
        costs = self._get_fitness(self.archive)
        if costs.size > 0:
            self.grid = self._create_hypercubes(costs)
            for particle in self.archive:
                particle.grid_index, particle.grid_sub_index = self._get_grid_index(particle)
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Main optimization loop (elimination-dispersal events)
        for iter in range(max_iter):
            
            # Reproduction loop
            for reproduction_iter in range(self.n_reproduction):
                
                # Reset health values for new reproduction cycle
                for bacterium in population:
                    bacterium.health = 0.0
                
                # Chemotaxis loop
                for chemotaxis_iter in range(self.n_chemotaxis):
                    
                    # Update each bacterium
                    for i, bacterium in enumerate(population):
                        # Generate random direction vector
                        direction = np.random.uniform(-1, 1, self.dim)
                        direction_norm = np.linalg.norm(direction)
                        
                        if direction_norm > 0:
                            direction = direction / direction_norm
                        
                        # Move bacterium
                        new_position = bacterium.position + self.step_size * direction
                        new_position = np.clip(new_position, self.lb, self.ub)
                        
                        # Evaluate new position
                        new_fitness = self.objective_func(new_position)
                        new_bacterium = BacteriaMultiMember(new_position, new_fitness)
                        
                        # Swim behavior - continue moving in same direction if improvement
                        swim_count = 0
                        while swim_count < self.n_swim:
                            # Use Pareto dominance for multi-objective comparison
                            if self._dominates(new_bacterium, bacterium):
                                # Accept move and continue swimming
                                bacterium.position = new_position
                                bacterium.multi_fitness = new_fitness
                                
                                # Move further in same direction
                                new_position = bacterium.position + self.step_size * direction
                                new_position = np.clip(new_position, self.lb, self.ub)
                                new_fitness = self.objective_func(new_position)
                                new_bacterium = BacteriaMultiMember(new_position, new_fitness)
                                swim_count += 1
                            else:
                                # Stop swimming
                                break
                    
                    # Update health (sum of fitness values for reproduction)
                    for bacterium in population:
                        # Use simple aggregation for health calculation
                        # Sum of normalized objectives (assuming minimization)
                        normalized_fitness = normalized_values(np.array([bacterium.multi_fitness]))
                        bacterium.health += np.sum(normalized_fitness)
                    
                    # Update archive with current population
                    self._add_to_archive(population)
                    
                    
                # Reproduction: Keep best half based on health and duplicate
                # Sort population by health (lower health is better for minimization)
                sorted_population = sorted(population, key=lambda x: x.health)
                best_half = sorted_population[:search_agents_no // 2]
                
                # Create new population by duplicating best half
                new_population = []
                for bacterium in best_half:
                    new_population.append(bacterium.copy())
                for bacterium in best_half:
                    new_population.append(bacterium.copy())
                
                population = new_population
            
            # Elimination-dispersal: Random elimination of some bacteria
            for i, bacterium in enumerate(population):
                if np.random.random() < self.elimination_prob:
                    # Randomly disperse this bacterium
                    new_position = np.random.uniform(self.lb, self.ub, self.dim)
                    new_fitness = self.objective_func(new_position)
                    population[i] = BacteriaMultiMember(new_position, new_fitness)

            # Store archive state for history
            history_archive.append([member.copy() for x`` in self.archive])
            # Call callback for progress tracking
            self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)

        # Final archive update
        self._add_to_archive(population)
        
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver
        self._end_step_solver()
        
        return history_archive, self.archive
    
    def _get_normalized_costs(self, population: List[BacteriaMultiMember]) -> np.ndarray:
        """
        Get normalized cost matrix from population for aggregation methods.
        
        Args:
            population (List[BacteriaMultiMember]): Population to normalize
            
        Returns:
            np.ndarray: Normalized cost matrix
        """
        costs = self._get_fitness(population)
        return normalized_values(costs)
