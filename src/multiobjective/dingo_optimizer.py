import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember

class MultiObjectiveDingoOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Dingo Optimization Algorithm (DOA)
    
    This algorithm extends the standard DOA for multi-objective optimization
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
        - p: Hunting probability (default: 0.5)
        - q: Group attack probability (default: 0.7)
        - na_min: Minimum number of attacking dingoes (default: 2)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Dingo Optimizer"
        
        # Algorithm-specific parameters
        self.p = kwargs.get('p', 0.5)  # Hunting probability
        self.q = kwargs.get('q', 0.7)  # Group attack probability
        self.na_min = kwargs.get('na_min', 2)  # Minimum attacking dingoes

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Main optimization method for multi-objective DOA
        
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
            # Calculate number of attacking dingoes for this iteration
            na = self._calculate_attacking_dingoes(search_agents_no)
            
            # Update all search agents
            for i, dingo in enumerate(population):
                # Generate new position based on hunting strategy
                new_position = self._update_position(population, i, na, iter, max_iter)
                
                # Ensure positions stay within bounds
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Update dingo position and fitness
                population[i].position = new_position
                population[i].multi_fitness = self.objective_func(new_position)
            
            # Update archive with current population
            self._add_to_archive(population)
            
            # Store archive state for history
            history_archive.append([dingo.copy() for dingo in self.archive])
            
            # Update progress
            self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
        
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver
        self._end_step_solver()
        
        return history_archive, self.archive
    
    def _calculate_attacking_dingoes(self, search_agents_no: int) -> int:
        """
        Calculate number of dingoes that will attack in current iteration.
        
        Args:
            search_agents_no: Total number of search agents
            
        Returns:
            Number of attacking dingoes
        """
        na_end = search_agents_no / self.na_min
        na = round(self.na_min + (na_end - self.na_min) * np.random.random())
        return max(self.na_min, min(int(na), search_agents_no))
    
    def _update_position(self, population: List, current_idx: int, na: int, 
                        iter: int, max_iter: int) -> np.ndarray:
        """
        Update position of a search agent based on hunting strategies.
        
        Args:
            population: Current population
            current_idx: Index of current search agent
            na: Number of attacking dingoes
            iter: Current iteration
            max_iter: Maximum iterations
            
        Returns:
            New position vector
        """
        # Select leader from archive for guidance
        leader = self._select_leader()
        if leader is None:
            # If no leader in archive, use random member from population
            leader_idx = np.random.randint(0, len(population))
            leader = population[leader_idx]
        
        if np.random.random() < self.p:  # Hunting strategy
            if np.random.random() < self.q:  # Group attack
                # Strategy 1: Group attack
                sumatory = self._group_attack(population, na, current_idx)
                beta1 = -2 + 4 * np.random.random()  # -2 < beta1 < 2
                new_position = beta1 * sumatory - leader.position
            else:  # Persecution
                # Strategy 2: Persecution
                r1 = np.random.randint(0, len(population))
                beta1 = -2 + 4 * np.random.random()  # -2 < beta1 < 2
                beta2 = -1 + 2 * np.random.random()  # -1 < beta2 < 1
                new_position = (leader.position + 
                               beta1 * np.exp(beta2) * 
                               (population[r1].position - population[current_idx].position))
        else:  # Scavenger strategy
            # Strategy 3: Scavenging
            r1 = np.random.randint(0, len(population))
            beta2 = -1 + 2 * np.random.random()  # -1 < beta2 < 1
            binary_val = 0 if np.random.random() < 0.5 else 1
            new_position = (np.exp(beta2) * population[r1].position - 
                           ((-1) ** binary_val) * population[current_idx].position) / 2
        
        # Apply survival strategy if needed
        survival_rate = self._calculate_survival_rate(population, current_idx)
        if survival_rate <= 0.3:
            # Strategy 4: Survival
            r1, r2 = self._get_two_distinct_indices(len(population), current_idx)
            binary_val = 0 if np.random.random() < 0.5 else 1
            new_position = (leader.position + 
                           (population[r1].position - 
                            ((-1) ** binary_val) * population[r2].position) / 2)
        
        return new_position
    
    def _group_attack(self, population: List, na: int, current_idx: int) -> np.ndarray:
        """
        Perform group attack strategy.
        
        Args:
            population: Current population
            na: Number of attacking dingoes
            current_idx: Index of current search agent
            
        Returns:
            Sumatory vector for group attack
        """
        attack_indices = self._get_attack_indices(len(population), na, current_idx)
        sumatory = np.zeros(self.dim)
        
        for idx in attack_indices:
            sumatory += population[idx].position - population[current_idx].position
        
        return sumatory / na
    
    def _get_attack_indices(self, population_size: int, na: int, exclude_idx: int) -> List[int]:
        """
        Get indices of dingoes that will participate in group attack.
        
        Args:
            population_size: Total population size
            na: Number of attacking dingoes
            exclude_idx: Index to exclude (current dingo)
            
        Returns:
            List of attack indices
        """
        indices = []
        while len(indices) < na:
            idx = np.random.randint(0, population_size)
            if idx != exclude_idx and idx not in indices:
                indices.append(idx)
        return indices
    
    def _get_two_distinct_indices(self, population_size: int, exclude_idx: int) -> Tuple[int, int]:
        """
        Get two distinct random indices excluding the specified index.
        
        Args:
            population_size: Total population size
            exclude_idx: Index to exclude
            
        Returns:
            Tuple of two distinct indices
        """
        while True:
            r1 = np.random.randint(0, population_size)
            r2 = np.random.randint(0, population_size)
            if r1 != r2 and r1 != exclude_idx and r2 != exclude_idx:
                return r1, r2
    
    def _calculate_survival_rate(self, population: List, current_idx: int) -> float:
        """
        Calculate survival rate for a search agent.
        
        Args:
            population: Current population
            current_idx: Index of current search agent
            
        Returns:
            Survival rate (0 to 1)
        """
        # For multi-objective, we need a different approach
        # Use the concept of dominance count instead of single fitness
        
        # Count how many solutions dominate the current solution
        dominated_count = 0
        current_solution = population[current_idx]
        
        for other_solution in population:
            if other_solution != current_solution and self._dominates(other_solution, current_solution):
                dominated_count += 1
        
        # Survival rate is inversely proportional to dominance count
        survival_rate = 1.0 - (dominated_count / len(population))
        
        return max(0.0, min(1.0, survival_rate))
