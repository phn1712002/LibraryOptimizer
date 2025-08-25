import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from utils.general import sort_population


class DingoOptimizer(Solver):
    """
    Dingo Optimization Algorithm (DOA) implementation.
    
    A bio-inspired optimization method inspired by dingoes hunting strategies.
    
    Parameters:
    -----------
    objective_func : Callable
        Objective function to optimize
    lb : Union[float, np.ndarray]
        Lower bounds for variables
    ub : Union[float, np.ndarray]
        Upper bounds for variables  
    dim : int
        Problem dimension
    maximize : bool, optional
        Optimization direction, default is True (maximize)
    **kwargs
        Additional algorithm parameters:
        - p: Hunting probability (default: 0.5)
        - q: Group attack probability (default: 0.7)
        - na_min: Minimum number of attacking dingoes (default: 2)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set solver name
        self.name_solver = "Dingo Optimizer"
        
        # Set default parameters
        self.p = kwargs.get('p', 0.5)  # Hunting probability
        self.q = kwargs.get('q', 0.7)  # Group attack probability
        self.na_min = kwargs.get('na_min', 2)  # Minimum attacking dingoes
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Main optimization method for Dingo Optimization Algorithm.
        
        Args:
            search_agents_no: Number of search agents (dingoes)
            max_iter: Maximum number of iterations
            
        Returns:
            Tuple containing optimization history and best solution
        """
        # Initialize population
        population = self._init_population(search_agents_no)
        
        # Initialize best solution
        sorted_population, _ = self._sort_population(population)
        best_solution = sorted_population[0].copy()
        
        # Initialize history
        history_step_solver = []
        
        # Start solver (show progress bar)
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Calculate number of attacking dingoes for this iteration
            na = self._calculate_attacking_dingoes(search_agents_no)
            
            # Update each search agent
            for i in range(search_agents_no):
                # Generate new position based on hunting strategy
                new_position = self._update_position(population, i, na, iter, max_iter)
                
                # Ensure positions stay within bounds
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Evaluate new fitness
                new_fitness = self.objective_func(new_position)
                
                # Compare and update if better
                if self._is_better(Member(new_position, new_fitness), population[i]):
                    population[i].position = new_position
                    population[i].fitness = new_fitness
            
            # Update best solution
            sorted_population, _ = self._sort_population(population)
            current_best = sorted_population[0]
            if self._is_better(current_best, best_solution):
                best_solution = current_best.copy()
            
            # Save history
            history_step_solver.append(best_solution.copy())
            
            # Call callback
            self._callbacks(iter, max_iter, best_solution)
        
        # End solver
        self.history_step_solver = history_step_solver
        self.best_solver = best_solution
        self._end_step_solver()
        
        return history_step_solver, best_solution
    
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
        if np.random.random() < self.p:  # Hunting strategy
            if np.random.random() < self.q:  # Group attack
                # Strategy 1: Group attack
                sumatory = self._group_attack(population, na, current_idx)
                beta1 = -2 + 4 * np.random.random()  # -2 < beta1 < 2
                new_position = beta1 * sumatory - self.best_solver.position
            else:  # Persecution
                # Strategy 2: Persecution
                r1 = np.random.randint(0, len(population))
                beta1 = -2 + 4 * np.random.random()  # -2 < beta1 < 2
                beta2 = -1 + 2 * np.random.random()  # -1 < beta2 < 1
                new_position = (self.best_solver.position + 
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
            new_position = (self.best_solver.position + 
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
        fitness_values = [member.fitness for member in population]
        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values)
        
        if max_fitness == min_fitness:
            return 1.0  # All have same fitness
        
        current_fitness = population[current_idx].fitness
        if self.maximize:
            return (max_fitness - current_fitness) / (max_fitness - min_fitness)
        else:
            return (current_fitness - min_fitness) / (max_fitness - min_fitness)

    def _sort_population(self, population):
        return sort_population(population, self.maximize)