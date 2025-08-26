import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
import random
from ._general import sort_population

class GravitationalSearchOptimizer(Solver):
    """
    Gravitational Search Algorithm (GSA) Optimizer
    
    Reference: Rashedi, Esmat, Hossein Nezamabadi-Pour, and Saeid Saryazdi. "GSA: a gravitational search algorithm." 
               Information sciences 179.13 (2009): 2232-2248.
    
    GSA is a population-based optimization algorithm inspired by the law of gravity and mass interactions.
    Each solution is considered as a mass, and their interactions are governed by gravitational forces.
    
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
        - elitist_check: Whether to use elitist strategy (default: True)
        - r_power: Power parameter for distance calculation (default: 1)
        - g0: Initial gravitational constant (default: 100)
        - alpha: Decay parameter for gravitational constant (default: 20)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize)
        
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set solver name
        self.name_solver = "Gravitational Search Optimizer"
        
        # Algorithm-specific parameters with defaults
        self.elitist_check = kwargs.get('elitist_check', True)
        self.r_power = kwargs.get('r_power', 1)
        self.g0 = kwargs.get('g0', 100)
        self.alpha = kwargs.get('alpha', 20)
        
        # Initialize velocity storage
        self.velocities = None
    
    def _mass_calculation(self, fitness_values: np.ndarray) -> np.ndarray:
        """
        Calculate masses for all agents based on their fitness.
        
        Args:
            fitness_values (np.ndarray): Array of fitness values for all agents
            
        Returns:
            np.ndarray: Normalized mass values for all agents
        """
        pop_size = len(fitness_values)
        
        if self.maximize:
            # For maximization, higher fitness is better
            best = max(fitness_values)
            worst = min(fitness_values)
        else:
            # For minimization, lower fitness is better
            best = min(fitness_values)
            worst = max(fitness_values)
        
        if best == worst:
            # All agents have same fitness
            return np.ones(pop_size) / pop_size
        
        # Calculate raw masses
        masses = np.zeros(pop_size)
        for i in range(pop_size):
            masses[i] = (fitness_values[i] - worst) / (best - worst)
        
        # Normalize masses
        mass_sum = np.sum(masses)
        if mass_sum > 0:
            masses = masses / mass_sum
        
        return masses
    
    def _gravitational_constant(self, iteration: int, max_iter: int) -> float:
        """
        Calculate gravitational constant for current iteration.
        
        Args:
            iteration (int): Current iteration number
            max_iter (int): Maximum number of iterations
            
        Returns:
            float: Gravitational constant value
        """
        gimd = np.exp(-self.alpha * float(iteration) / max_iter)
        return self.g0 * gimd
    
    def _gravitational_field(self, population: List[Member], masses: np.ndarray, 
                           iteration: int, max_iter: int, g: float) -> np.ndarray:
        """
        Calculate gravitational forces and accelerations for all agents.
        
        Args:
            population (List[Member]): Current population
            masses (np.ndarray): Mass values for all agents
            iteration (int): Current iteration number
            max_iter (int): Maximum number of iterations
            g (float): Gravitational constant
            
        Returns:
            np.ndarray: Acceleration matrix for all agents
        """
        pop_size = len(population)
        positions = np.array([member.position for member in population])
        
        # Determine kbest (number of best agents to consider)
        final_percent = 2  # Minimum percentage of best agents
        if self.elitist_check:
            kbest_percent = final_percent + (1 - iteration / max_iter) * (100 - final_percent)
            kbest = round(pop_size * kbest_percent / 100)
        else:
            kbest = pop_size
        
        kbest = max(1, min(kbest, pop_size))  # Ensure kbest is within valid range
        
        # Sort agents by fitness (best first for maximization, worst first for minimization)
        if self.maximize:
            sorted_indices = np.argsort([-member.fitness for member in population])
        else:
            sorted_indices = np.argsort([member.fitness for member in population])
        
        # Initialize force matrix
        forces = np.zeros((pop_size, self.dim))
        
        for i in range(pop_size):
            for j in range(kbest):
                agent_idx = sorted_indices[j]
                if agent_idx == i:
                    continue  # Skip self-interaction
                
                # Calculate Euclidean distance
                distance = np.linalg.norm(positions[i] - positions[agent_idx])
                
                # Avoid division by zero
                if distance < 1e-10:
                    distance = 1e-10
                
                # Calculate force components
                for d in range(self.dim):
                    rand_val = random.random()
                    force_component = rand_val * masses[agent_idx] * \
                                    (positions[agent_idx, d] - positions[i, d]) / \
                                    (distance ** self.r_power + np.finfo(float).eps)
                    forces[i, d] += force_component
        
        # Calculate accelerations
        accelerations = forces * g
        return accelerations
    
    def _update_positions(self, population: List[Member], accelerations: np.ndarray) -> Tuple[List[Member], np.ndarray]:
        """
        Update positions and velocities of all agents.
        
        Args:
            population (List[Member]): Current population
            accelerations (np.ndarray): Acceleration matrix
            
        Returns:
            Tuple[List[Member], np.ndarray]: Updated population and velocities
        """
        pop_size = len(population)
        positions = np.array([member.position for member in population])
        
        # Initialize velocities if not already done
        if self.velocities is None:
            self.velocities = np.zeros((pop_size, self.dim))
        
        # Update velocities and positions
        for i in range(pop_size):
            for d in range(self.dim):
                rand_val = random.random()
                self.velocities[i, d] = rand_val * self.velocities[i, d] + accelerations[i, d]
                positions[i, d] = positions[i, d] + self.velocities[i, d]
        
        # Ensure positions stay within bounds
        positions = np.clip(positions, self.lb, self.ub)
        
        # Update population with new positions and recalculate fitness
        new_population = []
        for i in range(pop_size):
            new_position = positions[i]
            fitness = self.objective_func(new_position)
            new_population.append(Member(new_position, fitness))
        
        return new_population, self.velocities
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Main optimization method for Gravitational Search Algorithm.
        
        Parameters:
        -----------
        search_agents_no : int
            Number of search agents (population size)
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List, Member]
            History of best solutions and the best solution found
        """
        # Initialize storage variables
        history_step_solver = []
        
        # Initialize population
        population = self._init_population(search_agents_no)
        
        # Initialize best solution
        sorted_population, _ = self._sort_population(population)
        best_solution = sorted_population[0].copy()
        
        # Initialize velocities
        self.velocities = np.zeros((search_agents_no, self.dim))
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iteration in range(max_iter):
            # Extract fitness values for mass calculation
            fitness_values = np.array([member.fitness for member in population])
            
            # Calculate masses
            masses = self._mass_calculation(fitness_values)
            
            # Calculate gravitational constant
            g = self._gravitational_constant(iteration, max_iter)
            
            # Calculate gravitational field and accelerations
            accelerations = self._gravitational_field(population, masses, iteration, max_iter, g)
            
            # Update positions
            population, self.velocities = self._update_positions(population, accelerations)
            
            # Update best solution
            sorted_population, _ = self._sort_population(population)
            current_best = sorted_population[0]
            if self._is_better(current_best, best_solution):
                best_solution = current_best.copy()
            
            # Save history
            history_step_solver.append(best_solution.copy())
            
            # Call callback for progress tracking
            self._callbacks(iteration, max_iter, best_solution)
        
        # End solver
        self.history_step_solver = history_step_solver
        self.best_solver = best_solution
        self._end_step_solver()
        
        return history_step_solver, best_solution
    
    def _sort_population(self, population):
        """
        Sort the population based on fitness.
        """
        return sort_population(population, self.maximize)