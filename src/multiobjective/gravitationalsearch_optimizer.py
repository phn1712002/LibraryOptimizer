import numpy as np
from typing import Callable, Union, Tuple, List
from .._core import Solver, Member
from ._core import MultiObjectiveSolver, MultiObjectiveMember
import random
import math


class GravitationalSearchMultiMember(MultiObjectiveMember):
    """
    Custom member class for Gravitational Search Algorithm with multi-objective support.
    Extends MultiObjectiveMember to include velocity information.
    """
    def __init__(self, position: np.ndarray, fitness: np.ndarray, velocity: np.ndarray = None):
        super().__init__(position, fitness)
        self.velocity = velocity if velocity is not None else np.zeros_like(position)
    
    def copy(self):
        new_member = GravitationalSearchMultiMember(
            self.position.copy(), 
            self.multi_fitness.copy(),
            self.velocity.copy()
        )
        new_member.dominated = self.dominated
        new_member.grid_index = self.grid_index
        new_member.grid_sub_index = self.grid_sub_index
        return new_member


class MultiObjectiveGravitationalSearchOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Gravitational Search Algorithm (GSA) Optimizer
    
    Reference: Rashedi, Esmat, Hossein Nezamabadi-Pour, and Saeid Saryazdi. "GSA: a gravitational search algorithm." 
               Information sciences 179.13 (2009): 2232-2248.
    
    Multi-objective version of GSA that maintains an archive of non-dominated solutions
    and uses grid-based selection for leader guidance.
    
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
    **kwargs
        Additional algorithm parameters:
        - elitist_check: Whether to use elitist strategy (default: True)
        - r_power: Power parameter for distance calculation (default: 1)
        - g0: Initial gravitational constant (default: 100)
        - alpha: Decay parameter for gravitational constant (default: 20)
        - archive_size: Size of external archive (default: 100)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize:bool=True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Gravitational Search Optimizer"
        
        # Algorithm-specific parameters with defaults
        self.elitist_check = kwargs.get('elitist_check', True)
        self.r_power = kwargs.get('r_power', 1)
        self.g0 = kwargs.get('g0', 100)
        self.alpha = kwargs.get('alpha', 20)
        
        # Initialize velocity storage
        self.velocities = None
    
    def _mass_calculation(self, fitness_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate masses for all agents based on their multi-objective fitness.
        
        Args:
            fitness_matrix (np.ndarray): Matrix of fitness values for all agents (pop_size x n_objectives)
            
        Returns:
            np.ndarray: Normalized mass values for all agents
        """
        pop_size = fitness_matrix.shape[0]
        
        # For multi-objective, we need to calculate a composite fitness
        # Use the sum of normalized objectives as a simple approach
        normalized_fitness = np.zeros(pop_size)
        
        for obj_idx in range(self.n_objectives):
            obj_values = fitness_matrix[:, obj_idx]
            min_val = np.min(obj_values)
            max_val = np.max(obj_values)
            
            if max_val != min_val:
                normalized_obj = (obj_values - min_val) / (max_val - min_val)
            else:
                normalized_obj = np.ones(pop_size)
            
            normalized_fitness += normalized_obj
        
        # Calculate masses based on composite fitness (lower is better for minimization)
        best = np.min(normalized_fitness)
        worst = np.max(normalized_fitness)
        
        if best == worst:
            return np.ones(pop_size) / pop_size
        
        masses = (normalized_fitness - worst) / (best - worst)
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
    
    def _gravitational_field(self, population: List[GravitationalSearchMultiMember], 
                           masses: np.ndarray, iteration: int, max_iter: int, 
                           g: float) -> np.ndarray:
        """
        Calculate gravitational forces and accelerations for all agents.
        
        Args:
            population (List[GravitationalSearchMultiMember]): Current population
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
        
        # Sort agents by their composite fitness (for leader selection)
        composite_fitness = np.array([np.sum(member.multi_fitness) for member in population])
        sorted_indices = np.argsort(composite_fitness)  # Lower is better for minimization
        
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
    
    def _update_positions(self, population: List[GravitationalSearchMultiMember], 
                         accelerations: np.ndarray) -> Tuple[List[GravitationalSearchMultiMember], np.ndarray]:
        """
        Update positions and velocities of all agents.
        
        Args:
            population (List[GravitationalSearchMultiMember]): Current population
            accelerations (np.ndarray): Acceleration matrix
            
        Returns:
            Tuple[List[GravitationalSearchMultiMember], np.ndarray]: Updated population and velocities
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
            new_population.append(GravitationalSearchMultiMember(
                new_position, fitness, self.velocities[i].copy()
            ))
        
        return new_population, self.velocities
    
    def _init_population(self, search_agents_no: int) -> List[GravitationalSearchMultiMember]:
        """
        Initialize population with custom member class.
        
        Args:
            search_agents_no (int): Number of search agents
            
        Returns:
            List[GravitationalSearchMultiMember]: Initialized population
        """
        population = []
        for _ in range(search_agents_no):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            fitness = self.objective_func(position)
            population.append(GravitationalSearchMultiMember(position, fitness))
        return population
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[GravitationalSearchMultiMember]]:
        """
        Main optimization method for Multi-Objective Gravitational Search Algorithm.
        
        Parameters:
        -----------
        search_agents_no : int
            Number of search agents (population size)
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List, List[GravitationalSearchMultiMember]]
            History of archive states and the final archive of non-dominated solutions
        """
        # Initialize storage
        history_archive = []
        
        # Initialize population
        population = self._init_population(search_agents_no)
        
        # Initialize velocities
        self.velocities = np.zeros((search_agents_no, self.dim))
        
        # Initialize archive with non-dominated solutions
        self._determine_domination(population)
        non_dominated = self._get_non_dominated_particles(population)
        self.archive.extend(non_dominated)
        
        # Initialize grid for archive management
        costs = self._get_fitness(self.archive)
        if costs.size > 0:
            self.grid = self._create_hypercubes(costs)
            for particle in self.archive:
                particle.grid_index, particle.grid_sub_index = self._get_grid_index(particle)
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iteration in range(max_iter):
            # Extract fitness values for mass calculation
            fitness_matrix = np.array([member.multi_fitness for member in population])
            
            # Calculate masses
            masses = self._mass_calculation(fitness_matrix)
            
            # Calculate gravitational constant
            g = self._gravitational_constant(iteration, max_iter)
            
            # Calculate gravitational field and accelerations
            accelerations = self._gravitational_field(population, masses, iteration, max_iter, g)
            
            # Update positions
            population, self.velocities = self._update_positions(population, accelerations)
            
            # Update archive with current population
            self._add_to_archive(population)
            
            # Trim archive if it exceeds maximum size
            if len(self.archive) > self.archive_size:
                self._trim_archive()
            
            # Store archive state for history
            history_archive.append([member.copy() for member in self.archive])
            
            # Update progress
            best_archive_member = self.archive[0] if self.archive else None
            self._callbacks(iteration, max_iter, best_archive_member)
        
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver
        self._end_step_solver()
        
        return history_archive, self.archive
