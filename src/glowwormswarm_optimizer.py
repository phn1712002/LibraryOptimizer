import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from ._general import sort_population, roulette_wheel_selection

class GlowwormSwarmOptimizer(Solver):
    """
    Glowworm Swarm Optimization (GSO) Algorithm.
    
    GSO is a nature-inspired metaheuristic optimization algorithm that mimics
    the behavior of glowworms (fireflies) that use bioluminescence to attract
    mates and prey. Each glowworm carries a luminescent quantity called luciferin,
    which they use to communicate with other glowworms in their neighborhood.
    
    Key features:
    - Luciferin-based communication
    - Dynamic neighborhood topology
    - Adaptive decision range
    - Probabilistic movement towards brighter neighbors
    
    References:
        Krishnanand, K. N., & Ghose, D. (2009). Glowworm swarm optimization: 
        a new method for optimizing multi-modal functions. International Journal 
        of Computational Intelligence Studies, 1(1), 93-119.
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        """
        Initialize the Glowworm Swarm Optimizer.
        
        Args:
            objective_func (Callable): Objective function to optimize
            lb (Union[float, np.ndarray]): Lower bounds of search space
            ub (Union[float, np.ndarray]): Upper bounds of search space
            dim (int): Number of dimensions in the problem
            maximize (bool): Whether to maximize (True) or minimize (False) objective
            **kwargs: Additional algorithm parameters including:
                - L0: Initial luciferin value (default: 5.0)
                - r0: Initial decision range (default: 3.0)
                - rho: Luciferin decay constant (default: 0.4)
                - gamma: Luciferin enhancement constant (default: 0.6)
                - beta: Decision range update constant (default: 0.08)
                - s: Step size for movement (default: 0.6)
                - rs: Maximum sensing range (default: 10.0)
                - nt: Desired number of neighbors (default: 10)
        """
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Store additional parameters for later use
        self.kwargs = kwargs
        self.name_solver = "Glowworm Swarm Optimizer"
        
        # Algorithm-specific parameters with defaults
        self.L0 = kwargs.get('L0', 5.0)  # Initial luciferin
        self.r0 = kwargs.get('r0', 3.0)  # Initial decision range
        self.rho = kwargs.get('rho', 0.4)  # Luciferin decay constant
        self.gamma = kwargs.get('gamma', 0.6)  # Luciferin enhancement constant
        self.beta = kwargs.get('beta', 0.08)  # Decision range update constant
        self.s = kwargs.get('s', 0.6)  # Step size for movement
        self.rs = kwargs.get('rs', 10.0)  # Maximum sensing range
        self.nt = kwargs.get('nt', 10)  # Desired number of neighbors
        
        # Internal state variables
        self.luciferin = None
        self.decision_range = None
        
    def _euclidean_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two positions.
        
        Args:
            pos1 (np.ndarray): First position vector
            pos2 (np.ndarray): Second position vector
            
        Returns:
            float: Euclidean distance between the two positions
        """
        return np.sqrt(np.sum((pos1 - pos2) ** 2))
    
    def _get_neighbors(self, current_idx: int, population: List[Member]) -> List[int]:
        """
        Get indices of neighbors within decision range that have higher luciferin.
        
        Args:
            current_idx (int): Index of current glowworm
            population (List[Member]): List of all glowworms
            
        Returns:
            List[int]: Indices of valid neighbors
        """
        current_pos = population[current_idx].position
        current_luciferin = self.luciferin[current_idx]
        
        neighbors = []
        for j in range(len(population)):
            if j == current_idx:
                continue
                
            distance = self._euclidean_distance(current_pos, population[j].position)
            if distance < self.decision_range[current_idx] and self.luciferin[j] > current_luciferin:
                neighbors.append(j)
                
        return neighbors
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Execute the Glowworm Swarm Optimization Algorithm.
        
        The algorithm consists of three main phases:
        1. Luciferin update: Each glowworm updates its luciferin based on fitness
        2. Movement: Each glowworm probabilistically moves towards brighter neighbors
        3. Decision range update: Each glowworm adjusts its sensing range
        
        Args:
            search_agents_no (int): Number of glowworms in the swarm
            max_iter (int): Maximum number of iterations for optimization
            
        Returns:
            Tuple[List, Member]: A tuple containing:
                - history_step_solver: List of best solutions at each iteration
                - best_solver: Best solution found overall
        """
        # Initialize the population of glowworms
        population = self._init_population(search_agents_no)
        
        # Initialize luciferin and decision range
        self.luciferin = np.full(search_agents_no, self.L0)
        self.decision_range = np.full(search_agents_no, self.r0)
        
        # Initialize storage variables
        history_step_solver = []
        best_solver = self.best_solver
        
        # Start solver execution
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Update luciferin for all glowworms
            fitness_values = np.array([member.fitness for member in population])
            
            # Convert fitness to luciferin (handle maximization/minimization)
            if self.maximize:
                # For maximization: higher fitness = higher luciferin
                luciferin_update = self.gamma * fitness_values
            else:
                # For minimization: lower fitness = higher luciferin
                # We need to invert the fitness for minimization
                max_fitness = np.max(fitness_values)
                luciferin_update = self.gamma * (max_fitness - fitness_values + 1e-10)
            
            self.luciferin = (1 - self.rho) * self.luciferin + luciferin_update
            
            # Find the best glowworm (highest luciferin)
            best_idx = np.argmax(self.luciferin)
            current_best = population[best_idx].copy()
            
            # Update best solution if better
            if self._is_better(current_best, best_solver):
                best_solver = current_best.copy()
            
            # Move each glowworm
            for i in range(search_agents_no):
                # Get neighbors within decision range that have higher luciferin
                neighbors = self._get_neighbors(i, population)
                
                if not neighbors:
                    # No neighbors found, stay in current position
                    continue
                
                # Calculate probabilities for movement towards each neighbor
                neighbor_luciferin = self.luciferin[neighbors]
                current_luciferin = self.luciferin[i]
                
                # Probability proportional to luciferin difference
                probabilities = (neighbor_luciferin - current_luciferin) / np.sum(neighbor_luciferin - current_luciferin)
                
                # Select a neighbor using roulette wheel selection
                selected_neighbor_idx = self._roulette_wheel_selection(probabilities)
                selected_neighbor = neighbors[selected_neighbor_idx]
                
                # Move towards the selected neighbor
                current_pos = population[i].position
                neighbor_pos = population[selected_neighbor].position
                
                # Calculate direction vector
                direction = neighbor_pos - current_pos
                distance = self._euclidean_distance(current_pos, neighbor_pos)
                
                if distance > 0:
                    # Normalize direction and move with step size s
                    direction_normalized = direction / distance
                    new_position = current_pos + self.s * direction_normalized
                else:
                    # If distance is zero, add small random perturbation
                    new_position = current_pos + self.s * np.random.uniform(-0.1, 0.1, self.dim)
                
                # Ensure positions stay within bounds
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Update glowworm position and fitness
                population[i].position = new_position
                population[i].fitness = self.objective_func(new_position)
                
                # Update decision range based on number of neighbors
                neighbor_count = len(neighbors)
                self.decision_range[i] = min(self.rs, max(0, self.decision_range[i] + self.beta * (self.nt - neighbor_count)))
            
            # Store the best solution at this iteration
            history_step_solver.append(best_solver.copy())
            
            # Update progress
            self._callbacks(iter, max_iter, best_solver)
        
        # Final evaluation and storage
        self.history_step_solver = history_step_solver
        self.best_solver = best_solver
        
        # End solver execution
        self._end_step_solver()
        
        return history_step_solver, best_solver
    
    def _sort_population(self, population):
        """
        Sort the population based on fitness.
        """
        return sort_population(population, self.maximize)
    
    def _roulette_wheel_selection(self, probabilities):
        """Perform roulette wheel selection (fitness proportionate selection)."""
        return roulette_wheel_selection(probabilities)
