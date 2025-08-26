import numpy as np
from typing import Callable, Union, Tuple, List
from .._core import Member
from ._core import MultiObjectiveSolver, MultiObjectiveMember
from .._general import roulette_wheel_selection

class GlowwormMultiMember(MultiObjectiveMember):
    """
    Custom member class for Multi-Objective Glowworm Swarm Optimization.
    
    Extends MultiObjectiveMember to include algorithm-specific attributes
    for tracking luciferin and decision range in multi-objective context.
    """
    
    def __init__(self, position: np.ndarray, fitness: np.ndarray, 
                 luciferin: float = 5.0, decision_range: float = 3.0):
        """
        Initialize a GlowwormMultiMember with position, fitness, and algorithm-specific attributes.
        
        Args:
            position (np.ndarray): Position vector in search space
            fitness (np.ndarray): Multi-objective fitness values
            luciferin (float): Luciferin value for the glowworm (default: 5.0)
            decision_range (float): Decision range for the glowworm (default: 3.0)
        """
        super().__init__(position, fitness)
        self.luciferin = luciferin
        self.decision_range = decision_range
    
    def copy(self):
        """
        Create a deep copy of the GlowwormMultiMember.
        
        Returns:
            GlowwormMultiMember: A new member with copied attributes
        """
        new_member = GlowwormMultiMember(
            self.position.copy(), 
            self.multi_fitness.copy(),
            self.luciferin,
            self.decision_range
        )
        new_member.dominated = self.dominated
        new_member.grid_index = self.grid_index
        new_member.grid_sub_index = self.grid_sub_index
        return new_member
    
    def __str__(self):
        """
        String representation of the GlowwormMultiMember.
        
        Returns:
            str: Formatted string showing position, fitness, and algorithm attributes
        """
        return (f"Position: {self.position} - Fitness: {self.multi_fitness} - "
                f"Luciferin: {self.luciferin:.3f} - Decision Range: {self.decision_range:.3f}")

class MultiObjectiveGlowwormSwarmOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Glowworm Swarm Optimization (GSO) Algorithm.
    
    This is the multi-objective version of the GSO algorithm that extends
    the single-objective implementation to handle multiple objectives using
    Pareto dominance and archive management.
    
    Key features:
    - Pareto dominance-based selection
    - Archive management for non-dominated solutions
    - Grid-based diversity maintenance
    - Luciferin-based communication adapted for multi-objective optimization
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        """
        Initialize the Multi-Objective Glowworm Swarm Optimizer.
        
        Args:
            objective_func (Callable): Multi-objective function to optimize
            lb (Union[float, np.ndarray]): Lower bounds of search space
            ub (Union[float, np.ndarray]): Upper bounds of search space
            dim (int): Number of dimensions in the problem
            maximize (bool): Whether to maximize (True) or minimize (False) objectives
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
        self.name_solver = "Multi-Objective Glowworm Swarm Optimizer"
        
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
    
    def _get_neighbors(self, current_idx: int, population: List[GlowwormMultiMember]) -> List[int]:
        """
        Get indices of neighbors within decision range that dominate the current glowworm.
        
        Args:
            current_idx (int): Index of current glowworm
            population (List[GlowwormMultiMember]): List of all glowworms
            
        Returns:
            List[int]: Indices of valid neighbors that dominate the current glowworm
        """
        current_member = population[current_idx]
        current_pos = current_member.position
        current_luciferin = self.luciferin[current_idx]
        
        neighbors = []
        for j in range(len(population)):
            if j == current_idx:
                continue
                
            neighbor_member = population[j]
            distance = self._euclidean_distance(current_pos, neighbor_member.position)
            
            # Check if neighbor is within decision range and dominates current glowworm
            if (distance < self.decision_range[current_idx] and 
                self._dominates(neighbor_member, current_member) and
                self.luciferin[j] > current_luciferin):
                neighbors.append(j)
                
        return neighbors
    
    def _calculate_luciferin(self, member: GlowwormMultiMember) -> float:
        """
        Calculate luciferin value for a member based on its multi-objective fitness.
        
        For multi-objective optimization, luciferin is calculated based on
        the member's position in the Pareto front and its diversity contribution.
        
        Args:
            member (GlowwormMultiMember): The member to calculate luciferin for
            
        Returns:
            float: Luciferin value
        """
        # Simple approach: use the average of normalized fitness values
        # More sophisticated approaches could consider crowding distance, etc.
        if self.maximize:
            # For maximization: higher fitness values are better
            return np.mean(member.multi_fitness)
        else:
            # For minimization: lower fitness values are better
            # We invert the values for consistency
            max_possible = np.max([m.multi_fitness for m in self.archive], axis=0) if self.archive else member.multi_fitness
            normalized = 1.0 / (member.multi_fitness + 1e-10)
            return np.mean(normalized)
    
    def _init_population(self, search_agents_no) -> List[GlowwormMultiMember]:
        """
        Initialize multi-objective population with custom member class.
        
        Args:
            search_agents_no (int): Number of glowworms to initialize
            
        Returns:
            List[GlowwormMultiMember]: List of initialized glowworms
        """
        population = []
        for _ in range(search_agents_no):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            fitness = self.objective_func(position)
            population.append(GlowwormMultiMember(position, fitness, self.L0, self.r0))
        return population
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[GlowwormMultiMember]]:
        """
        Execute the Multi-Objective Glowworm Swarm Optimization Algorithm.
        
        The algorithm extends the single-objective GSO with:
        - Pareto dominance for neighbor selection
        - Archive management for non-dominated solutions
        - Grid-based diversity maintenance
        
        Args:
            search_agents_no (int): Number of glowworms in the swarm
            max_iter (int): Maximum number of iterations for optimization
            
        Returns:
            Tuple[List, List[GlowwormMultiMember]]: A tuple containing:
                - history_archive: List of archive states at each iteration
                - final_archive: Final archive of non-dominated solutions
        """
        # Initialize the population of glowworms
        population = self._init_population(search_agents_no)
        
        # Initialize luciferin and decision range arrays
        self.luciferin = np.array([self.L0] * search_agents_no)
        self.decision_range = np.array([self.r0] * search_agents_no)
        
        # Initialize archive with non-dominated solutions from initial population
        self._determine_domination(population)
        non_dominated = self._get_non_dominated_particles(population)
        self.archive.extend(non_dominated)
        
        # Build initial grid for archive
        costs = self._get_fitness(self.archive)
        if costs.size > 0:
            self.grid = self._create_hypercubes(costs)
            for particle in self.archive:
                particle.grid_index, particle.grid_sub_index = self._get_grid_index(particle)
        
        # Initialize history storage
        history_archive = []
        
        # Start solver execution
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Update luciferin for all glowworms based on their fitness
            for i, member in enumerate(population):
                self.luciferin[i] = (1 - self.rho) * self.luciferin[i] + self.gamma * self._calculate_luciferin(member)
            
            # Move each glowworm
            for i in range(search_agents_no):
                # Get neighbors that dominate the current glowworm
                neighbors = self._get_neighbors(i, population)
                
                if not neighbors:
                    # No dominating neighbors found, stay in current position
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
                population[i].multi_fitness = self.objective_func(new_position)
                
                # Update decision range based on number of neighbors
                neighbor_count = len(neighbors)
                self.decision_range[i] = min(self.rs, max(0, self.decision_range[i] + self.beta * (self.nt - neighbor_count)))
            
            # Update archive with current population
            self._add_to_archive(population)
            
            # Store current archive state in history
            history_archive.append([member.copy() for member in self.archive])
            
            # Update progress
            self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
        
        # Final storage
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver execution
        self._end_step_solver()
        
        return history_archive, self.archive

    def _roulette_wheel_selection(self, probabilities):
        """Perform roulette wheel selection (fitness proportionate selection)."""
        return roulette_wheel_selection(probabilities)