import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from ._general import sort_population

class HenryGasSolubilityOptimizer(Solver):
    """
    Henry Gas Solubility Optimization (HGSO) Algorithm.
    
    HGSO is a physics-inspired metaheuristic optimization algorithm that mimics
    the behavior of gas solubility in liquid based on Henry's law. The algorithm
    uses the principles of Henry's gas solubility to optimize search processes.
    
    The algorithm features:
    - Group-based population structure with different Henry constants
    - Temperature-dependent solubility updates
    - Position updates based on gas solubility principles
    - Worst agent replacement mechanism for diversity
    
    Parameters:
    -----------
    objective_func : Callable
        Objective function to optimize
    lb : Union[float, np.ndarray]
        Lower bounds of search space
    ub : Union[float, np.ndarray]
        Upper bounds of search space  
    dim : int
        Number of dimensions in the problem
    maximize : bool, optional
        Optimization direction, default is True (maximize)
    **kwargs
        Additional algorithm parameters:
        - n_types: Number of gas types/groups (default: 5)
        - l1: Constant for Henry's constant initialization (default: 5e-3)
        - l2: Constant for partial pressure initialization (default: 100)
        - l3: Constant for constant C initialization (default: 1e-2)
        - alpha: Constant for position update (default: 1)
        - beta: Constant for position update (default: 1)
        - M1: Minimum fraction of worst agents to replace (default: 0.1)
        - M2: Maximum fraction of worst agents to replace (default: 0.2)
    
    References:
        Original MATLAB implementation by Essam Houssein
        Based on Henry's gas solubility law
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set solver name
        self.name_solver = "Henry Gas Solubility Optimizer"
        
        # Algorithm-specific parameters with defaults
        self.n_types = kwargs.get('n_types', 5)   # Number of gas types/groups
        self.l1 = kwargs.get('l1', 5e-3)          # Constant for Henry's constant
        self.l2 = kwargs.get('l2', 100)           # Constant for partial pressure
        self.l3 = kwargs.get('l3', 1e-2)          # Constant for constant C
        self.alpha = kwargs.get('alpha', 1)       # Position update constant
        self.beta = kwargs.get('beta', 1)         # Position update constant
        self.M1 = kwargs.get('M1', 0.1)           # Min fraction of worst agents
        self.M2 = kwargs.get('M2', 0.2)           # Max fraction of worst agents
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Execute the Henry Gas Solubility Optimization Algorithm.
        
        The algorithm simulates gas solubility behavior based on Henry's law
        with group-based optimization and temperature-dependent updates.
        
        Args:
            search_agents_no (int): Number of gas particles in the population
            max_iter (int): Maximum number of iterations for optimization
            
        Returns:
            Tuple[List, Member]: A tuple containing:
                - history_step_solver: List of best solutions at each iteration
                - best_solver: Best solution found overall
        """
        # Initialize storage variables
        history_step_solver = []
        
        # Initialize population
        population = self._init_population(search_agents_no)
        
        # Initialize best solution
        sorted_population, _ = self._sort_population(population)
        best_solver = sorted_population[0].copy()
        
        # Initialize algorithm parameters
        K = self.l1 * np.random.rand(self.n_types)  # Henry's constants
        P = self.l2 * np.random.rand(search_agents_no)  # Partial pressures
        C = self.l3 * np.random.rand(self.n_types)  # Constants
        
        # Create groups
        groups = self._create_groups(population)
        
        # Evaluate initial groups
        group_best_fitness = np.zeros(self.n_types)
        group_best_positions = [None] * self.n_types
        
        for i in range(self.n_types):
            groups[i], group_best_fitness[i], group_best_positions[i] = self._evaluate_group(
                groups[i], None, True
            )
        
        # Find global best
        global_best_idx = np.argmin(group_best_fitness) if not self.maximize else np.argmax(group_best_fitness)
        global_best_fitness = group_best_fitness[global_best_idx]
        global_best_position = group_best_positions[global_best_idx]
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Update variables (solubility)
            S = self._update_variables(search_agents_no, iter, max_iter, K, P, C)
            
            # Update positions
            new_groups = self._update_positions(
                groups, group_best_positions, global_best_position, S, 
                global_best_fitness, search_agents_no
            )
            
            # Ensure positions stay within bounds
            new_groups = self._check_positions(new_groups)
            
            # Evaluate new groups and update
            for i in range(self.n_types):
                groups[i], group_best_fitness[i], group_best_positions[i] = self._evaluate_group(
                    groups[i], new_groups[i], False
                )
                
                # Replace worst agents
                groups[i] = self._worst_agents(groups[i])
            
            # Update global best
            current_best_idx = np.argmin(group_best_fitness) if not self.maximize else np.argmax(group_best_fitness)
            current_best_fitness = group_best_fitness[current_best_idx]
            current_best_position = group_best_positions[current_best_idx]
            
            if self._is_better(current_best_fitness, global_best_fitness):
                global_best_fitness = current_best_fitness
                global_best_position = current_best_position
                best_solver = Member(global_best_position.copy(), global_best_fitness)
            
            # Store history
            history_step_solver.append(best_solver.copy())
            
            # Update progress
            self._callbacks(iter, max_iter, best_solver)
        
        # Final processing
        self.history_step_solver = history_step_solver
        self.best_solver = best_solver
        
        # End solver
        self._end_step_solver()
        
        return history_step_solver, best_solver
    
    def _create_groups(self, population: List[Member]) -> List[List[Member]]:
        """Create groups from population."""
        group_size = len(population) // self.n_types
        groups = [[] for _ in range(self.n_types)]
        
        for i in range(self.n_types):
            start_idx = i * group_size
            end_idx = start_idx + group_size
            groups[i] = population[start_idx:end_idx].copy()
        
        return groups
    
    def _evaluate_group(self, group: List[Member], new_group: List[Member], 
                       init_flag: bool) -> Tuple[List[Member], float, np.ndarray]:
        """Evaluate group fitness and find best solution."""
        group_size = len(group)
        
        if init_flag:
            # Initial evaluation
            for j in range(group_size):
                group[j].fitness = self.objective_func(group[j].position)
        else:
            # Update evaluation
            for j in range(group_size):
                new_fitness = self.objective_func(new_group[j].position)
                if (not self.maximize and new_fitness < group[j].fitness) or \
                   (self.maximize and new_fitness > group[j].fitness):
                    group[j].fitness = new_fitness
                    group[j].position = new_group[j].position.copy()
        
        # Find best in group
        fitness_values = [member.fitness for member in group]
        if self.maximize:
            best_idx = np.argmax(fitness_values)
        else:
            best_idx = np.argmin(fitness_values)
        
        best_fitness = fitness_values[best_idx]
        best_position = group[best_idx].position.copy()
        
        return group, best_fitness, best_position
    
    def _update_variables(self, search_agents_no: int, iter: int, max_iter: int, K: np.ndarray, 
                         P: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Update solubility variables."""
        T = np.exp(-iter / max_iter)  # Temperature
        T0 = 298.15  # Reference temperature
        
        group_size = search_agents_no // self.n_types
        S = np.zeros(search_agents_no)  # Solubility
        
        for j in range(self.n_types):
            # Update Henry's constant
            K[j] = K[j] * np.exp(-C[j] * (1/T - 1/T0))
            
            # Update solubility for this group
            start_idx = j * group_size
            end_idx = start_idx + group_size
            S[start_idx:end_idx] = P[start_idx:end_idx] * K[j]
        
        return S
    
    def _update_positions(self, groups: List[List[Member]], group_best_positions: List[np.ndarray],
                         global_best_position: np.ndarray, S: np.ndarray, 
                         global_best_fitness: float, search_agents_no: int) -> List[List[Member]]:
        """Update particle positions."""
        new_groups = [[] for _ in range(self.n_types)]
        group_size = search_agents_no // self.n_types
        flag_options = [1, -1]  # Direction flags
        
        for i in range(self.n_types):
            new_group = []
            for j in range(group_size):
                # Calculate gamma parameter
                current_fitness = groups[i][j].fitness
                gamma = self.beta * np.exp(
                    -(global_best_fitness + 0.05) / (current_fitness + 0.05)
                )
                
                # Random direction flag
                flag_idx = np.random.randint(0, 2)
                direction_flag = flag_options[flag_idx]
                
                # Update position
                new_position = groups[i][j].position.copy()
                for k in range(self.dim):
                    # Group best influence
                    group_best_influence = direction_flag * np.random.random() * gamma * \
                                          (group_best_positions[i][k] - groups[i][j].position[k])
                    
                    # Global best influence
                    global_best_influence = np.random.random() * self.alpha * direction_flag * \
                                           (S[i * group_size + j] * global_best_position[k] - groups[i][j].position[k])
                    
                    new_position[k] += group_best_influence + global_best_influence
                
                new_member = Member(new_position, 0.0)
                new_group.append(new_member)
            
            new_groups[i] = new_group
        
        return new_groups
    
    def _check_positions(self, groups: List[List[Member]]) -> List[List[Member]]:
        """Ensure positions stay within bounds."""
        for i in range(self.n_types):
            for j in range(len(groups[i])):
                groups[i][j].position = np.clip(groups[i][j].position, self.lb, self.ub)
        
        return groups
    
    def _worst_agents(self, group: List[Member]) -> List[Member]:
        """Replace worst agents in group."""
        group_size = len(group)
        
        # Calculate number of worst agents to replace
        M1N = self.M1 * group_size
        M2N = self.M2 * group_size
        Nw = int(round((M2N - M1N) * np.random.random() + M1N))
        
        if Nw > 0:
            # Sort by fitness (worst first)
            if self.maximize:
                sorted_indices = np.argsort([m.fitness for m in group])  # Ascending for maximization
            else:
                sorted_indices = np.argsort([m.fitness for m in group])[::-1]  # Descending for minimization
            
            # Replace worst agents with random positions
            for k in range(min(Nw, group_size)):
                worst_idx = sorted_indices[k]
                new_position = np.random.uniform(self.lb, self.ub, self.dim)
                group[worst_idx].position = new_position
                group[worst_idx].fitness = self.objective_func(new_position)
        
        return group
    
    def _sort_population(self, population):
        """
        Sort the population based on fitness.
        """
        return sort_population(population, self.maximize)
