import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember

class MultiObjectiveHenryGasSolubilityOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Henry Gas Solubility Optimization (HGSO) Algorithm.
    
    This algorithm extends the standard HGSO for multi-objective optimization
    using archive management and grid-based selection for maintaining diversity.
    
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
        Optimization direction (True: maximize, False: minimize)
    **kwargs
        Additional parameters:
        - n_types: Number of gas types/groups (default: 5)
        - l1: Constant for Henry's constant initialization (default: 5e-3)
        - l2: Constant for partial pressure initialization (default: 100)
        - l3: Constant for constant C initialization (default: 1e-2)
        - alpha: Constant for position update (default: 1)
        - beta: Constant for position update (default: 1)
        - M1: Minimum fraction of worst agents to replace (default: 0.1)
        - M2: Maximum fraction of worst agents to replace (default: 0.2)
        - archive_size: Size of the external archive (default: 100)
        - alpha_grid: Grid inflation parameter (default: 0.1)
        - n_grid: Number of grids per dimension (default: 7)
        - beta_grid: Leader selection pressure (default: 2)
        - gamma_grid: Archive removal pressure (default: 2)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Henry Gas Solubility Optimizer"
        
        # Algorithm-specific parameters with defaults
        self.n_types = kwargs.get('n_types', 5)   # Number of gas types/groups
        self.l1 = kwargs.get('l1', 5e-3)          # Constant for Henry's constant
        self.l2 = kwargs.get('l2', 100)           # Constant for partial pressure
        self.l3 = kwargs.get('l3', 1e-2)          # Constant for constant C
        self.alpha = kwargs.get('alpha', 1)       # Position update constant
        self.beta = kwargs.get('beta', 1)         # Position update constant
        self.M1 = kwargs.get('M1', 0.1)           # Min fraction of worst agents
        self.M2 = kwargs.get('M2', 0.2)           # Max fraction of worst agents

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Main optimization method for multi-objective HGSO.
        
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
        
        # Initialize algorithm parameters
        K = self.l1 * np.random.rand(self.n_types)  # Henry's constants
        P = self.l2 * np.random.rand(search_agents_no)  # Partial pressures
        C = self.l3 * np.random.rand(self.n_types)  # Constants
        
        # Create groups
        groups = self._create_groups(population)
        
        # Evaluate initial groups
        group_best_members = [None] * self.n_types
        
        for i in range(self.n_types):
            groups[i], group_best_members[i] = self._evaluate_group_multi(
                groups[i], None, True
            )
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Update variables (solubility)
            S = self._update_variables(search_agents_no, iter, max_iter, K, P, C)
            
            # Select leader from archive using grid-based selection
            leader = self._select_leader()
            
            # Update positions
            new_groups = self._update_positions_multi(
                groups, group_best_members, leader, S, search_agents_no
            )
            
            # Ensure positions stay within bounds
            new_groups = self._check_positions_multi(new_groups)
            
            # Evaluate new groups
            new_population = []
            for i in range(self.n_types):
                evaluated_group, _ = self._evaluate_group_multi(
                    groups[i], new_groups[i], False
                )
                new_population.extend(evaluated_group)
            
            # Update archive with new population
            self._add_to_archive(new_population)
            
            # Update groups with evaluated groups
            for i in range(self.n_types):
                groups[i], group_best_members[i] = self._evaluate_group_multi(
                    groups[i], new_groups[i], False
                )
                
                # Replace worst agents
                groups[i] = self._worst_agents_multi(groups[i])
            
            # Store archive state for history
            history_archive.append([member.copy() for member in self.archive])
            
            # Update progress
            self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
        
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver
        self._end_step_solver()
        
        return history_archive, self.archive
    
    def _create_groups(self, population: List[MultiObjectiveMember]) -> List[List[MultiObjectiveMember]]:
        """Create groups from population."""
        group_size = len(population) // self.n_types
        groups = [[] for _ in range(self.n_types)]
        
        for i in range(self.n_types):
            start_idx = i * group_size
            end_idx = start_idx + group_size
            groups[i] = population[start_idx:end_idx].copy()
        
        return groups
    
    def _evaluate_group_multi(self, group: List[MultiObjectiveMember], 
                            new_group: List[MultiObjectiveMember], 
                            init_flag: bool) -> Tuple[List[MultiObjectiveMember], MultiObjectiveMember]:
        """Evaluate group fitness for multi-objective optimization."""
        group_size = len(group)
        
        if init_flag:
            # Initial evaluation
            for j in range(group_size):
                fitness = self.objective_func(group[j].position)
                group[j].multi_fitness = np.array(fitness)
        else:
            # Update evaluation
            for j in range(group_size):
                new_fitness = self.objective_func(new_group[j].position)
                group[j].multi_fitness = np.array(new_fitness)
                group[j].position = new_group[j].position.copy()
        
        # Find best in group based on random fitness for selection
        best_member = self._select_best_from_group(group)
        
        return group, best_member
    
    def _select_best_from_group(self, group: List[MultiObjectiveMember]) -> MultiObjectiveMember:
        """Select best member from group using random fitness for diversity."""
        if not group:
            return None
        
        # Use random fitness for selection to maintain diversity
        fitness_values = [self._get_random_fitness(member) for member in group]
        
        if self.maximize:
            best_idx = np.argmax(fitness_values)
        else:
            best_idx = np.argmin(fitness_values)
        
        return group[best_idx]
    
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
    
    def _update_positions_multi(self, groups: List[List[MultiObjectiveMember]], 
                              group_best_members: List[MultiObjectiveMember],
                              leader: MultiObjectiveMember, S: np.ndarray, 
                              search_agents_no: int) -> List[List[MultiObjectiveMember]]:
        """Update particle positions for multi-objective optimization."""
        new_groups = [[] for _ in range(self.n_types)]
        group_size = search_agents_no // self.n_types
        flag_options = [1, -1]  # Direction flags
        
        # If no leader available, use a random member from archive
        if leader is None and self.archive:
            leader = np.random.choice(self.archive)
        elif leader is None:
            # If no archive, use best from first group
            leader = group_best_members[0] if group_best_members[0] else groups[0][0]
        
        for i in range(self.n_types):
            new_group = []
            group_best = group_best_members[i] if group_best_members[i] else groups[i][0]
            
            for j in range(group_size):
                # Calculate gamma parameter using random fitness
                current_fitness = self._get_random_fitness(groups[i][j])
                leader_fitness = self._get_random_fitness(leader)
                gamma = self.beta * np.exp(
                    -(leader_fitness + 0.05) / (current_fitness + 0.05)
                )
                
                # Random direction flag
                flag_idx = np.random.randint(0, 2)
                direction_flag = flag_options[flag_idx]
                
                # Update position
                new_position = groups[i][j].position.copy()
                for k in range(self.dim):
                    # Group best influence
                    group_best_influence = direction_flag * np.random.random() * gamma * \
                                          (group_best.position[k] - groups[i][j].position[k])
                    
                    # Leader influence
                    leader_influence = np.random.random() * self.alpha * direction_flag * \
                                      (S[i * group_size + j] * leader.position[k] - groups[i][j].position[k])
                    
                    new_position[k] += group_best_influence + leader_influence
                
                new_member = MultiObjectiveMember(new_position, np.zeros(self.n_objectives))
                new_group.append(new_member)
            
            new_groups[i] = new_group
        
        return new_groups
    
    def _check_positions_multi(self, groups: List[List[MultiObjectiveMember]]) -> List[List[MultiObjectiveMember]]:
        """Ensure positions stay within bounds for multi-objective."""
        for i in range(self.n_types):
            for j in range(len(groups[i])):
                groups[i][j].position = np.clip(groups[i][j].position, self.lb, self.ub)
        
        return groups
    
    def _worst_agents_multi(self, group: List[MultiObjectiveMember]) -> List[MultiObjectiveMember]:
        """Replace worst agents in group for multi-objective optimization."""
        group_size = len(group)
        
        # Calculate number of worst agents to replace
        M1N = self.M1 * group_size
        M2N = self.M2 * group_size
        Nw = int(round((M2N - M1N) * np.random.random() + M1N))
        
        if Nw > 0:
            # Sort by random fitness (worst first)
            random_fitness = [self._get_random_fitness(m) for m in group]
            
            if self.maximize:
                sorted_indices = np.argsort(random_fitness)  # Ascending for maximization
            else:
                sorted_indices = np.argsort(random_fitness)[::-1]  # Descending for minimization
            
            # Replace worst agents with random positions
            for k in range(min(Nw, group_size)):
                worst_idx = sorted_indices[k]
                new_position = np.random.uniform(self.lb, self.ub, self.dim)
                new_fitness = self.objective_func(new_position)
                group[worst_idx].position = new_position
                group[worst_idx].multi_fitness = np.array(new_fitness)
        
        return group
