import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember


class MultiObjectiveShuffledFrogLeapingOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Shuffled Frog Leaping Optimizer
    
    This algorithm extends the standard SFLA for multi-objective optimization
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
        - archive_size: Size of the external archive (default: 100)
        - alpha: Grid inflation parameter (default: 0.1)
        - n_grid: Number of grids per dimension (default: 7)
        - beta: Leader selection pressure (default: 2)
        - gamma: Archive removal pressure (default: 2)
        - n_memeplex: Number of memeplexes (default: 5)
        - memeplex_size: Size of each memeplex (default: 10)
        - fla_q: Number of parents in FLA (default: 30% of memeplex size)
        - fla_alpha: Number of offsprings in FLA (default: 3)
        - fla_beta: Maximum iterations in FLA (default: 5)
        - fla_sigma: Step size in FLA (default: 2.0)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Shuffled Frog Leaping Optimizer"
        
        # Set SFLA parameters with defaults
        self.n_memeplex = kwargs.get('n_memeplex', 5)
        self.memeplex_size = kwargs.get('memeplex_size', 10)
        self.fla_q = kwargs.get('fla_q', None)  # Number of parents
        self.fla_alpha = kwargs.get('fla_alpha', 3)  # Number of offsprings
        self.fla_beta = kwargs.get('fla_beta', 5)  # Maximum FLA iterations
        self.fla_sigma = kwargs.get('fla_sigma', 2.0)  # Step size

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Main optimization method for multi-objective SFLA
        
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
        # Initialize parameters
        if self.fla_q is None:
            # Default: 30% of memeplex size, at least 2
            self.fla_q = max(round(0.3 * self.memeplex_size), 2)
        
        # Ensure memeplex size is at least dimension + 1 (Nelder-Mead standard)
        self.memeplex_size = max(self.memeplex_size, self.dim + 1)
        
        # Calculate total population size
        total_pop_size = self.n_memeplex * self.memeplex_size
        if total_pop_size != search_agents_no:
            print(f"Warning: Adjusted population size from {search_agents_no} to {total_pop_size} "
                  f"to match memeplex structure ({self.n_memeplex} memeplexes × {self.memeplex_size} frogs)")
            search_agents_no = total_pop_size
        
        # Initialize storage
        history_archive = []
        
        # Initialize population
        population = self._init_population(search_agents_no)
        
        # Initialize archive with non-dominated solutions
        self._determine_domination(population)
        non_dominated = self._get_non_dominated_particles(population)
        self.archive.extend(non_dominated)
        
        # Initialize grid for archive
        if self.archive:
            costs = self._get_costs(self.archive)
            if costs.size > 0:
                self.grid = self._create_hypercubes(costs)
                for particle in self.archive:
                    particle.grid_index, particle.grid_sub_index = self._get_grid_index(particle)
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Create memeplex indices
        memeplex_indices = np.arange(search_agents_no).reshape(self.n_memeplex, self.memeplex_size)
        
        # Main SFLA loop
        for iter in range(max_iter):
            # Shuffle population (main SFLA step)
            np.random.shuffle(population)
            
            # Process each memeplex
            for j in range(self.n_memeplex):
                # Extract memeplex
                start_idx = j * self.memeplex_size
                end_idx = start_idx + self.memeplex_size
                memeplex = population[start_idx:end_idx]
                
                # Run FLA on memeplex
                updated_memeplex = self._run_fla(memeplex)
                
                # Update population
                population[start_idx:end_idx] = updated_memeplex
            
            # Update archive with current population
            self._add_to_archive(population)
            
            # Store archive state for history
            history_archive.append([frog.copy() for frog in self.archive])
            
            # Update progress
            self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
        
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver
        self._end_step_solver()
        
        return history_archive, self.archive
    
    def _is_in_range(self, x: np.ndarray) -> bool:
        """
        Check if position is within variable bounds.
        
        Parameters:
        -----------
        x : np.ndarray
            Position to check
            
        Returns:
        --------
        bool
            True if position is within bounds, False otherwise
        """
        return np.all(x >= self.lb) and np.all(x <= self.ub)

    def _rand_sample(self, probabilities: np.ndarray, q: int, replacement: bool = False) -> List[int]:
        """
        Random sampling with probabilities.
        
        Parameters:
        -----------
        probabilities : np.ndarray
            Selection probabilities
        q : int
            Number of samples to draw
        replacement : bool, optional
            Whether to sample with replacement, default is False
            
        Returns:
        --------
        List[int]
            List of selected indices
        """
        selected_indices = []
        current_probs = probabilities.copy()
        
        for _ in range(q):
            # Normalize probabilities
            if np.sum(current_probs) == 0:
                # If all probabilities are zero, use uniform distribution
                current_probs = np.ones_like(current_probs) / len(current_probs)
            else:
                current_probs = current_probs / np.sum(current_probs)
            
            # Select one index
            r = np.random.random()
            cumulative_sum = np.cumsum(current_probs)
            selected_idx = np.argmax(r <= cumulative_sum)
            selected_indices.append(selected_idx)
            
            if not replacement:
                # Set probability to zero for selected index
                current_probs[selected_idx] = 0
        
        return selected_indices

    def _run_fla(self, memeplex: List[MultiObjectiveMember]) -> List[MultiObjectiveMember]:
        """
        Run Frog Leaping Algorithm on a memeplex for multi-objective optimization.
        
        Parameters:
        -----------
        memeplex : List[MultiObjectiveMember]
            Current memeplex to optimize
            
        Returns:
        --------
        List[MultiObjectiveMember]
            Updated memeplex after FLA
        """
        n_pop = len(memeplex)
        
        # Calculate selection probabilities (rank-based)
        # For multi-objective, we use grid-based diversity for ranking
        if self.archive and len(self.archive) > 0:
            # Calculate diversity scores based on grid occupancy
            grid_counts = {}
            for frog in memeplex:
                if frog.grid_index is not None:
                    grid_counts[frog.grid_index] = grid_counts.get(frog.grid_index, 0) + 1
            
            # Higher probability for frogs in less crowded grid cells
            selection_probs = np.ones(n_pop)
            for i, frog in enumerate(memeplex):
                if frog.grid_index is not None and frog.grid_index in grid_counts:
                    selection_probs[i] = 1.0 / (grid_counts[frog.grid_index] + 1)
        else:
            # Fallback: uniform probabilities
            selection_probs = np.ones(n_pop) / n_pop
        
        # Calculate population range (smallest hypercube)
        # Handle case where memeplex might be empty
        if n_pop == 0:
            return []
            
        positions = np.array([member.position for member in memeplex])
        lower_bound = np.min(positions, axis=0)
        upper_bound = np.max(positions, axis=0)
        
        # FLA main loop
        for _ in range(self.fla_beta):
            # Select parents
            parent_indices = self._rand_sample(selection_probs, self.fla_q)
            parents = [memeplex[i] for i in parent_indices]
            
            # Generate offsprings
            for _ in range(self.fla_alpha):
                # For multi-objective, we need to select leaders from archive
                # Select multiple leaders from archive for guidance
                leaders = self._select_multiple_leaders(3)  # Select 3 leaders
                
                # If we don't have enough leaders, use random frogs from memeplex
                if len(leaders) < 3:
                    available_frogs = [f for f in memeplex if f not in leaders]
                    needed = 3 - len(leaders)
                    if available_frogs:
                        additional = list(np.random.choice(available_frogs, 
                                                         size=min(needed, len(available_frogs)), 
                                                         replace=False))
                        leaders.extend(additional)
                
                # Ensure we have exactly 3 leaders
                if len(leaders) > 3:
                    leaders = leaders[:3]
                
                # Sort parents to find worst one
                # For multi-objective, we use crowding distance or random selection
                worst_idx = np.random.randint(len(parents))
                worst_parent = parents[worst_idx]
                
                # Flags for improvement steps
                improvement_step2 = False
                censorship = False
                
                # Improvement Step 1: Move towards best leader
                new_sol_1 = worst_parent.copy()
                step = self.fla_sigma * np.random.random(self.dim) * (
                    leaders[0].position - worst_parent.position
                )
                new_sol_1.position = worst_parent.position + step
                
                if self._is_in_range(new_sol_1.position):
                    new_sol_1.multi_fitness = self.objective_func(new_sol_1.position)
                    # For multi-objective, we check if new solution is non-dominated
                    if not self._dominates(worst_parent, new_sol_1):  # New solution is not worse
                        parents[worst_idx] = new_sol_1
                    else:
                        improvement_step2 = True
                else:
                    improvement_step2 = True
                
                # Improvement Step 2: Move towards other leaders
                if improvement_step2:
                    new_sol_2 = worst_parent.copy()
                    # Choose a different leader (not the first one)
                    leader_idx = np.random.randint(1, len(leaders))
                    step = self.fla_sigma * np.random.random(self.dim) * (
                        leaders[leader_idx].position - worst_parent.position
                    )
                    new_sol_2.position = worst_parent.position + step
                    
                    if self._is_in_range(new_sol_2.position):
                        new_sol_2.multi_fitness = self.objective_func(new_sol_2.position)
                        if not self._dominates(worst_parent, new_sol_2):  # New solution is not worse
                            parents[worst_idx] = new_sol_2
                        else:
                            censorship = True
                    else:
                        censorship = True
                
                # Censorship: Replace with random solution
                if censorship:
                    random_position = np.random.uniform(lower_bound, upper_bound, self.dim)
                    random_fitness = self.objective_func(random_position)
                    parents[worst_idx] = MultiObjectiveMember(random_position, random_fitness)
        
        return parents
