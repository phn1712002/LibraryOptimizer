import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from utils.general import sort_population


class ShuffledFrogLeapingOptimizer(Solver):
    """
    Shuffled Frog Leaping Algorithm (SFLA) optimizer.
    
    SFLA is a memetic metaheuristic that combines elements of particle swarm
    optimization and shuffled complex evolution. It works by dividing the
    population into memeplexes and performing local search within each memeplex.
    
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
        Additional SFLA parameters:
        - n_memeplex: Number of memeplexes (default: 5)
        - memeplex_size: Size of each memeplex (default: 10)
        - fla_q: Number of parents in FLA (default: 30% of memeplex size)
        - fla_alpha: Number of offsprings in FLA (default: 3)
        - fla_beta: Maximum iterations in FLA (default: 5)
        - fla_sigma: Step size in FLA (default: 2.0)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set solver name
        self.name_solver = "Shuffled Frog Leaping Optimizer"
        
        # Set SFLA parameters with defaults
        self.n_memeplex = kwargs.get('n_memeplex', 5)
        self.memeplex_size = kwargs.get('memeplex_size', 10)
        self.fla_q = kwargs.get('fla_q', None)  # Number of parents
        self.fla_alpha = kwargs.get('fla_alpha', 3)  # Number of offsprings
        self.fla_beta = kwargs.get('fla_beta', 5)  # Maximum FLA iterations
        self.fla_sigma = kwargs.get('fla_sigma', 2.0)  # Step size

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Main optimization method for SFLA.
        
        Parameters:
        -----------
        search_agents_no : int
            Number of search agents (population size)
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List, Member]
            Optimization history and best solution found
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
        
        # Initialize population
        population = self._init_population(search_agents_no)
        
        # Initialize best solution
        sorted_population, _ = self._sort_population(population)
        best_solution = sorted_population[0].copy()
        
        # Initialize history
        history_step_solver = []
        
        # Begin optimization
        self._begin_step_solver(max_iter)
        
        # Create memeplex indices
        memeplex_indices = np.arange(search_agents_no).reshape(self.n_memeplex, self.memeplex_size)
        
        # Main SFLA loop
        for iter in range(max_iter):
            # Prepare FLA parameters
            fla_params = {
                'BestSol': best_solution
            }
            
            # Shuffle population (main SFLA step)
            np.random.shuffle(population)
            
            # Process each memeplex
            for j in range(self.n_memeplex):
                # Extract memeplex
                start_idx = j * self.memeplex_size
                end_idx = start_idx + self.memeplex_size
                memeplex = population[start_idx:end_idx]
                
                # Run FLA on memeplex
                updated_memeplex = self._run_fla(memeplex, best_solution)
                
                # Update population
                population[start_idx:end_idx] = updated_memeplex
            
            # Sort population and update best solution
            sorted_population, _ = self._sort_population(population)
            current_best = sorted_population[0]
            
            if self._is_better(current_best, best_solution):
                best_solution = current_best.copy()
            
            # Store history
            history_step_solver.append(best_solution.copy())
            
            # Call callbacks
            self._callbacks(iter, max_iter, best_solution)
        
        # Finalize optimization
        self.history_step_solver = history_step_solver
        self.best_solver = best_solution
        self._end_step_solver()
        
        return history_step_solver, best_solution
    
    def _sort_population(self, population):
        """
        Sort population based on optimization direction.
        
        Parameters:
        -----------
        population : List[Member]
            Population to sort
            
        Returns:
        --------
        Tuple[List, List]
            Sorted population and sort indices
        """
        return sort_population(population, self.maximize)

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

    def _run_fla(self, memeplex: List[Member], best_solution: Member) -> List[Member]:
        """
        Run Frog Leaping Algorithm on a memeplex.
        
        Parameters:
        -----------
        memeplex : List[Member]
            Current memeplex to optimize
        best_solution : Member
            Global best solution
            
        Returns:
        --------
        List[Member]
            Updated memeplex after FLA
        """
        n_pop = len(memeplex)
        
        # Calculate selection probabilities (rank-based)
        ranks = np.arange(n_pop, 0, -1)  # Higher rank for better solutions
        selection_probs = 2 * (n_pop + 1 - ranks) / (n_pop * (n_pop + 1))
        
        # Calculate population range (smallest hypercube)
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
                # Sort parents (best to worst)
                sorted_parents, sort_order = sort_population(parents, self.maximize)
                parent_indices_sorted = [parent_indices[i] for i in sort_order]
                
                # Get worst parent
                worst_parent = sorted_parents[-1]
                worst_idx = parent_indices_sorted[-1]
                
                # Flags for improvement steps
                improvement_step2 = False
                censorship = False
                
                # Improvement Step 1: Move towards best parent
                new_sol_1 = worst_parent.copy()
                step = self.fla_sigma * np.random.random(self.dim) * (
                    sorted_parents[0].position - worst_parent.position
                )
                new_sol_1.position = worst_parent.position + step
                
                if self._is_in_range(new_sol_1.position):
                    new_sol_1.fitness = self.objective_func(new_sol_1.position)
                    if self._is_better(new_sol_1, worst_parent):
                        memeplex[worst_idx] = new_sol_1
                    else:
                        improvement_step2 = True
                else:
                    improvement_step2 = True
                
                # Improvement Step 2: Move towards global best
                if improvement_step2:
                    new_sol_2 = worst_parent.copy()
                    step = self.fla_sigma * np.random.random(self.dim) * (
                        best_solution.position - worst_parent.position
                    )
                    new_sol_2.position = worst_parent.position + step
                    
                    if self._is_in_range(new_sol_2.position):
                        new_sol_2.fitness = self.objective_func(new_sol_2.position)
                        if self._is_better(new_sol_2, worst_parent):
                            memeplex[worst_idx] = new_sol_2
                        else:
                            censorship = True
                    else:
                        censorship = True
                
                # Censorship: Replace with random solution
                if censorship:
                    random_position = np.random.uniform(lower_bound, upper_bound, self.dim)
                    random_fitness = self.objective_func(random_position)
                    memeplex[worst_idx] = Member(random_position, random_fitness)
        
        return memeplex