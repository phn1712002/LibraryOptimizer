import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember
from .._general import roulette_wheel_selection

class MultiObjectiveAntColonyOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Ant Colony Optimization for Continuous Domains (MO-ACOR).
    
    This algorithm extends the standard ACOR for multi-objective optimization
    using archive management and grid-based selection for solution guidance.
    
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
        - q: Intensification factor (selection pressure), default 0.5
        - zeta: Deviation-distance ratio, default 1.0
        - alpha: Grid inflation parameter (default: 0.1)
        - n_grid: Number of grids per dimension (default: 7)
        - beta: Leader selection pressure (default: 2)
        - gamma: Archive removal pressure (default: 2)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Ant Colony Optimizer"
        
        # Set algorithm-specific parameters with defaults
        self.q = kwargs.get('q', 0.5)  # Intensification factor
        self.zeta = kwargs.get('zeta', 1.0)  # Deviation-distance ratio

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Main optimization method for multi-objective ACOR
        
        Parameters:
        -----------
        search_agents_no : int
            Number of search agents (population/archive size)
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List, List[MultiObjectiveMember]]
            History of archive states and the final archive
        """
        # Initialize storage
        history_archive = []
        
        # Initialize population (archive)
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
            # Calculate solution weights (Gaussian kernel weights)
            w = self._calculate_weights(len(population))
            
            # Calculate selection probabilities
            p = w / np.sum(w)
            
            # Calculate means (positions of all solutions in population)
            means = np.array([member.position for member in population])
            
            # Calculate standard deviations for each solution
            sigma = self._calculate_standard_deviations(means)
            
            # Create new population by sampling from Gaussian distributions
            new_population = self._sample_new_population(means, sigma, p, search_agents_no)
            
            # Update archive with new population
            self._add_to_archive(new_population)
            
            # Update population for next iteration (use archive as new population)
            # This maintains diversity and focuses search on promising regions
            if len(self.archive) >= search_agents_no:
                # Select diverse solutions from archive using grid-based selection
                population = self._select_diverse_population(search_agents_no)
            else:
                # If archive is smaller than population size, use archive + random solutions
                population = self.archive.copy()
                remaining = search_agents_no - len(population)
                if remaining > 0:
                    additional = self._init_population(remaining)
                    population.extend(additional)
            
            # Store archive state for history
            history_archive.append([ant.copy() for ant in self.archive])
            
            # Update progress
            self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
        
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver
        self._end_step_solver()
        
        return history_archive, self.archive
    
    def _calculate_weights(self, n_pop: int) -> np.ndarray:
        """
        Calculate Gaussian kernel weights for solution selection.
        
        Parameters:
        -----------
        n_pop : int
            Population size
            
        Returns:
        --------
        np.ndarray
            Array of weights for each solution
        """
        w = (1 / (np.sqrt(2 * np.pi) * self.q * n_pop)) * \
            np.exp(-0.5 * (((np.arange(n_pop)) / (self.q * n_pop)) ** 2))
        return w
    
    def _calculate_standard_deviations(self, means: np.ndarray) -> np.ndarray:
        """
        Calculate standard deviations for Gaussian sampling.
        
        Parameters:
        -----------
        means : np.ndarray
            Array of solution positions (means)
            
        Returns:
        --------
        np.ndarray
            Array of standard deviations for each solution
        """
        n_pop = means.shape[0]
        sigma = np.zeros_like(means)
        
        for l in range(n_pop):
            # Calculate average distance to other solutions
            D = np.sum(np.abs(means[l] - means), axis=0)
            sigma[l] = self.zeta * D / (n_pop - 1)
        
        return sigma
    
    def _sample_new_population(self, means: np.ndarray, sigma: np.ndarray, 
                              probabilities: np.ndarray, n_sample: int) -> List[MultiObjectiveMember]:
        """
        Sample new solutions using Gaussian distributions.
        
        Parameters:
        -----------
        means : np.ndarray
            Array of solution positions (means)
        sigma : np.ndarray
            Array of standard deviations
        probabilities : np.ndarray
            Selection probabilities for each solution
        n_sample : int
            Number of samples to generate
            
        Returns:
        --------
        List[MultiObjectiveMember]
            List of newly sampled solutions
        """
        new_population = []
        
        for _ in range(n_sample):
            # Initialize new position
            new_position = np.zeros(self.dim)
            
            # Construct solution component by component
            for i in range(self.dim):
                # Select Gaussian kernel using roulette wheel selection
                l = roulette_wheel_selection(probabilities)
                
                # Generate Gaussian random variable
                new_position[i] = means[l, i] + sigma[l, i] * np.random.randn()
            
            # Ensure positions stay within bounds
            new_position = np.clip(new_position, self.lb, self.ub)
            
            # Evaluate fitness
            new_fitness = self.objective_func(new_position)
            
            # Create new member
            new_population.append(MultiObjectiveMember(new_position, new_fitness))
        
        return new_population
    
    def _select_diverse_population(self, n_select: int) -> List[MultiObjectiveMember]:
        """
        Select diverse population from archive using grid-based selection.
        
        Parameters:
        -----------
        n_select : int
            Number of solutions to select
            
        Returns:
        --------
        List[MultiObjectiveMember]
            Selected diverse population
        """
        if not self.archive or n_select <= 0:
            return []
        
        # Get grid indices of all archive members
        grid_indices = [p.grid_index for p in self.archive if p.grid_index is not None]
        
        if not grid_indices:
            # If no grid indices, return random unique members
            n_available = min(n_select, len(self.archive))
            return list(np.random.choice(self.archive, size=n_available, replace=False))
        
        # Get occupied cells and their counts
        occupied_cells, counts = np.unique(grid_indices, return_counts=True)
        n_cells = len(occupied_cells)
        
        # Selection probabilities (lower density cells have higher probability)
        probabilities = np.exp(-self.beta * counts)
        probabilities = probabilities / np.sum(probabilities)
        
        selected_population = []
        temp_probabilities = probabilities.copy()
        temp_cells = occupied_cells.copy()
        
        # Select solutions from different cells to maintain diversity
        for _ in range(min(n_select, n_cells)):
            if len(temp_cells) == 0:
                break
                
            # Select a cell using roulette wheel
            r = np.random.random()
            cum_probs = np.cumsum(temp_probabilities)
            selected_cell_idx = np.where(r <= cum_probs)[0][0]
            selected_cell = temp_cells[selected_cell_idx]
            
            # Get members in selected cell
            cell_members = [p for p in self.archive if p.grid_index == selected_cell]
            
            # Select one member from the cell
            selected_member = np.random.choice(cell_members)
            selected_population.append(selected_member)
            
            # Remove selected cell from consideration
            mask = temp_cells != selected_cell
            temp_cells = temp_cells[mask]
            temp_probabilities = temp_probabilities[mask]
            if len(temp_probabilities) > 0:
                temp_probabilities = temp_probabilities / np.sum(temp_probabilities)
        
        # If we need more solutions than available cells, fill with random selection
        if len(selected_population) < n_select:
            remaining = n_select - len(selected_population)
            available_members = [p for p in self.archive if p not in selected_population]
            if available_members:
                additional = list(np.random.choice(available_members, 
                                                 size=min(remaining, len(available_members)), 
                                                 replace=False))
                selected_population.extend(additional)
        
        return selected_population
