import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember
from .._general import levy_flight


class MultiObjectiveCuckooSearchOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Cuckoo Search Optimizer
    
    This algorithm extends the standard Cuckoo Search for multi-objective optimization
    using archive management and grid-based selection for solution evaluation.
    
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
        - pa: Discovery rate of alien eggs/solutions (default: 0.25)
        - beta_levy: Levy exponent for flight steps (default: 1.5)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Cuckoo Search Optimizer"
        
        # Set algorithm-specific parameters with defaults
        self.pa = kwargs.get('pa', 0.25)  # Discovery rate
        self.beta_levy = kwargs.get('beta_levy', 1.5)  # Levy exponent

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Main optimization method for multi-objective Cuckoo Search
        
        Parameters:
        -----------
        search_agents_no : int
            Number of nests (search agents)
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
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Generate new solutions via Levy flights
            new_population = self._get_cuckoos(population)
            
            # Evaluate new solutions and update population using Pareto dominance
            population = self._update_population(population, new_population)
            
            # Discovery and randomization: abandon some nests and build new ones
            abandoned_nests = self._empty_nests(population)
            
            # Evaluate abandoned nests and update population
            population = self._update_population(population, abandoned_nests)
            
            # Update archive with current population
            self._add_to_archive(population)
            
            # Store archive state for history
            history_archive.append([nest.copy() for nest in self.archive])
            
            # Update progress
            self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
        
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver
        self._end_step_solver()
        
        return history_archive, self.archive
    
    def _get_cuckoos(self, population: List[MultiObjectiveMember]) -> List[MultiObjectiveMember]:
        """
        Generate new solutions via Levy flights for multi-objective optimization.
        
        Parameters:
        -----------
        population : List[MultiObjectiveMember]
            Current population of nests
            
        Returns:
        --------
        List[MultiObjectiveMember]
            New solutions generated via Levy flights
        """
        new_population = []
        
        # Select a leader from archive for guidance
        leader = self._select_leader()
        
        for member in population:
            # Generate Levy flight step
            step = self._levy_flight()
            
            if leader is not None:
                # Use leader guidance for Levy flight
                step_size = 0.01 * step * (member.position - leader.position)
            else:
                # Use random direction if no leader available
                step_size = 0.01 * step * np.random.randn(self.dim)
            
            # Generate new position
            new_position = member.position + step_size
            
            # Apply bounds
            new_position = np.clip(new_position, self.lb, self.ub)
            
            # Evaluate fitness
            new_fitness = self.objective_func(new_position)
            
            # Create new member
            new_population.append(MultiObjectiveMember(new_position, new_fitness))
        
        return new_population
    
    def _empty_nests(self, population: List[MultiObjectiveMember]) -> List[MultiObjectiveMember]:
        """
        Discover and replace abandoned nests for multi-objective optimization.
        
        Parameters:
        -----------
        population : List[MultiObjectiveMember]
            Current population of nests
            
        Returns:
        --------
        List[MultiObjectiveMember]
            New nests to replace abandoned ones
        """
        n = len(population)
        new_nests = []
        
        # Create discovery status vector
        discovery_status = np.random.random(n) > self.pa
        
        for i, discovered in enumerate(discovery_status):
            if discovered:
                # This nest is discovered and will be abandoned
                # Generate new solution via random walk using archive members
                if self.archive:
                    # Select two random archive members for guidance
                    if len(self.archive) >= 2:
                        idx1, idx2 = np.random.choice(len(self.archive), 2, replace=False)
                        step_size = np.random.random() * (self.archive[idx1].position - self.archive[idx2].position)
                    else:
                        # If not enough archive members, use population members
                        idx1, idx2 = np.random.choice(n, 2, replace=False)
                        step_size = np.random.random() * (population[idx1].position - population[idx2].position)
                else:
                    # If no archive, use population members
                    idx1, idx2 = np.random.choice(n, 2, replace=False)
                    step_size = np.random.random() * (population[idx1].position - population[idx2].position)
                
                new_position = population[i].position + step_size
                
                # Apply bounds
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Evaluate fitness
                new_fitness = self.objective_func(new_position)
                
                new_nests.append(MultiObjectiveMember(new_position, new_fitness))
            else:
                # Keep the original nest
                new_nests.append(population[i].copy())
        
        return new_nests
    
    def _update_population(self, current_population: List[MultiObjectiveMember], 
                          new_population: List[MultiObjectiveMember]) -> List[MultiObjectiveMember]:
        """
        Update population using Pareto dominance for multi-objective optimization.
        
        Parameters:
        -----------
        current_population : List[MultiObjectiveMember]
            Current population
        new_population : List[MultiObjectiveMember]
            Newly generated population
            
        Returns:
        --------
        List[MultiObjectiveMember]
            Updated population with non-dominated solutions
        """
        updated_population = []
        
        for current, new in zip(current_population, new_population):
            # Use Pareto dominance comparison instead of simple fitness comparison
            if self._dominates(new, current):
                updated_population.append(new)
            elif self._dominates(current, new):
                updated_population.append(current)
            else:
                # If neither dominates the other, randomly choose one
                if np.random.random() < 0.5:
                    updated_population.append(new)
                else:
                    updated_population.append(current)
        
        return updated_population

    def _levy_flight(self):
        """Generate Levy flight step using utility function"""
        return levy_flight(self.dim, self.beta_levy)
