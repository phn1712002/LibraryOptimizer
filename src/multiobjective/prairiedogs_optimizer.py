import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember
from utils.general import levy_flight

class MultiObjectivePrairieDogsOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Prairie Dogs Optimization (PDO) algorithm.
    
    Based on the MATLAB implementation from:
    Absalom E. Ezugwu, Jeffrey O. Agushaka, Laith Abualigah, Seyedali Mirjalili, Amir H Gandomi
    "Prairie Dogs Optimization: A Nature-inspired Metaheuristic"
    
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
        - archive_size: Size of external archive (default: 100)
        - rho: float (default=0.005) - Account for individual PD difference
        - eps_pd: float (default=0.1) - Food source alarm parameter
        - eps: float (default=1e-10) - Small epsilon value for numerical stability
        - beta: float (default=1.5) - Levy flight parameter
        - alpha: Grid inflation parameter (default: 0.1)
        - n_grid: Number of grids per dimension (default: 7)
        - beta_leader: Leader selection pressure (default: 2)
        - gamma: Archive removal pressure (default: 2)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Prairie Dogs Optimizer"
        
        # Store PDO-specific parameters
        self.rho = kwargs.get('rho', 0.005)  # Account for individual PD difference
        self.eps_pd = kwargs.get('eps_pd', 0.1)  # Food source alarm
        self.eps = kwargs.get('eps', 1e-10)  # Small epsilon for numerical stability
        self.beta = kwargs.get('beta', 1.5)  # Levy flight parameter

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Main optimization method for multi-objective PDO
        
        Parameters:
        -----------
        search_agents_no : int
            Number of search agents (prairie dogs)
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
        costs = self._get_costs(self.archive)
        if costs.size > 0:
            self.grid = self._create_hypercubes(costs)
            for particle in self.archive:
                particle.grid_index, particle.grid_sub_index = self._get_grid_index(particle)
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Determine mu value based on iteration parity
            mu = -1 if (iter + 1) % 2 == 0 else 1
            
            # Calculate dynamic parameters
            DS = 1.5 * np.random.randn() * (1 - iter/max_iter) ** (2 * iter/max_iter) * mu  # Digging strength
            PE = 1.5 * (1 - iter/max_iter) ** (2 * iter/max_iter) * mu  # Predator effect
            
            # Generate Levy flight steps for all prairie dogs
            RL = np.array([self._levy_flight() for _ in range(search_agents_no)])
            
            # Select leader from archive for guidance
            leader = self._select_leader()
            if leader is None:
                # If no leader in archive, use random member from population
                leader = np.random.choice(population)
            
            # Create matrix of leader positions for all prairie dogs
            TPD = np.tile(leader.position, (search_agents_no, 1))
            
            # Update each prairie dog's position
            for i in range(search_agents_no):
                new_position = np.zeros(self.dim)
                
                for j in range(self.dim):
                    # Choose a random prairie dog different from current one
                    k = np.random.choice([idx for idx in range(search_agents_no) if idx != i])
                    
                    # Calculate PDO-specific parameters
                    cpd = np.random.rand() * (TPD[i, j] - population[k].position[j]) / (TPD[i, j] + self.eps)
                    P = self.rho + (population[i].position[j] - np.mean(population[i].position)) / (TPD[i, j] * (self.ub[j] - self.lb[j]) + self.eps)
                    eCB = leader.position[j] * P
                    
                    # Different position update strategies based on iteration phase
                    if iter < max_iter / 4:
                        new_position[j] = leader.position[j] - eCB * self.eps_pd - cpd * RL[i, j]
                    elif iter < 2 * max_iter / 4:
                        new_position[j] = leader.position[j] * population[k].position[j] * DS * RL[i, j]
                    elif iter < 3 * max_iter / 4:
                        new_position[j] = leader.position[j] * PE * np.random.rand()
                    else:
                        new_position[j] = leader.position[j] - eCB * self.eps - cpd * np.random.rand()
                
                # Apply bounds
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Update prairie dog position and fitness
                population[i].position = new_position
                population[i].multi_fitness = self.objective_func(new_position)
            
            # Update archive with current population
            self._add_to_archive(population)
            
            # Store archive state for history
            history_archive.append([pd.copy() for pd in self.archive])
            
            # Update progress
            self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
        
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver
        self._end_step_solver()
        
        return history_archive, self.archive
    
    def _levy_flight(self):
        """Generate Levy flight step"""
        return levy_flight(self.dim, self.beta)
