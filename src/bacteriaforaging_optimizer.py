import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from ._general import sort_population

class BacteriaForagingOptimizer(Solver):
    """
    Bacteria Foraging Optimization (BFO) Algorithm.
    
    BFO is a nature-inspired metaheuristic optimization algorithm that mimics
    the foraging behavior of E. coli bacteria. The algorithm simulates three
    main processes in bacterial foraging:
    1. Chemotaxis: Movement towards nutrients (better solutions)
    2. Reproduction: Reproduction of successful bacteria
    3. Elimination-dispersal: Random elimination and dispersal to avoid local optima
    
    The algorithm maintains a population of bacteria that move through the search
    space using a combination of random walks and directed movement based on
    previous success.
    
    Parameters:
    -----------
    n_elimination : int
        Number of elimination-dispersal events (Ne)
    n_reproduction : int
        Number of reproduction steps (Nr)
    n_chemotaxis : int
        Number of chemotaxis steps (Nc)
    n_swim : int
        Number of swim steps (Ns)
    step_size : float
        Step size for movement (C)
    elimination_prob : float
        Probability of elimination-dispersal (Ped)
    
    References:
        Passino, K. M. (2002). Biomimicry of bacterial foraging for distributed 
        optimization and control. IEEE Control Systems Magazine, 22(3), 52-67.
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        """
        Initialize the Bacteria Foraging Optimizer.
        
        Args:
            objective_func (Callable): Objective function to optimize
            lb (Union[float, np.ndarray]): Lower bounds of search space
            ub (Union[float, np.ndarray]): Upper bounds of search space
            dim (int): Number of dimensions in the problem
            maximize (bool): Whether to maximize (True) or minimize (False) objective
            **kwargs: Additional algorithm parameters
        """
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        # Store additional parameters for later use
        self.kwargs = kwargs
        self.name_solver = "Bacteria Foraging Optimizer"
        
        # Algorithm-specific parameters with defaults
        self.n_elimination = kwargs.get('n_elimination', 4)
        self.n_reproduction = kwargs.get('n_reproduction', 4)
        self.n_chemotaxis = kwargs.get('n_chemotaxis', 10)
        self.n_swim = kwargs.get('n_swim', 4)
        self.step_size = kwargs.get('step_size', 0.1)
        self.elimination_prob = kwargs.get('elimination_prob', 0.25)
        
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Execute the Bacteria Foraging Optimization Algorithm.
        
        The algorithm simulates bacterial foraging through three main processes:
        1. Chemotaxis: Bacteria move towards better solutions
        2. Reproduction: Successful bacteria reproduce
        3. Elimination-dispersal: Random elimination and dispersal
        
        Args:
            search_agents_no (int): Number of bacteria in the population
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
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Main optimization loop (elimination-dispersal events)
        for elimination_iter in range(self.n_elimination):
            
            # Reproduction loop
            for reproduction_iter in range(self.n_reproduction):
                
                # Chemotaxis loop
                for chemotaxis_iter in range(self.n_chemotaxis):
                    
                    # Update each bacterium
                    for i, bacterium in enumerate(population):
                        # Generate random direction vector
                        direction = np.random.uniform(-1, 1, self.dim)
                        direction_norm = np.linalg.norm(direction)
                        
                        if direction_norm > 0:
                            direction = direction / direction_norm
                        
                        # Move bacterium
                        new_position = bacterium.position + self.step_size * direction
                        new_position = np.clip(new_position, self.lb, self.ub)
                        
                        # Evaluate new position
                        new_fitness = self.objective_func(new_position)
                        
                        # Swim behavior - continue moving in same direction if improvement
                        swim_count = 0
                        while swim_count < self.n_swim:
                            if self._is_better(new_fitness, bacterium.fitness):
                                # Accept move and continue swimming
                                bacterium.position = new_position
                                bacterium.fitness = new_fitness
                                
                                # Move further in same direction
                                new_position = bacterium.position + self.step_size * direction
                                new_position = np.clip(new_position, self.lb, self.ub)
                                new_fitness = self.objective_func(new_position)
                                swim_count += 1
                            else:
                                # Stop swimming
                                break
                    
                    # Update best solution
                    sorted_population, _ = self._sort_population(population)
                    current_best = sorted_population[0]
                    if self._is_better(current_best, best_solver):
                        best_solver = current_best.copy()
                
                # Store history
                history_step_solver.append(best_solver.copy())
                
                # Call callback for progress tracking
                current_iter = (elimination_iter * self.n_reproduction * self.n_chemotaxis + 
                              reproduction_iter * self.n_chemotaxis + chemotaxis_iter)
                self._callbacks(current_iter, max_iter, best_solver)
            
            # Reproduction: Keep best half and duplicate
            sorted_population, _ = self._sort_population(population)
            best_half = sorted_population[:search_agents_no // 2]
            
            # Create new population by duplicating best half
            new_population = []
            for bacterium in best_half:
                new_population.append(bacterium.copy())
            for bacterium in best_half:
                new_population.append(bacterium.copy())
            
            population = new_population
        
        # Elimination-dispersal: Random elimination of some bacteria
        for i, bacterium in enumerate(population):
            if np.random.random() < self.elimination_prob:
                # Randomly disperse this bacterium
                new_position = np.random.uniform(self.lb, self.ub, self.dim)
                new_fitness = self.objective_func(new_position)
                population[i] = Member(new_position, new_fitness)
        
        # Final update of best solution
        sorted_population, _ = self._sort_population(population)
        current_best = sorted_population[0]
        if self._is_better(current_best, best_solver):
            best_solver = current_best.copy()
        
        # Store final history
        history_step_solver.append(best_solver.copy())
        
        # Finalize solver
        self.history_step_solver = history_step_solver
        self.best_solver = best_solver
        self._end_step_solver()
        
        return history_step_solver, best_solver
    
    def _sort_population(self, population):
        """
        Sort the population based on fitness.
        """
        return sort_population(population, self.maximize)
