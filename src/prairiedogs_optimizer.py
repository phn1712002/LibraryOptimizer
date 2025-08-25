import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from utils.general import sort_population, levy_flight

class PrairieDogsOptimizer(Solver):
    """
    Prairie Dogs Optimization (PDO) algorithm.
    
    Based on the MATLAB implementation from:
    Absalom E. Ezugwu, Jeffrey O. Agushaka, Laith Abualigah, Seyedali Mirjalili, Amir H Gandomi
    "Prairie Dogs Optimization: A Nature-inspired Metaheuristic"
    
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
        Additional algorithm parameters:
        - rho: float (default=0.005) - Account for individual PD difference
        - eps_pd: float (default=0.1) - Food source alarm parameter
        - eps: float (default=1e-10) - Small epsilon value for numerical stability
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set default PDO parameters
        self.name_solver = "Prairie Dogs Optimizer"
        self.rho = kwargs.get('rho', 0.005)  # Account for individual PD difference
        self.eps_pd = kwargs.get('eps_pd', 0.1)  # Food source alarm
        self.eps = kwargs.get('eps', 1e-10)  # Small epsilon for numerical stability
        self.beta = kwargs.get('beta', 1.5)

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Main optimization loop for Prairie Dogs Optimization.
        
        Parameters:
        -----------
        search_agents_no : int
            Number of search agents (prairie dogs)
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List, PrairieDog]
            History of best solutions and the best solution found
        """
        # Initialize storage variables
        history_step_solver = []
        
        # Initialize the population of prairie dogs
        population = self._init_population(search_agents_no)
        
        # Initialize best solution
        sorted_population, _ = self._sort_population(population)
        best_solution = sorted_population[0].copy()
        
        # Call the begin function
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
            
            # Create matrix of best positions for all prairie dogs
            TPD = np.tile(best_solution.position, (search_agents_no, 1))
            
            # Update each prairie dog's position
            for i in range(search_agents_no):
                new_position = np.zeros(self.dim)
                
                for j in range(self.dim):
                    # Choose a random prairie dog different from current one
                    k = np.random.choice([idx for idx in range(search_agents_no) if idx != i])
                    
                    # Calculate PDO-specific parameters
                    cpd = np.random.rand() * (TPD[i, j] - population[k].position[j]) / (TPD[i, j] + self.eps)
                    P = self.rho + (population[i].position[j] - np.mean(population[i].position)) / (TPD[i, j] * (self.ub[j] - self.lb[j]) + self.eps)
                    eCB = best_solution.position[j] * P
                    
                    # Different position update strategies based on iteration phase
                    if iter < max_iter / 4:
                        new_position[j] = best_solution.position[j] - eCB * self.eps_pd - cpd * RL[i, j]
                    elif iter < 2 * max_iter / 4:
                        new_position[j] = best_solution.position[j] * population[k].position[j] * DS * RL[i, j]
                    elif iter < 3 * max_iter / 4:
                        new_position[j] = best_solution.position[j] * PE * np.random.rand()
                    else:
                        new_position[j] = best_solution.position[j] - eCB * self.eps - cpd * np.random.rand()
                
                # Apply bounds
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Evaluate new fitness
                new_fitness = self.objective_func(new_position)
                
                # Create new prairie dog candidate
                new_prairie_dog = Member(new_position, new_fitness)
                
                # Greedy selection
                if self._is_better(new_prairie_dog, population[i]):
                    population[i] = new_prairie_dog
                
                # Update global best solution
                if self._is_better(population[i], best_solution):
                    best_solution = population[i].copy()
            
            # Store the best solution at this iteration
            history_step_solver.append(best_solution.copy())
            
            # Call the callbacks
            self._callbacks(iter, max_iter, best_solution)
            
            # Print progress every 50 iterations
            if (iter + 1) % 50 == 0:
                print(f'At iteration {iter + 1}, the best solution fitness is {best_solution.fitness:.6f}')
        
        # Final evaluation and storage
        self.history_step_solver = history_step_solver
        self.best_solver = best_solution
        
        # Call the end function
        self._end_step_solver()
        return history_step_solver, best_solution
    
    def _sort_population(self, population):
        """
        Sort the population based on fitness.
        
        Parameters:
        -----------
        population : List[PrairieDog]
            Population to sort
            
        Returns:
        --------
        Tuple[List, List]
            Sorted population and sorted indices
        """
        return sort_population(population, self.maximize)

    def _levy_flight(self):
        return levy_flight(self.dim, self.beta)