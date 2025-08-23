import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from utils.general import sort_population


class ArtificialEcosystemOptimizer(Solver):
    """
    Artificial Ecosystem-based Optimization (AEO) algorithm.
    
    A nature-inspired meta-heuristic algorithm based on energy flow in ecosystems.
    
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
        Additional algorithm parameters
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize)
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set solver name
        self.name_solver = "Artificial Ecosystem Optimizer"
        
        # Set default parameters
        self.production_weight = kwargs.get('production_weight', 1.0)
        self.consumption_weight = kwargs.get('consumption_weight', 1.0)
        self.decomposition_weight = kwargs.get('decomposition_weight', 1.0)

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Main optimization method for AEO algorithm.
        
        Parameters:
        -----------
        search_agents_no : int
            Number of search agents (population size)
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List, Member]
            History of best solutions and the best solution found
        """
        # Initialize storage variables
        history_step_solver = []
        
        # Initialize the population
        population = self._init_population(search_agents_no)
        
        # Sort population and get best solution
        sorted_population, _ = self._sort_population(population)
        best_solver = sorted_population[0].copy()
        
        # Call the begin function
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Production phase: Create new organism based on best and random position
            new_population = self._production_phase(population, iter, max_iter)
            
            # Consumption phase: Update organisms based on consumption behavior
            new_population = self._consumption_phase(new_population, population)
            
            # Decomposition phase: Update organisms based on decomposition behavior
            new_population = self._decomposition_phase(new_population, best_solver)
            
            # Evaluate new population and update
            population = self._evaluate_and_update(population, new_population)
            
            # Update best solution
            sorted_population, _ = self._sort_population(population)
            current_best = sorted_population[0]
            if self._is_better(current_best, best_solver):
                best_solver = current_best.copy()
            
            # Store the best solution at this iteration
            history_step_solver.append(best_solver.copy())
            
            # Call the callbacks
            self._callbacks(iter, max_iter, best_solver)
        
        # Final evaluation and storage
        self.history_step_solver = history_step_solver
        self.best_solver = best_solver
        
        # Call the end function
        self._end_step_solver()
        return history_step_solver, best_solver
    
    def _production_phase(self, population: List, iter: int, max_iter: int) -> List:
        """
        Production phase: Create new organism based on best and random position.
        
        Parameters:
        -----------
        population : List[Member]
            Current population
        iter : int
            Current iteration
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        List[Member]
            New population after production phase
        """
        new_population = []
        
        # Get sorted population indices
        sorted_population, sorted_indices = self._sort_population(population)
        
        # Create random position in search space
        random_position = np.random.uniform(self.lb, self.ub, self.dim)
        
        # Calculate production weight (decreases linearly)
        r1 = np.random.random()
        a = (1 - iter / max_iter) * r1
        
        # Create first organism: combination of best and random position
        best_position = sorted_population[-1].position  # Worst becomes producer
        new_position = (1 - a) * best_position + a * random_position
        new_position = np.clip(new_position, self.lb, self.ub)
        new_fitness = self.objective_func(new_position)
        new_population.append(Member(new_position, new_fitness))
        
        return new_population
    
    def _consumption_phase(self, new_population: List, old_population: List) -> List:
        """
        Consumption phase: Update organisms based on consumption behavior.
        
        Parameters:
        -----------
        new_population : List[Member]
            Population from production phase
        old_population : List[Member]
            Original population
            
        Returns:
        --------
        List[Member]
            Population after consumption phase
        """
        # Get sorted population indices
        sorted_old_population, _ = self._sort_population(old_population)
        
        # Handle second organism (special case)
        if len(old_population) >= 2:
            # Generate consumption factor C using Levy flight
            u = np.random.normal(0, 1, self.dim)
            v = np.random.normal(0, 1, self.dim)
            C = 0.5 * u / np.abs(v)
            
            # Second organism consumes from producer (first organism)
            new_position = old_population[1].position + C * (
                old_population[1].position - new_population[0].position
            )
            
            # Apply bounds
            new_position = np.clip(new_position, self.lb, self.ub)
            new_fitness = self.objective_func(new_position)
            new_population.append(Member(new_position, new_fitness))
        
        # For remaining organisms (starting from third one)
        for i in range(2, len(old_population)):
            # Generate consumption factor C using Levy flight
            u = np.random.normal(0, 1, self.dim)
            v = np.random.normal(0, 1, self.dim)
            C = 0.5 * u / np.abs(v)
            
            r = np.random.random()
            
            if r < 1/3:
                # Consume from producer (first organism)
                new_position = old_population[i].position + C * (
                    old_population[i].position - new_population[0].position
                )
            elif 1/3 <= r < 2/3:
                # Consume from random consumer (between 1 and i-1)
                random_idx = np.random.randint(1, i)
                new_position = old_population[i].position + C * (
                    old_population[i].position - old_population[random_idx].position
                )
            else:
                # Consume from both producer and random consumer
                r2 = np.random.random()
                random_idx = np.random.randint(1, i)
                new_position = old_population[i].position + C * (
                    r2 * (old_population[i].position - new_population[0].position) +
                    (1 - r2) * (old_population[i].position - old_population[random_idx].position)
                )
            
            # Apply bounds
            new_position = np.clip(new_position, self.lb, self.ub)
            new_fitness = self.objective_func(new_position)
            new_population.append(Member(new_position, new_fitness))
        
        return new_population
    
    def _decomposition_phase(self, population: List, best_solver: Member) -> List:
        """
        Decomposition phase: Update organisms based on decomposition behavior.
        
        Parameters:
        -----------
        population : List[Member]
            Current population
        best_solver : Member
            Best solution found so far
            
        Returns:
        --------
        List[Member]
            Population after decomposition phase
        """
        new_population = []
        
        # Find the best organism in current population
        sorted_population, _ = self._sort_population(population)
        best_current = sorted_population[0]
        
        for i in range(len(population)):
            # Generate decomposition factors
            r3 = np.random.random()
            # Random dimension selection (0 or 1)
            dim_choice = np.random.choice([0, 1])
            weight_factor = 3 * np.random.normal(0, 1)
            
            # Calculate new position using decomposition equation
            # Note: In MATLAB, this uses (r3*randi([1 2])-1) which gives either 0 or 1
            random_multiplier = np.random.randint(1, 3)  # This gives 1 or 2
            new_position = best_solver.position + weight_factor * (
                (r3 * random_multiplier - 1) * best_solver.position -
                (2 * r3 - 1) * population[i].position
            )
            
            # Apply bounds
            new_position = np.clip(new_position, self.lb, self.ub)
            new_fitness = self.objective_func(new_position)
            new_population.append(Member(new_position, new_fitness))
        
        return new_population
    
    def _evaluate_and_update(self, old_population: List, new_population: List) -> List:
        """
        Evaluate new population and update if better solutions are found.
        
        Parameters:
        -----------
        old_population : List[Member]
            Original population
        new_population : List[Member]
            New population after all phases
            
        Returns:
        --------
        List[Member]
            Updated population
        """
        updated_population = []
        
        for i in range(len(old_population)):
            if i < len(new_population) and self._is_better(new_population[i], old_population[i]):
                updated_population.append(new_population[i].copy())
            else:
                updated_population.append(old_population[i].copy())
        
        return updated_population
    
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
            Sorted population and sorted indices
        """
        return sort_population(population, self.maximize)
