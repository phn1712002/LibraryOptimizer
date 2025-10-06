import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from ._general import sort_population, levy_flight


class CuckooSearchOptimizer(Solver):
    """
    Cuckoo Search optimization algorithm.
    
    Cuckoo Search is a nature-inspired metaheuristic algorithm based on the 
    brood parasitism of some cuckoo species combined with Levy flight behavior.
    
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
        Additional algorithm parameters including:
        - pa: Discovery rate of alien eggs/solutions (default: 0.25)
        - beta: Levy exponent for flight steps (default: 1.5)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set solver name
        self.name_solver = "Cuckoo Search Optimizer"
        
        # Set algorithm parameters with defaults
        self.pa = kwargs.get('pa', 0.25)  # Discovery rate
        self.beta = kwargs.get('beta', 1.5)  # Levy exponent
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Main optimization method for Cuckoo Search algorithm.
        
        Parameters:
        -----------
        search_agents_no : int
            Number of nests (search agents)
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List, Member]
            History of best solutions and the best solution found
        """
        # Initialize population of nests
        population = self._init_population(search_agents_no)
        
        # Find initial best solution
        sorted_population, _ = sort_population(population, self.maximize)
        best_solver = sorted_population[0].copy()
        
        # Initialize history
        history_step_solver = []
        
        # Start solver (show progress bar)
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Generate new solutions via Levy flights (keep current best)
            new_population = self._get_cuckoos(population, best_solver)
            
            # Evaluate new solutions and update population
            population = self._update_population(population, new_population)
            
            # Discovery and randomization: abandon some nests and build new ones
            abandoned_nests = self._empty_nests(population)
            
            # Evaluate abandoned nests and update population
            population = self._update_population(population, abandoned_nests)
            
            # Update best solution
            sorted_population, _ = sort_population(population, self.maximize)
            current_best = sorted_population[0]
            if self._is_better(current_best, best_solver):
                best_solver = current_best.copy()
            
            # Save history
            history_step_solver.append(best_solver.copy())
            
            # Call callback
            self._callbacks(iter, max_iter, best_solver)
        
        # End solver
        self.history_step_solver = history_step_solver
        self.best_solver = best_solver
        self._end_step_solver()
        
        return history_step_solver, best_solver
    
    def _get_cuckoos(self, population: List[Member], best_solver: Member) -> List[Member]:
        """
        Generate new solutions via Levy flights.
        
        Parameters:
        -----------
        population : List[Member]
            Current population of nests
        best_solver : Member
            Current best solution
            
        Returns:
        --------
        List[Member]
            New solutions generated via Levy flights
        """
        new_population = []
        
        for member in population:
            # Generate Levy flight step
            step = self._levy_flight()
            
            # Scale step size (0.01 factor as in original implementation)
            step_size = 0.01 * step * (member.position - best_solver.position)
            
            # Generate new position
            new_position = member.position + step_size * np.random.randn(self.dim)
            
            # Apply bounds
            new_position = np.clip(new_position, self.lb, self.ub)
            
            # Evaluate fitness
            new_fitness = self.objective_func(new_position)
            
            # Create new member
            new_population.append(Member(new_position, new_fitness))
        
        return new_population
    
    def _empty_nests(self, population: List[Member]) -> List[Member]:
        """
        Discover and replace abandoned nests.
        
        Parameters:
        -----------
        population : List[Member]
            Current population of nests
            
        Returns:
        --------
        List[Member]
            New nests to replace abandoned ones
        """
        n = len(population)
        new_nests = []
        
        # Create discovery status vector
        discovery_status = np.random.random(n) > self.pa
        
        for i, discovered in enumerate(discovery_status):
            if discovered:
                # This nest is discovered and will be abandoned
                # Generate new solution via random walk
                idx1, idx2 = np.random.choice(n, 2, replace=False)
                step_size = np.random.random() * (population[idx1].position - population[idx2].position)
                
                new_position = population[i].position + step_size
                
                # Apply bounds
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Evaluate fitness
                new_fitness = self.objective_func(new_position)
                
                new_nests.append(Member(new_position, new_fitness))
            else:
                # Keep the original nest
                new_nests.append(population[i].copy())
        
        return new_nests
    
    def _update_population(self, current_population: List[Member], 
                          new_population: List[Member]) -> List[Member]:
        """
        Update population by keeping better solutions.
        
        Parameters:
        -----------
        current_population : List[Member]
            Current population
        new_population : List[Member]
            Newly generated population
            
        Returns:
        --------
        List[Member]
            Updated population with better solutions
        """
        updated_population = []
        
        for current, new in zip(current_population, new_population):
            if self._is_better(new, current):
                updated_population.append(new)
            else:
                updated_population.append(current)
        
        return updated_population

    def _levy_flight(self):
        """Generate Levy flight step using utility function"""
        return levy_flight(self.dim, self.beta)