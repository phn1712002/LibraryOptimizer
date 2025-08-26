import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from ._general import sort_population


class GeneticAlgorithmOptimizer(Solver):
    """
    Genetic Algorithm Optimizer implementation.
    
    This optimizer implements a genetic algorithm with uniform crossover,
    mutation, and natural selection based on the MATLAB implementation.
    
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
        - num_groups: Number of groups of people (default: 5)
        - crossover_rate: Probability of crossover (default: 0.8)
        - mutation_rate: Probability of mutation (default: 0.1)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set default GA parameters
        self.name_solver = "Genetic Algorithm Optimizer"
        self.num_groups = kwargs.get('num_groups', 5)
        self.crossover_rate = kwargs.get('crossover_rate', 0.8)
        self.mutation_rate = kwargs.get('mutation_rate', 0.1)

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Main genetic algorithm optimization method.
        
        Parameters:
        -----------
        search_agents_no : int
            Number of search agents (chromosomes) per population
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List[Member], Member]
            Optimization history and best solution found
        """
        # Initialize storage variables
        history_step_solver = []
        global_best = self.best_solver
        
        # Call the begin function
        self._begin_step_solver(max_iter)
        
        # Run multiple populations (as in MATLAB implementation)
        for pop_idx in range(self.num_groups):
            # Initialize population for this run
            population = self._init_population(search_agents_no)
            
            # Initialize best solution for this population
            sorted_population, sorted_indices = self._sort_population(population)
            population_best = sorted_population[0].copy()
            
            # Main optimization loop for this population
            for iter in range(max_iter):
                # Evaluate fitness for all chromosomes
                for i, member in enumerate(population):
                    population[i].fitness = self.objective_func(member.position)
                
                # Natural selection - sort population by fitness
                sorted_population, sorted_indices = self._sort_population(population)
                current_best = sorted_population[0]
                
                # Update population best
                if self._is_better(current_best, population_best):
                    population_best = current_best.copy()
                
                # Perform crossover
                population = self._uniform_crossover(population, sorted_indices)
                
                # Perform mutation
                population = self._mutation(population, sorted_indices)
                
                # Update global best
                if self._is_better(population_best, global_best):
                    global_best = population_best.copy()
                
                # Store the best solution at this iteration
                history_step_solver.append(global_best.copy())
                
                # Call the callbacks
                self._callbacks(iter + pop_idx * max_iter, self.num_groups * max_iter, global_best)
        
        # Final evaluation and storage
        self.history_step_solver = history_step_solver
        self.best_solver = global_best
        
        # Call the end function
        self._end_step_solver()
        return history_step_solver, global_best
    
    def _sort_population(self, population):
        """
        Sort population based on fitness.
        
        Parameters:
        -----------
        population : List[Member]
            Population to sort
            
        Returns:
        --------
        Tuple[List[Member], List[int]]
            Sorted population and corresponding indices
        """
        return sort_population(population, self.maximize)
    def _uniform_crossover(self, population: List, sorted_indices: List[int]) -> List:
        """
        Perform uniform crossover operation.
        
        Parameters:
        -----------
        population : List[Member]
            Current population
        sorted_indices : List[int]
            Indices of population sorted by fitness
            
        Returns:
        --------
        List[Member]
            Population after crossover
        """
        new_population = [member.copy() for member in population]
        
        for i in range(len(population)):
            # Skip the best two chromosomes (elitism)
            if i == sorted_indices[0] or i == sorted_indices[1]:
                continue
                
            # Perform crossover with probability crossover_rate
            if np.random.random() < self.crossover_rate:
                # Choose random parent from the best two
                parent_idx = np.random.choice([sorted_indices[0], sorted_indices[1]])
                
                # Perform uniform crossover for each dimension
                for d in range(self.dim):
                    if np.random.random() < 0.5:  # 50% chance to inherit from parent
                        new_population[i].position[d] = population[parent_idx].position[d]
        
        return new_population

    def _mutation(self, population: List, sorted_indices: List[int]) -> List:
        """
        Perform mutation operation.
        
        Parameters:
        -----------
        population : List[Member]
            Current population
        sorted_indices : List[int]
            Indices of population sorted by fitness
            
        Returns:
        --------
        List[Member]
            Population after mutation
        """
        new_population = [member.copy() for member in population]
        
        # Mutate the worst chromosome with probability mutation_rate
        worst_idx = sorted_indices[-1]
        if np.random.random() < self.mutation_rate:
            # Replace with random solution
            new_population[worst_idx].position = np.random.uniform(self.lb, self.ub, self.dim)
            new_population[worst_idx].fitness = self.objective_func(new_population[worst_idx].position)
        
        return new_population