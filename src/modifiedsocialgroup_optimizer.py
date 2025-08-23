import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from utils.general import sort_population


class ModifiedSocialGroupOptimizer(Solver):
    """
    Modified Social Group Optimization (MSGO) algorithm.
    
    A metaheuristic optimization algorithm inspired by social group behavior,
    featuring guru phase (learning from best) and learner phase (mutual learning).
    
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
        - c: Learning coefficient for guru phase (default: 0.2)
        - sap: Self-adaptive probability for random exploration (default: 0.7)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize)
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set default MSGO parameters
        self.name_solver = "Modified Social Group Optimizer"
        self.c = kwargs.get('c', 0.2)  # Learning coefficient for guru phase
        self.sap = kwargs.get('sap', 0.7)  # Self-adaptive probability for random exploration

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Execute the MSGO optimization algorithm.
        
        Args:
            search_agents_no: Number of search agents (population size)
            max_iter: Maximum number of iterations
            
        Returns:
            Tuple containing optimization history and best solution found
        """
        # Initialize the population
        population = self._init_population(search_agents_no)
        
        # Initialize best solution
        sorted_population, _ = self._sort_population(population)
        best_solution = sorted_population[0].copy()
        
        # Initialize history
        history_step_solver = []
        
        # Call the begin function
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Phase 1: Guru Phase (improvement due to best person)
            sorted_population, _ = self._sort_population(population)
            guru = sorted_population[0].copy()  # Best individual
            
            # Create new population for guru phase
            new_population = []
            for i in range(search_agents_no):
                new_position = np.zeros(self.dim)
                for j in range(self.dim):
                    # Update position: c*current + rand*(guru - current)
                    new_position[j] = (self.c * population[i].position[j] + 
                                      np.random.random() * (guru.position[j] - population[i].position[j]))
                
                # Apply bounds
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Evaluate new fitness
                new_fitness = self.objective_func(new_position)
                
                # Greedy selection: accept if better
                if self._is_better(Member(new_position, new_fitness), population[i]):
                    population[i].position = new_position
                    population[i].fitness = new_fitness
                
                new_population.append(population[i].copy())
            
            population = new_population
            
            # Phase 2: Learner Phase (mutual learning with random exploration)
            sorted_population, _ = self._sort_population(population)
            global_best = sorted_population[0].copy()  # Global best for guidance
            
            # Create new population for learner phase
            new_population_learner = []
            for i in range(search_agents_no):
                # Choose a random partner different from current individual
                r1 = np.random.randint(0, search_agents_no)
                while r1 == i:
                    r1 = np.random.randint(0, search_agents_no)
                
                if population[i].fitness < population[r1].fitness if not self.maximize else population[i].fitness > population[r1].fitness:
                    # Current individual is better than random partner
                    if np.random.random() > self.sap:
                        # Learning strategy: current + rand*(current - partner) + rand*(global_best - current)
                        new_position = np.zeros(self.dim)
                        for j in range(self.dim):
                            new_position[j] = (population[i].position[j] + 
                                              np.random.random() * (population[i].position[j] - population[r1].position[j]) +
                                              np.random.random() * (global_best.position[j] - population[i].position[j]))
                    else:
                        # Random exploration
                        new_position = np.random.uniform(self.lb, self.ub, self.dim)
                else:
                    # Current individual is worse than random partner
                    # Learning strategy: current + rand*(partner - current) + rand*(global_best - current)
                    new_position = np.zeros(self.dim)
                    for j in range(self.dim):
                        new_position[j] = (population[i].position[j] + 
                                          np.random.random() * (population[r1].position[j] - population[i].position[j]) +
                                          np.random.random() * (global_best.position[j] - population[i].position[j]))
                
                # Apply bounds
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Evaluate new fitness
                new_fitness = self.objective_func(new_position)
                
                # Greedy selection: accept if better
                if self._is_better(Member(new_position, new_fitness), population[i]):
                    population[i].position = new_position
                    population[i].fitness = new_fitness
                
                new_population_learner.append(population[i].copy())
            
            population = new_population_learner
            
            # Update best solution
            sorted_population, _ = self._sort_population(population)
            current_best = sorted_population[0]
            if self._is_better(current_best, best_solution):
                best_solution = current_best.copy()
            
            # Store the best solution at this iteration
            history_step_solver.append(best_solution.copy())
            
            # Call the callbacks
            self._callbacks(iter, max_iter, best_solution)
        
        # Final evaluation and storage
        self.history_step_solver = history_step_solver
        self.best_solver = best_solution
        
        # Call the end function
        self._end_step_solver()
        return history_step_solver, best_solution
    
    def _sort_population(self, population):
        """Sort population based on fitness (best first)"""
        return sort_population(population, self.maximize)
