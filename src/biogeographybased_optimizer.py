import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import Solver, Member
from ._general import roulette_wheel_selection, sort_population


class BiogeographyBasedOptimizer(Solver):
    """
    Biogeography-Based Optimization (BBO) algorithm.
    
    BBO is a population-based optimization algorithm inspired by the migration
    of species between habitats in biogeography. It models how species migrate
    between islands based on habitat suitability index (HSI).
    
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
        - keep_rate: Rate of habitats to keep (default: 0.2)
        - alpha: Migration coefficient (default: 0.9)
        - p_mutation: Mutation probability (default: 0.1)
        - sigma: Mutation step size (default: 2% of variable range)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set solver name
        self.name_solver = "Biogeography Based Optimizer"
        
        # Set algorithm parameters with defaults
        self.keep_rate = kwargs.get('keep_rate', 0.2)
        self.alpha = kwargs.get('alpha', 0.9)
        self.p_mutation = kwargs.get('p_mutation', 0.1)
        
        # Calculate sigma as 2% of variable range if not provided
        var_range = self.ub - self.lb
        self.sigma = kwargs.get('sigma', 0.02 * var_range)
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Main optimization method for BBO algorithm.
        
        Parameters:
        -----------
        search_agents_no : int
            Number of habitats (population size)
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List, Member]
            History of best solutions and the best solution found
        """
        # Initialize storage variables
        history_step_solver = []
        
        # Calculate derived parameters
        n_keep = round(self.keep_rate * search_agents_no)  # Number of habitats to keep
        n_new = search_agents_no - n_keep  # Number of new habitats
        
        # Initialize migration rates (emigration and immigration)
        mu = np.linspace(1, 0, search_agents_no)  # Emigration rates (decreasing)
        lambda_rates = 1 - mu  # Immigration rates (increasing)
        
        # Initialize population
        population = self._init_population(search_agents_no)
        
        # Sort initial population and get best solution
        sorted_population, _ = self._sort_population(population)
        best_solution = sorted_population[0].copy()
        
        # Start solver (show progress bar)
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Create new population for this iteration
            new_population = [member.copy() for member in population]
            
            # Update each habitat
            for i in range(search_agents_no):
                # Migration phase for each dimension
                for k in range(self.dim):
                    # Immigration: if random number <= immigration rate
                    if np.random.random() <= lambda_rates[i]:
                        # Calculate emigration probabilities (excluding current habitat)
                        ep = mu.copy()
                        ep[i] = 0  # Set current habitat probability to 0
                        ep_sum = np.sum(ep)
                        
                        if ep_sum > 0:
                            ep = ep / ep_sum  # Normalize probabilities
                            
                            # Select source habitat using roulette wheel selection
                            j = self._roulette_wheel_selection(ep)
                            
                            # Perform migration
                            new_population[i].position[k] = (
                                population[i].position[k] + 
                                self.alpha * (population[j].position[k] - population[i].position[k])
                            )
                    
                    # Mutation: if random number <= mutation probability
                    if np.random.random() <= self.p_mutation:
                        new_population[i].position[k] += self.sigma[k] * np.random.randn()
                
                # Apply bounds constraints
                new_population[i].position = np.clip(new_population[i].position, self.lb, self.ub)
                
                # Evaluate new fitness
                new_population[i].fitness = self.objective_func(new_population[i].position)
            
            # Sort new population
            sorted_new_population, _ = self._sort_population(new_population)
            
            # Select next iteration population: keep best + new solutions
            next_population = sorted_population[:n_keep] + sorted_new_population[:n_new]
            
            # Sort the combined population
            sorted_next_population, _ = self._sort_population(next_population)
            population = sorted_next_population
            
            # Update best solution
            current_best = population[0]
            if self._is_better(current_best, best_solution):
                best_solution = current_best.copy()
            
            # Save history
            history_step_solver.append(best_solution.copy())
            
            # Call callback for progress tracking
            self._callbacks(iter, max_iter, best_solution)
        
        # End solver
        self.history_step_solver = history_step_solver
        self.best_solver = best_solution
        self._end_step_solver()
        
        return history_step_solver, best_solution
    
    def _sort_population(self, population):
        """
        Sort the population based on fitness.
        """
        return sort_population(population, self.maximize)

    def _roulette_wheel_selection(self, probabilities):
        """Perform roulette wheel selection (fitness proportionate selection)."""
        return roulette_wheel_selection(probabilities)