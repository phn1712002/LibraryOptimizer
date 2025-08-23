import numpy as np
from typing import Callable, Union, Tuple, List
from .core import Solver, Member
from utils.general import roulette_wheel_selection, sort_population

class Bee(Member):
    def __init__(self, position: np.ndarray, fitness: float, trial: int = 0):
        super().__init__(position, fitness)
        self.trial = trial  # Trial counter for abandonment
    
    def copy(self):
        return Bee(self.position.copy(), self.fitness, self.trial)
    
    def __str__(self):
        return f"Position: {self.position} - Fitness: {self.fitness} - Trial: {self.trial}"

class ArtificialBeeColonyOptimizer(Solver):
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize)
        # Store additional parameters for later use
        self.kwargs = kwargs
        
        # Set default ABC parameters
        self.name_solver = "Artificial Bee Colony Optimizer"
        self.n_onlooker = kwargs.get('n_onlooker', None)  # Number of onlooker bees
        self.abandonment_limit = kwargs.get('abandonment_limit', None)  # Trial limit (L)
        self.acceleration_coef = kwargs.get('acceleration_coef', 1.0)  # Acceleration coefficient (a)
        

    def _init_population(self, search_agents_no) -> List:
        population = []
        for _ in range(search_agents_no):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            fitness = self.objective_func(position)
            population.append(Bee(position, fitness, 0))
        return population

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Bee]:
       
        # Initialize storage variables
        history_step_solver = []
        

        # Initialize parameters
        if self.n_onlooker is None:
            self.n_onlooker = search_agents_no  # Default: same as population size
        
        if self.abandonment_limit is None:
            # Default: 60% of variable dimension * population size (as in MATLAB code)
            self.abandonment_limit = int(0.6 * self.dim * search_agents_no)
        
        # Initialize the population of bees
        population = self._init_population(search_agents_no)
        
        # Initialize best solution
        sorted_population, _ = self._sort_population(population)
        best_solution = sorted_population[0].copy()
        
        # Call the begin function
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Phase 1: Employed Bees
            for i in range(search_agents_no):
                # Choose a random neighbor different from current bee
                neighbors = [j for j in range(search_agents_no) if j != i]
                k = np.random.choice(neighbors)
                
                # Define acceleration coefficient
                phi = self.acceleration_coef * np.random.uniform(-1, 1, self.dim)
                
                # Generate new candidate solution
                new_position = population[i].position + phi * (population[i].position - population[k].position)
                
                # Apply bounds
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Evaluate new fitness
                new_fitness = self.objective_func(new_position)
                
                # Comparison (greedy selection)
                if self._is_better(Bee(new_position, new_fitness), population[i]):
                    population[i].position = new_position
                    population[i].fitness = new_fitness
                    population[i].trial = 0  # Reset trial counter
                else:
                    population[i].trial += 1  # Increase trial counter
            
            # Calculate fitness values and selection probabilities
            # Convert cost to fitness (for minimization problems, we need to invert)
            if not self.maximize:
                # For minimization, lower cost is better, so we use negative cost for fitness calculation
                costs = np.array([bee.fitness for bee in population])
                max_cost = np.max(costs)
                fitness_values = max_cost - costs + 1e-10  # Add small value to avoid division by zero
            else:
                # For maximization, higher fitness is already good
                fitness_values = np.array([bee.fitness for bee in population])
            
            # Normalize to get probabilities
            probabilities = fitness_values / np.sum(fitness_values)
            
            # Phase 2: Onlooker Bees
            for m in range(self.n_onlooker):
                # Select source site using roulette wheel selection
                i = roulette_wheel_selection(probabilities)
                
                # Choose a random neighbor different from current bee
                neighbors = [j for j in range(search_agents_no) if j != i]
                k = np.random.choice(neighbors)
                
                # Define acceleration coefficient
                phi = self.acceleration_coef * np.random.uniform(-1, 1, self.dim)
                
                # Generate new candidate solution
                new_position = population[i].position + phi * (population[i].position - population[k].position)
                
                # Apply bounds
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Evaluate new fitness
                new_fitness = self.objective_func(new_position)
                
                # Comparison (greedy selection)
                if self._is_better(Bee(new_position, new_fitness), population[i]):
                    population[i].position = new_position
                    population[i].fitness = new_fitness
                    population[i].trial = 0  # Reset trial counter
                else:
                    population[i].trial += 1  # Increase trial counter
            
            # Phase 3: Scout Bees
            for i in range(search_agents_no):
                if population[i].trial >= self.abandonment_limit:
                    # Abandon and replace with random solution
                    population[i].position = np.random.uniform(self.lb, self.ub, self.dim)
                    population[i].fitness = self.objective_func(population[i].position)
                    population[i].trial = 0  # Reset trial counter
            
            # Update best solution using _sort_population
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
        return sort_population(population, self.maximize)