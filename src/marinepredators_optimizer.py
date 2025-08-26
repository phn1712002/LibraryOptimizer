import numpy as np
from typing import Callable, Union, Tuple, List
from tqdm import tqdm
from ._core import Solver, Member
from ._general import sort_population

def levy_flight(n: int, m: int, beta: float = 1.5) -> np.ndarray:
    """
    Generate Levy flight random numbers.
    
    Args:
        n: Number of samples
        m: Number of dimensions
        beta: Power law index (1 < beta < 2)
        
    Returns:
        np.ndarray: Levy flight random numbers of shape (n, m)
    """
    num = np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = np.math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma_u = (num / den) ** (1 / beta)
    
    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, 1, (n, m))
    
    return u / (np.abs(v) ** (1 / beta))

class MarinePredatorsOptimizer(Solver):
    """
    Marine Predators Algorithm (MPA) Optimizer.
    
    MPA is a nature-inspired metaheuristic optimization algorithm that mimics
    the optimal foraging strategy and encounter rate policy between predator 
    and prey in marine ecosystems. The algorithm follows three main phases:
    
    1. Phase 1 (Iter < Max_iter/3): High velocity ratio - Brownian motion
    2. Phase 2 (Max_iter/3 < Iter < 2*Max_iter/3): Unit velocity ratio - Mixed strategy
    3. Phase 3 (Iter > 2*Max_iter/3): Low velocity ratio - Levy flight
    
    The algorithm also includes environmental effects like FADs and eddy formation.
    
    References:
        Faramarzi, A., Heidarinejad, M., Mirjalili, S., & Gandomi, A. H. (2020).
        Marine Predators Algorithm: A Nature-inspired Metaheuristic.
        Expert Systems with Applications, 152, 113377.
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        """
        Initialize the Marine Predators Algorithm Optimizer.
        
        Args:
            objective_func (Callable): Objective function to optimize
            lb (Union[float, np.ndarray]): Lower bounds of search space
            ub (Union[float, np.ndarray]): Upper bounds of search space
            dim (int): Number of dimensions in the problem
            maximize (bool): Whether to maximize (True) or minimize (False) objective
            **kwargs: Additional algorithm parameters including:
                - FADs (float): Fish Aggregating Devices effect probability (default: 0.2)
                - P (float): Memory rate parameter (default: 0.5)
        """
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        # Store additional parameters for later use
        self.kwargs = kwargs
        self.name_solver = "Marine Predators Algorithm"
        
        # Algorithm-specific parameters with defaults
        self.FADs = kwargs.get('FADs', 0.2)
        self.P = kwargs.get('P', 0.5)
        
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Execute the Marine Predators Algorithm.
        
        The algorithm follows three main phases based on velocity ratio:
        1. High velocity ratio (Iter < Max_iter/3): Brownian motion
        2. Unit velocity ratio (Max_iter/3 < Iter < 2*Max_iter/3): Mixed strategy
        3. Low velocity ratio (Iter > 2*Max_iter/3): Levy flight
        
        Args:
            search_agents_no (int): Number of search agents (predators/prey)
            max_iter (int): Maximum number of iterations for optimization
            
        Returns:
            Tuple[List, Member]: A tuple containing:
                - history_step_solver: List of best solutions at each iteration
                - best_solver: Best solution found overall
        """
        # Initialize the population of search agents
        population = self._init_population(search_agents_no)
        
        # Initialize storage variables
        history_step_solver = []
        best_solver = self.best_solver
        
        # Memory for previous population and fitness
        Prey_old = [member.copy() for member in population]
        fit_old = np.array([member.fitness for member in population])
        
        # Initialize top predator
        Top_predator_pos = np.zeros(self.dim)
        Top_predator_fit = np.inf if not self.maximize else -np.inf
        
        # Boundary matrices for FADs effect
        Xmin = np.tile(self.lb, (search_agents_no, 1))
        Xmax = np.tile(self.ub, (search_agents_no, 1))
        
        # Call the begin function
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # ------------------- Detecting top predator -------------------
            for i, member in enumerate(population):
                # Ensure positions stay within bounds
                member.position = np.clip(member.position, self.lb, self.ub)
                member.fitness = self.objective_func(member.position)
                
                # Update top predator
                if self._is_better(member, Member(Top_predator_pos, Top_predator_fit)):
                    Top_predator_fit = member.fitness
                    Top_predator_pos = member.position.copy()
            
            # ------------------- Marine Memory saving -------------------
            if iter == 0:
                fit_old = np.array([member.fitness for member in population])
                Prey_old = [member.copy() for member in population]
            
            # Update population based on memory
            current_fitness = np.array([member.fitness for member in population])
            Inx = fit_old < current_fitness if not self.maximize else fit_old > current_fitness
            Indx = np.tile(Inx[:, np.newaxis], (1, self.dim))
            
            # Update positions based on memory
            positions = np.array([member.position for member in population])
            old_positions = np.array([member.position for member in Prey_old])
            new_positions = np.where(Indx, old_positions, positions)
            
            # Update fitness based on memory
            new_fitness = np.where(Inx, fit_old, current_fitness)
            
            # Update population
            for i in range(search_agents_no):
                population[i].position = new_positions[i]
                population[i].fitness = new_fitness[i]
            
            # Update memory
            fit_old = new_fitness.copy()
            Prey_old = [member.copy() for member in population]
            
            # ------------------------------------------------------------
            
            # Create elite matrix (replicate top predator)
            Elite = np.tile(Top_predator_pos, (search_agents_no, 1))
            
            # Compute convergence factor
            CF = (1 - iter / max_iter) ** (2 * iter / max_iter)
            
            # Generate random vectors
            RL = 0.05 * self._levy_flight(search_agents_no, self.dim, 1.5)  # Levy flight
            RB = np.random.randn(search_agents_no, self.dim)           # Brownian motion
            
            # Update positions based on current phase
            positions = np.array([member.position for member in population])
            
            for i in range(search_agents_no):
                for j in range(self.dim):
                    R = np.random.random()
                    
                    # ------------------- Phase 1 (Eq.12) -------------------
                    if iter < max_iter / 3:
                        stepsize = RB[i, j] * (Elite[i, j] - RB[i, j] * positions[i, j])
                        positions[i, j] = positions[i, j] + self.P * R * stepsize
                    
                    # --------------- Phase 2 (Eqs. 13 & 14)----------------
                    elif iter < 2 * max_iter / 3:
                        if i > search_agents_no / 2:
                            stepsize = RB[i, j] * (RB[i, j] * Elite[i, j] - positions[i, j])
                            positions[i, j] = Elite[i, j] + self.P * CF * stepsize
                        else:
                            stepsize = RL[i, j] * (Elite[i, j] - RL[i, j] * positions[i, j])
                            positions[i, j] = positions[i, j] + self.P * R * stepsize
                    
                    # ------------------ Phase 3 (Eq. 15)-------------------
                    else:
                        stepsize = RL[i, j] * (RL[i, j] * Elite[i, j] - positions[i, j])
                        positions[i, j] = Elite[i, j] + self.P * CF * stepsize
            
            # Update population positions
            for i in range(search_agents_no):
                population[i].position = positions[i]
            
            # ------------------- Detecting top predator -------------------
            for i, member in enumerate(population):
                # Ensure positions stay within bounds
                member.position = np.clip(member.position, self.lb, self.ub)
                member.fitness = self.objective_func(member.position)
                
                # Update top predator
                if self._is_better(member, Member(Top_predator_pos, Top_predator_fit)):
                    Top_predator_fit = member.fitness
                    Top_predator_pos = member.position.copy()
            
            # ----------------------- Marine Memory saving ----------------
            current_fitness = np.array([member.fitness for member in population])
            Inx = fit_old < current_fitness if not self.maximize else fit_old > current_fitness
            Indx = np.tile(Inx[:, np.newaxis], (1, self.dim))
            
            # Update positions based on memory
            positions = np.array([member.position for member in population])
            old_positions = np.array([member.position for member in Prey_old])
            new_positions = np.where(Indx, old_positions, positions)
            
            # Update fitness based on memory
            new_fitness = np.where(Inx, fit_old, current_fitness)
            
            # Update population
            for i in range(search_agents_no):
                population[i].position = new_positions[i]
                population[i].fitness = new_fitness[i]
            
            # Update memory
            fit_old = new_fitness.copy()
            Prey_old = [member.copy() for member in population]
            
            # ---------- Eddy formation and FADs' effect (Eq 16) -----------
            if np.random.random() < self.FADs:
                # FADs effect
                U = np.random.rand(search_agents_no, self.dim) < self.FADs
                random_positions = Xmin + np.random.rand(search_agents_no, self.dim) * (Xmax - Xmin)
                positions = positions + CF * random_positions * U
            else:
                # Eddy formation effect
                r = np.random.random()
                Rs = search_agents_no
                idx1 = np.random.permutation(Rs)
                idx2 = np.random.permutation(Rs)
                stepsize = (self.FADs * (1 - r) + r) * (positions[idx1] - positions[idx2])
                positions = positions + stepsize
            
            # Update population positions
            for i in range(search_agents_no):
                population[i].position = positions[i]
            
            # Update best solution
            sorted_population, _ = self._sort_population(population)
            current_best = sorted_population[0]
            if self._is_better(current_best, best_solver):
                best_solver = current_best.copy()
            
            # Store the best solution at this iteration
            history_step_solver.append(best_solver.copy())
            
            # Call the callbacks
            self._callbacks(iter, max_iter, best_solver)
        
        # Final evaluation
        self.history_step_solver = history_step_solver
        self.best_solver = best_solver
        
        # Call the end function
        self._end_step_solver()
        return history_step_solver, best_solver
    
    def _sort_population(self, population):
        """
        Sort the population based on fitness.
        """
        return sort_population(population, self.maximize)
    
    def _levy_flight(self, n, m, beta):
        return levy_flight(n, m, beta)
