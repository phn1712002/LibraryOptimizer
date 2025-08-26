import numpy as np
from typing import Callable, Union, Tuple, List
from tqdm import tqdm
from ._core import Solver, Member
from ._general import sort_population

class EquilibriumOptimizer(Solver):
    """
    Equilibrium Optimizer (EO) Algorithm.
    
    EO is a physics-inspired optimization algorithm that mimics the control volume
    mass balance model to estimate both dynamic and equilibrium states. The algorithm
    uses equilibrium candidates to guide the search process towards optimal solutions.
    
    The algorithm maintains an equilibrium pool consisting of:
    - 4 best candidates (Ceq1, Ceq2, Ceq3, Ceq4)
    - 1 average candidate (Ceq_ave)
    
    References:
        Faramarzi, A., Heidarinejad, M., Stephens, B., & Mirjalili, S. (2020).
        Equilibrium optimizer: A novel optimization algorithm.
        Knowledge-Based Systems, 191, 105190.
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        """
        Initialize the Equilibrium Optimizer.
        
        Args:
            objective_func (Callable): Objective function to optimize
            lb (Union[float, np.ndarray]): Lower bounds of search space
            ub (Union[float, np.ndarray]): Upper bounds of search space
            dim (int): Number of dimensions in the problem
            maximize (bool): Whether to maximize (True) or minimize (False) objective
            **kwargs: Additional algorithm parameters including:
                - a1 (float): Exploration parameter (default: 2)
                - a2 (float): Exploitation parameter (default: 1)
                - GP (float): Generation probability (default: 0.5)
        """
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        # Store additional parameters for later use
        self.kwargs = kwargs
        self.name_solver = "Equilibrium Optimizer"
        
        # Algorithm-specific parameters with defaults
        self.a1 = kwargs.get('a1', 2)
        self.a2 = kwargs.get('a2', 1)
        self.GP = kwargs.get('GP', 0.5)
        
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, Member]:
        """
        Execute the Equilibrium Optimization Algorithm.
        
        The algorithm uses an equilibrium pool of candidates to guide the search
        process towards optimal solutions through physics-inspired update rules.
        
        Args:
            search_agents_no (int): Number of search agents
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
        
        # Initialize equilibrium candidates
        Ceq1 = Member(np.zeros(self.dim), np.inf if not self.maximize else -np.inf)
        Ceq2 = Member(np.zeros(self.dim), np.inf if not self.maximize else -np.inf)
        Ceq3 = Member(np.zeros(self.dim), np.inf if not self.maximize else -np.inf)
        Ceq4 = Member(np.zeros(self.dim), np.inf if not self.maximize else -np.inf)
        
        # Memory for previous population and fitness
        C_old = [member.copy() for member in population]
        fit_old = np.array([member.fitness for member in population])
        
        # Call the begin function
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Update equilibrium candidates
            for i, member in enumerate(population):
                # Ensure positions stay within bounds
                member.position = np.clip(member.position, self.lb, self.ub)
                member.fitness = self.objective_func(member.position)
                
                # Update equilibrium candidates based on fitness
                if self._is_better(member, Ceq1):
                    Ceq4 = Ceq3.copy()
                    Ceq3 = Ceq2.copy()
                    Ceq2 = Ceq1.copy()
                    Ceq1 = member.copy()
                elif self._is_better(member, Ceq2) and not self._is_better(member, Ceq1):
                    Ceq4 = Ceq3.copy()
                    Ceq3 = Ceq2.copy()
                    Ceq2 = member.copy()
                elif self._is_better(member, Ceq3) and not self._is_better(member, Ceq2):
                    Ceq4 = Ceq3.copy()
                    Ceq3 = member.copy()
                elif self._is_better(member, Ceq4) and not self._is_better(member, Ceq3):
                    Ceq4 = member.copy()
            
            # ----------------- Memory saving -----------------
            if iter == 0:
                fit_old = np.array([member.fitness for member in population])
                C_old = [member.copy() for member in population]
            
            # Update population based on memory
            current_fitness = np.array([member.fitness for member in population])
            for i in range(search_agents_no):
                if (not self.maximize and fit_old[i] < current_fitness[i]) or \
                   (self.maximize and fit_old[i] > current_fitness[i]):
                    population[i].fitness = fit_old[i]
                    population[i].position = C_old[i].position.copy()
            
            # Update memory
            C_old = [member.copy() for member in population]
            fit_old = np.array([member.fitness for member in population])
            # -------------------------------------------------
            
            # Create equilibrium pool
            Ceq_ave = Member((Ceq1.position + Ceq2.position + Ceq3.position + Ceq4.position) / 4, 0)
            Ceq_pool = [Ceq1, Ceq2, Ceq3, Ceq4, Ceq_ave]
            
            # Compute time parameter
            t = (1 - iter / max_iter) ** (self.a2 * iter / max_iter)
            
            # Update all search agents
            positions = np.array([member.position for member in population])
            
            for i in range(search_agents_no):
                # Randomly select one candidate from the pool
                Ceq = np.random.choice(Ceq_pool)
                
                # Generate random vectors
                lambda_vec = np.random.random(self.dim)
                r = np.random.random(self.dim)
                
                # Compute F parameter
                F = self.a1 * np.sign(r - 0.5) * (np.exp(-lambda_vec * t) - 1)
                
                # Compute generation control parameter
                r1 = np.random.random()
                r2 = np.random.random()
                GCP = 0.5 * r1 * np.ones(self.dim) * (r2 >= self.GP)
                
                # Compute generation rate
                G0 = GCP * (Ceq.position - lambda_vec * population[i].position)
                G = G0 * F
                
                # Update position using EO equation
                new_position = Ceq.position + \
                              (population[i].position - Ceq.position) * F + \
                              (G / (lambda_vec * 1.0)) * (1 - F)
                
                # Ensure positions stay within bounds
                new_position = np.clip(new_position, self.lb, self.ub)
                population[i].position = new_position
                population[i].fitness = self.objective_func(new_position)
            
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
