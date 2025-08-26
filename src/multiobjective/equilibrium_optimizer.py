import numpy as np
from typing import Callable, Union, Tuple, List
from tqdm import tqdm
from ._core import MultiObjectiveSolver, MultiObjectiveMember
from .._general import sort_population

class EquilibriumMultiMember(MultiObjectiveMember):
    def __init__(self, position: np.ndarray, fitness: np.ndarray):
        super().__init__(position, fitness)
    
    def copy(self):
        new_member = EquilibriumMultiMember(self.position.copy(), self.multi_fitness.copy())
        new_member.dominated = self.dominated
        new_member.grid_index = self.grid_index
        new_member.grid_sub_index = self.grid_sub_index
        return new_member

class MultiObjectiveEquilibriumOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Equilibrium Optimizer (EO) Algorithm.
    
    Multi-objective version of the Equilibrium Optimizer that handles
    optimization problems with multiple conflicting objectives. The algorithm
    maintains an archive of non-dominated solutions and uses grid-based
    selection to maintain diversity in the Pareto front.
    
    The algorithm uses an equilibrium pool consisting of:
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
        Initialize the Multi-Objective Equilibrium Optimizer.
        
        Args:
            objective_func (Callable): Objective function to optimize (returns array)
            lb (Union[float, np.ndarray]): Lower bounds of search space
            ub (Union[float, np.ndarray]): Upper bounds of search space
            dim (int): Number of dimensions in the problem
            maximize (bool): Whether to maximize (True) or minimize (False) objectives
            **kwargs: Additional algorithm parameters including:
                - a1 (float): Exploration parameter (default: 2)
                - a2 (float): Exploitation parameter (default: 1)
                - GP (float): Generation probability (default: 0.5)
                - archive_size (int): Maximum size of Pareto archive (default: 100)
        """
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        self.name_solver = "Multi-Objective Equilibrium Optimizer"
        
        # Algorithm-specific parameters with defaults
        self.a1 = kwargs.get('a1', 2)
        self.a2 = kwargs.get('a2', 1)
        self.GP = kwargs.get('GP', 0.5)
        
    def _init_population(self, search_agents_no) -> List[EquilibriumMultiMember]:
        """Initialize multi-objective population with custom member class"""
        population = []
        for _ in range(search_agents_no):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            fitness = self.objective_func(position)
            population.append(EquilibriumMultiMember(position, fitness))
        return population
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[EquilibriumMultiMember]]:
        """
        Execute the Multi-Objective Equilibrium Optimization Algorithm.
        
        The algorithm maintains an archive of non-dominated solutions and uses
        grid-based selection to maintain diversity in the Pareto front.
        
        Args:
            search_agents_no (int): Number of search agents
            max_iter (int): Maximum number of iterations for optimization
            
        Returns:
            Tuple[List, List[EquilibriumMultiMember]]: A tuple containing:
                - history_archive: List of archive states at each iteration
                - final_archive: Final archive of non-dominated solutions
        """
        # Initialize the population of search agents
        population = self._init_population(search_agents_no)
        
        # Initialize storage for archive history
        history_archive = []
        
        # Initialize equilibrium candidates (for guidance)
        Ceq1 = EquilibriumMultiMember(np.zeros(self.dim), np.full(self.n_objectives, np.inf if not self.maximize else -np.inf))
        Ceq2 = EquilibriumMultiMember(np.zeros(self.dim), np.full(self.n_objectives, np.inf if not self.maximize else -np.inf))
        Ceq3 = EquilibriumMultiMember(np.zeros(self.dim), np.full(self.n_objectives, np.inf if not self.maximize else -np.inf))
        Ceq4 = EquilibriumMultiMember(np.zeros(self.dim), np.full(self.n_objectives, np.inf if not self.maximize else -np.inf))
        
        # Memory for previous population and fitness
        C_old = [member.copy() for member in population]
        fit_old = np.array([member.multi_fitness for member in population])
        
        # Initialize archive with non-dominated solutions
        self._determine_domination(population)
        non_dominated = self._get_non_dominated_particles(population)
        self.archive.extend(non_dominated)
        
        # Build initial grid
        costs = self._get_fitness(self.archive)
        if costs.size > 0:
            self.grid = self._create_hypercubes(costs)
            for particle in self.archive:
                particle.grid_index, particle.grid_sub_index = self._get_grid_index(particle)
        
        # Call the begin function
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Update equilibrium candidates (for single best guidance)
            for i, member in enumerate(population):
                # Ensure positions stay within bounds
                member.position = np.clip(member.position, self.lb, self.ub)
                member.multi_fitness = self.objective_func(member.position)
                
                # Update equilibrium candidates based on Pareto dominance
                if self._dominates(member, Ceq1):
                    Ceq4 = Ceq3.copy()
                    Ceq3 = Ceq2.copy()
                    Ceq2 = Ceq1.copy()
                    Ceq1 = member.copy()
                elif not self._dominates(Ceq1, member) and self._dominates(member, Ceq2):
                    Ceq4 = Ceq3.copy()
                    Ceq3 = Ceq2.copy()
                    Ceq2 = member.copy()
                elif not self._dominates(Ceq2, member) and self._dominates(member, Ceq3):
                    Ceq4 = Ceq3.copy()
                    Ceq3 = member.copy()
                elif not self._dominates(Ceq3, member) and self._dominates(member, Ceq4):
                    Ceq4 = member.copy()
            
            # ----------------- Memory saving -----------------
            if iter == 0:
                fit_old = np.array([member.multi_fitness for member in population])
                C_old = [member.copy() for member in population]
            
            # Update population based on memory using Pareto dominance
            current_fitness = np.array([member.multi_fitness for member in population])
            for i in range(search_agents_no):
                old_member = EquilibriumMultiMember(C_old[i].position, fit_old[i])
                if self._dominates(old_member, population[i]):
                    population[i].multi_fitness = fit_old[i]
                    population[i].position = C_old[i].position.copy()
            
            # Update memory
            C_old = [member.copy() for member in population]
            fit_old = np.array([member.multi_fitness for member in population])
            # -------------------------------------------------
            
            # Create equilibrium pool from archive leaders
            if len(self.archive) >= 4:
                # Select 4 diverse leaders from archive
                leaders = self._select_multiple_leaders(4)
                Ceq1_arch = leaders[0] if len(leaders) > 0 else Ceq1
                Ceq2_arch = leaders[1] if len(leaders) > 1 else Ceq2
                Ceq3_arch = leaders[2] if len(leaders) > 2 else Ceq3
                Ceq4_arch = leaders[3] if len(leaders) > 3 else Ceq4
                
                # Create average candidate
                Ceq_ave_pos = (Ceq1_arch.position + Ceq2_arch.position + Ceq3_arch.position + Ceq4_arch.position) / 4
                Ceq_ave_fit = self.objective_func(Ceq_ave_pos)
                Ceq_ave = EquilibriumMultiMember(Ceq_ave_pos, Ceq_ave_fit)
                
                Ceq_pool = [Ceq1_arch, Ceq2_arch, Ceq3_arch, Ceq4_arch, Ceq_ave]
            else:
                # Fallback to original candidates if archive is small
                Ceq_ave_pos = (Ceq1.position + Ceq2.position + Ceq3.position + Ceq4.position) / 4
                Ceq_ave_fit = self.objective_func(Ceq_ave_pos)
                Ceq_ave = EquilibriumMultiMember(Ceq_ave_pos, Ceq_ave_fit)
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
                population[i].multi_fitness = self.objective_func(new_position)
            
            # Update archive with new solutions
            self._add_to_archive(population)
            
            # Store archive history
            history_archive.append([member.copy() for member in self.archive])
            
            # Call the callbacks
            best_leader = self._select_leader()
            self._callbacks(iter, max_iter, best_leader)
        
        # Final evaluation
        self.history_step_solver = history_archive
        
        # Call the end function
        self._end_step_solver()
        return history_archive, self.archive
