import numpy as np
from typing import Callable, Union, Tuple, List
from tqdm import tqdm
from ._core import MultiObjectiveSolver, MultiObjectiveMember
from ..marinepredators_optimizer import levy_flight

class MultiObjectiveMarinePredatorsOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Marine Predators Algorithm (MPA) Optimizer.
    
    Multi-objective version of the Marine Predators Algorithm that handles
    optimization problems with multiple conflicting objectives. The algorithm
    maintains an archive of non-dominated solutions and uses grid-based
    selection to maintain diversity in the Pareto front.
    
    The algorithm follows the same three phases as the single-objective version:
    1. Phase 1 (Iter < Max_iter/3): High velocity ratio - Brownian motion
    2. Phase 2 (Max_iter/3 < Iter < 2*Max_iter/3): Unit velocity ratio - Mixed strategy
    3. Phase 3 (Iter > 2*Max_iter/3): Low velocity ratio - Levy flight
    
    References:
        Faramarzi, A., Heidarinejad, M., Mirjalili, S., & Gandomi, A. H. (2020).
        Marine Predators Algorithm: A Nature-inspired Metaheuristic.
        Expert Systems with Applications, 152, 113377.
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        """
        Initialize the Multi-Objective Marine Predators Algorithm Optimizer.
        
        Args:
            objective_func (Callable): Objective function to optimize (returns array)
            lb (Union[float, np.ndarray]): Lower bounds of search space
            ub (Union[float, np.ndarray]): Upper bounds of search space
            dim (int): Number of dimensions in the problem
            maximize (bool): Whether to maximize (True) or minimize (False) objectives
            **kwargs: Additional algorithm parameters including:
                - FADs (float): Fish Aggregating Devices effect probability (default: 0.2)
                - P (float): Memory rate parameter (default: 0.5)
                - archive_size (int): Maximum size of Pareto archive (default: 100)
        """
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        self.name_solver = "Multi-Objective Marine Predators Algorithm"
        
        # Algorithm-specific parameters with defaults
        self.FADs = kwargs.get('FADs', 0.2)
        self.P = kwargs.get('P', 0.5)
        
    def _init_population(self, search_agents_no) -> List[MultiObjectiveMember]:
        """Initialize multi-objective population with custom member class"""
        population = []
        for _ in range(search_agents_no):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            fitness = self.objective_func(position)
            population.append(MultiObjectiveMember(position, fitness))
        return population
    
    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Execute the Multi-Objective Marine Predators Algorithm.
        
        The algorithm maintains an archive of non-dominated solutions and uses
        grid-based selection to maintain diversity in the Pareto front.
        
        Args:
            search_agents_no (int): Number of search agents (predators/prey)
            max_iter (int): Maximum number of iterations for optimization
            
        Returns:
            Tuple[List, List[MultiObjectiveMember]]: A tuple containing:
                - history_archive: List of archive states at each iteration
                - final_archive: Final archive of non-dominated solutions
        """
        # Initialize the population of search agents
        population = self._init_population(search_agents_no)
        
        # Initialize storage for archive history
        history_archive = []
        
        # Memory for previous population and fitness
        Prey_old = [member.copy() for member in population]
        fit_old = np.array([member.multi_fitness for member in population])
        
        # Initialize top predator (for guidance)
        Top_predator_pos = np.zeros(self.dim)
        Top_predator_fit = np.full(self.n_objectives, np.inf if not self.maximize else -np.inf)
        
        # Boundary matrices for FADs effect
        Xmin = np.tile(self.lb, (search_agents_no, 1))
        Xmax = np.tile(self.ub, (search_agents_no, 1))
        
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
            # ------------------- Detecting top predator -------------------
            for i, member in enumerate(population):
                # Ensure positions stay within bounds
                member.position = np.clip(member.position, self.lb, self.ub)
                member.multi_fitness = self.objective_func(member.position)
                
                # Update top predator (for single best guidance)
                if self._dominates(member, MultiObjectiveMember(Top_predator_pos, Top_predator_fit)):
                    Top_predator_fit = member.multi_fitness.copy()
                    Top_predator_pos = member.position.copy()
            
            # ------------------- Marine Memory saving -------------------
            if iter == 0:
                fit_old = np.array([member.multi_fitness for member in population])
                Prey_old = [member.copy() for member in population]
            
            # Update population based on memory
            current_fitness = np.array([member.multi_fitness for member in population])
            
            # For multi-objective, we need to compare using Pareto dominance
            Inx = np.zeros(len(population), dtype=bool)
            for i in range(len(population)):
                old_member = MultiObjectiveMember(Prey_old[i].position, fit_old[i])
                if self._dominates(old_member, population[i]):
                    Inx[i] = True
            
            Indx = np.tile(Inx[:, np.newaxis], (1, self.dim))
            
            # Update positions based on memory
            positions = np.array([member.position for member in population])
            old_positions = np.array([member.position for member in Prey_old])
            new_positions = np.where(Indx, old_positions, positions)
            
            # Update fitness based on memory
            new_fitness = np.where(Inx[:, np.newaxis], fit_old, current_fitness)
            
            # Update population
            for i in range(search_agents_no):
                population[i].position = new_positions[i]
                population[i].multi_fitness = new_fitness[i]
            
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
                # Re-evaluate fitness after position update
                population[i].multi_fitness = self.objective_func(population[i].position)
            
            # ------------------- Detecting top predator -------------------
            for i, member in enumerate(population):
                # Ensure positions stay within bounds
                member.position = np.clip(member.position, self.lb, self.ub)
                member.multi_fitness = self.objective_func(member.position)
                
                # Update top predator
                if self._dominates(member, MultiObjectiveMember(Top_predator_pos, Top_predator_fit)):
                    Top_predator_fit = member.multi_fitness.copy()
                    Top_predator_pos = member.position.copy()
            
            # ----------------------- Marine Memory saving ----------------
            current_fitness = np.array([member.multi_fitness for member in population])
            
            # For multi-objective, we need to compare using Pareto dominance
            Inx = np.zeros(len(population), dtype=bool)
            for i in range(len(population)):
                old_member = MultiObjectiveMember(Prey_old[i].position, fit_old[i])
                if self._dominates(old_member, population[i]):
                    Inx[i] = True
            
            Indx = np.tile(Inx[:, np.newaxis], (1, self.dim))
            
            # Update positions based on memory
            positions = np.array([member.position for member in population])
            old_positions = np.array([member.position for member in Prey_old])
            new_positions = np.where(Indx, old_positions, positions)
            
            # Update fitness based on memory
            new_fitness = np.where(Inx[:, np.newaxis], fit_old, current_fitness)
            
            # Update population
            for i in range(search_agents_no):
                population[i].position = new_positions[i]
                population[i].multi_fitness = new_fitness[i]
            
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
            
            # Update population positions and re-evaluate fitness
            for i in range(search_agents_no):
                population[i].position = positions[i]
                population[i].multi_fitness = self.objective_func(population[i].position)
            
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

    def _levy_flight(self, n, m, beta):
        return levy_flight(n, m, beta)