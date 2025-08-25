import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember

class MultiObjectiveModifiedSocialGroupOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Modified Social Group Optimization (MSGO) algorithm.
    
    A metaheuristic optimization algorithm inspired by social group behavior,
    featuring guru phase (learning from best) and learner phase (mutual learning)
    adapted for multi-objective optimization using archive management and grid-based selection.
    
    Parameters:
    -----------
    objective_func : Callable
        Objective function that returns a list of fitness values
    lb : Union[float, np.ndarray]
        Lower bounds for variables
    ub : Union[float, np.ndarray]
        Upper bounds for variables
    dim : int
        Problem dimension
    **kwargs
        Additional parameters:
        - archive_size: Size of the external archive (default: 100)
        - alpha: Grid inflation parameter (default: 0.1)
        - n_grid: Number of grids per dimension (default: 7)
        - beta: Leader selection pressure (default: 2)
        - gamma: Archive removal pressure (default: 2)
        - c: Learning coefficient for guru phase (default: 0.2)
        - sap: Self-adaptive probability for random exploration (default: 0.7)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Modified Social Group Optimizer"
        
        # MSGO-specific parameters
        self.c = kwargs.get('c', 0.2)  # Learning coefficient for guru phase
        self.sap = kwargs.get('sap', 0.7)  # Self-adaptive probability for random exploration

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Main optimization method for multi-objective MSGO
        
        Parameters:
        -----------
        search_agents_no : int
            Number of search agents (population size)
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List, List[MultiObjectiveMember]]
            History of archive states and the final archive
        """
        # Initialize storage
        history_archive = []
        
        # Initialize population
        population = self._init_population(search_agents_no)
        
        # Initialize archive with non-dominated solutions
        self._determine_domination(population)
        non_dominated = self._get_non_dominated_particles(population)
        self.archive.extend(non_dominated)
        
        # Initialize grid for archive
        costs = self._get_costs(self.archive)
        if costs.size > 0:
            self.grid = self._create_hypercubes(costs)
            for particle in self.archive:
                particle.grid_index, particle.grid_sub_index = self._get_grid_index(particle)
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Phase 1: Guru Phase (improvement due to best person)
            # Select guru from archive using grid-based selection
            guru = self._select_leader()
            if guru is None:
                # If no leader in archive, use random member from population
                guru = np.random.choice(population)
            
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
                
                # Create new member for comparison
                new_member = MultiObjectiveMember(new_position, new_fitness)
                
                # Greedy selection: accept if new solution dominates current or is non-dominated
                if self._dominates(new_member, population[i]) or not self._dominates(population[i], new_member):
                    population[i].position = new_position
                    population[i].multi_fitness = new_fitness
                
                new_population.append(population[i].copy())
            
            population = new_population
            
            # Phase 2: Learner Phase (mutual learning with random exploration)
            # Select global best from archive for guidance
            global_best = self._select_leader()
            if global_best is None:
                # If no leader in archive, use random member from population
                global_best = np.random.choice(population)
            
            # Create new population for learner phase
            new_population_learner = []
            for i in range(search_agents_no):
                # Choose a random partner different from current individual
                r1 = np.random.randint(0, search_agents_no)
                while r1 == i:
                    r1 = np.random.randint(0, search_agents_no)
                
                # Check dominance between current and random partner
                current_dominates_partner = self._dominates(population[i], population[r1])
                partner_dominates_current = self._dominates(population[r1], population[i])
                
                if current_dominates_partner and not partner_dominates_current:
                    # Current individual dominates partner
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
                    # Current individual is dominated by or non-dominated with partner
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
                
                # Create new member for comparison
                new_member = MultiObjectiveMember(new_position, new_fitness)
                
                # Greedy selection: accept if new solution dominates current or is non-dominated
                if self._dominates(new_member, population[i]) or not self._dominates(population[i], new_member):
                    population[i].position = new_position
                    population[i].multi_fitness = new_fitness
                
                new_population_learner.append(population[i].copy())
            
            population = new_population_learner
            
            # Update archive with current population
            self._add_to_archive(population)
            
            # Store archive state for history
            history_archive.append([member.copy() for member in self.archive])
            
            # Update progress
            self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
        
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver
        self._end_step_solver()
        
        return history_archive, self.archive
