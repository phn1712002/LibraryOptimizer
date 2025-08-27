import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember

class MultiObjectiveElectromagneticChargedParticlesOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Electromagnetic Charged Particles Optimization (ECPO) Algorithm.
    
    This algorithm extends the standard ECPO for multi-objective optimization
    using archive management and grid-based selection for maintaining diversity.
    
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
    maximize : bool
        Optimization direction (True: maximize, False: minimize)
    **kwargs
        Additional parameters:
        - strategy: Movement strategy (1, 2, or 3, default: 1)
        - npi: Number of particles for interaction (default: 2)
        
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize: bool = True, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Electromagnetic Charged Particles Optimizer"
        
        # Algorithm-specific parameters with defaults
        self.strategy = kwargs.get('strategy', 1)  # Movement strategy (1, 2, or 3)
        self.npi = kwargs.get('npi', 2)  # Number of particles for interaction
        
        # Validate parameters
        if self.strategy not in [1, 2, 3]:
            raise ValueError("Strategy must be 1, 2, or 3")
        if self.npi < 2:
            raise ValueError("NPI must be at least 2")

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[MultiObjectiveMember]]:
        """
        Main optimization method for multi-objective ECPO.
        
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
        costs = self._get_fitness(self.archive)
        if costs.size > 0:
            self.grid = self._create_hypercubes(costs)
            for particle in self.archive:
                particle.grid_index, particle.grid_sub_index = self._get_grid_index(particle)
        
        # Start solver
        self._begin_step_solver(max_iter)
        
        # Main optimization loop
        for iter in range(max_iter):
            # Generate new particles based on strategy
            new_particles = self._generate_new_particles(population, search_agents_no)
            
            # Evaluate new particles
            for particle in new_particles:
                particle.multi_fitness = self.objective_func(particle.position)
            
            # Update archive with new particles
            self._add_to_archive(new_particles)
            
            # Update population with new particles (replace worst ones)
            self._update_population(population, new_particles)
            
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
    
    def _generate_new_particles(self, population: List[MultiObjectiveMember], 
                               search_agents_no: int) -> List[MultiObjectiveMember]:
        """
        Generate new particles using electromagnetic force-based movement.
        
        Args:
            population (List[MultiObjectiveMember]): Current population
            search_agents_no (int): Population size
            
        Returns:
            List[MultiObjectiveMember]: List of newly generated particles
        """
        new_particles = []
        
        # Calculate population factor based on strategy
        if self.strategy == 1:
            pop_factor = 2 * self._n_choose_2(self.npi)
        elif self.strategy == 2:
            pop_factor = self.npi
        else:  # strategy == 3
            pop_factor = 2 * self._n_choose_2(self.npi) + self.npi
        
        # Number of iterations to generate enough particles
        n_iterations = int(np.ceil(search_agents_no / pop_factor))
        
        for _ in range(n_iterations):
            # Generate Gaussian force with mean=0.7, std=0.2
            force = np.random.normal(0.7, 0.2)
            
            # Select random particles for interaction
            selected_indices = np.random.choice(len(population), self.npi, replace=False)
            selected_particles = [population[i] for i in selected_indices]
            
            # Select leader from archive using grid-based selection
            leader = self._select_leader()
            if leader is None:
                # If no leader available, use best from selected particles
                leader = self._get_best_from_population(selected_particles)
            
            if self.strategy == 1:
                # Strategy 1: Pairwise interactions
                new_particles.extend(self._strategy_1(selected_particles, leader, force))
            elif self.strategy == 2:
                # Strategy 2: Combined interactions
                new_particles.extend(self._strategy_2(selected_particles, leader, force))
            else:  # strategy == 3
                # Strategy 3: Hybrid approach
                new_particles.extend(self._strategy_3(selected_particles, leader, force))
        
        # Ensure positions stay within bounds
        for particle in new_particles:
            particle.position = np.clip(particle.position, self.lb, self.ub)
        
        # Apply archive-based mutation
        self._apply_archive_mutation(new_particles)
        
        return new_particles
    
    def _strategy_1(self, selected_particles: List[MultiObjectiveMember], 
                   leader: MultiObjectiveMember, force: float) -> List[MultiObjectiveMember]:
        """Strategy 1: Pairwise interactions between particles."""
        new_particles = []
        
        for i in range(self.npi):
            for j in range(self.npi):
                if i == j:
                    continue
                
                # Base movement towards leader
                new_position = selected_particles[i].position.copy()
                new_position += force * (leader.position - selected_particles[i].position)
                
                # Add interaction with other particle
                if j < i:
                    new_position += force * (selected_particles[j].position - selected_particles[i].position)
                else:  # j > i
                    new_position -= force * (selected_particles[j].position - selected_particles[i].position)
                
                new_particles.append(MultiObjectiveMember(new_position, np.zeros(self.n_objectives)))
        
        return new_particles
    
    def _strategy_2(self, selected_particles: List[MultiObjectiveMember], 
                   leader: MultiObjectiveMember, force: float) -> List[MultiObjectiveMember]:
        """Strategy 2: Combined interactions from all particles."""
        new_particles = []
        
        for i in range(self.npi):
            new_position = selected_particles[i].position.copy()
            
            # Movement towards leader (no force for this component)
            new_position += 0 * force * (leader.position - selected_particles[i].position)
            
            # Combined interactions with all other particles
            for j in range(self.npi):
                if j < i:
                    new_position += force * (selected_particles[j].position - selected_particles[i].position)
                elif j > i:
                    new_position -= force * (selected_particles[j].position - selected_particles[i].position)
            
            new_particles.append(MultiObjectiveMember(new_position, np.zeros(self.n_objectives)))
        
        return new_particles
    
    def _strategy_3(self, selected_particles: List[MultiObjectiveMember], 
                   leader: MultiObjectiveMember, force: float) -> List[MultiObjectiveMember]:
        """Strategy 3: Hybrid approach with two types of movements."""
        new_particles_1 = []
        new_particles_2 = []
        
        for i in range(self.npi):
            # Type 1 movement (similar to strategy 1)
            s1_position = selected_particles[i].position.copy()
            s1_position += force * (leader.position - selected_particles[i].position)
            
            # Type 2 movement (full force towards leader)
            s2_position = selected_particles[i].position.copy()
            s2_position += 1 * force * (leader.position - selected_particles[i].position)
            
            for j in range(self.npi):
                if j < i:
                    s1_position += force * (selected_particles[j].position - selected_particles[i].position)
                    s2_position += force * (selected_particles[j].position - selected_particles[i].position)
                elif j > i:
                    s1_position -= force * (selected_particles[j].position - selected_particles[i].position)
                    s2_position -= force * (selected_particles[j].position - selected_particles[i].position)
            
            new_particles_1.append(MultiObjectiveMember(s1_position, np.zeros(self.n_objectives)))
            new_particles_2.append(MultiObjectiveMember(s2_position, np.zeros(self.n_objectives)))
        
        return new_particles_1 + new_particles_2
    
    def _apply_archive_mutation(self, new_particles: List[MultiObjectiveMember]):
        """Apply archive-based mutation to new particles."""
        if not self.archive:
            return
        
        for particle in new_particles:
            for j in range(self.dim):
                if np.random.random() < 0.2:  # 20% chance of mutation
                    # Replace dimension with value from random archive member
                    archive_member = np.random.choice(self.archive)
                    particle.position[j] = archive_member.position[j]
    
    def _update_population(self, population: List[MultiObjectiveMember], 
                          new_particles: List[MultiObjectiveMember]):
        """
        Update population by replacing worst particles with new ones.
        
        Args:
            population (List[MultiObjectiveMember]): Current population
            new_particles (List[MultiObjectiveMember]): Newly generated particles
        """
        if not new_particles:
            return
        
        # Determine domination status of current population
        self._determine_domination(population)
        
        # Get dominated particles (worst ones)
        dominated_particles = [p for p in population if p.dominated]
        
        # Replace dominated particles with new particles
        n_to_replace = min(len(dominated_particles), len(new_particles))
        
        for i in range(n_to_replace):
            idx = population.index(dominated_particles[i])
            population[idx] = new_particles[i]
    
    def _get_best_from_population(self, population: List[MultiObjectiveMember]) -> MultiObjectiveMember:
        """
        Get the best particle from population based on random fitness.
        
        Args:
            population (List[MultiObjectiveMember]): Population to select from
            
        Returns:
            MultiObjectiveMember: Best particle
        """
        if not population:
            return None
        
        # Use random fitness for selection
        fitness_values = [self._get_random_fitness(p) for p in population]
        
        if self.maximize:
            best_idx = np.argmax(fitness_values)
        else:
            best_idx = np.argmin(fitness_values)
        
        return population[best_idx]
    
    def _n_choose_2(self, n: int) -> int:
        """Calculate n choose 2 (number of combinations)."""
        if n < 2:
            return 0
        return n * (n - 1) // 2
