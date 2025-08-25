import numpy as np
from typing import Callable, Union, Tuple, List
from ._core import MultiObjectiveSolver, MultiObjectiveMember

class ParticleMultiMember(MultiObjectiveMember):
    def __init__(self, position: np.ndarray, fitness: np.ndarray, velocity: np.ndarray = None):
        super().__init__(position, fitness)
        self.velocity = velocity if velocity is not None else np.zeros_like(position)
        self.personal_best_position = position.copy()
        self.personal_best_fitness = fitness.copy()
    
    def copy(self):
        new_member = ParticleMultiMember(self.position.copy(), self.multi_fitness.copy(), self.velocity.copy())
        new_member.dominated = self.dominated
        new_member.grid_index = self.grid_index
        new_member.grid_sub_index = self.grid_sub_index
        new_member.personal_best_position = self.personal_best_position.copy()
        new_member.personal_best_fitness = self.personal_best_fitness.copy()
        return new_member
    
    def __str__(self):
        return f"Position: {self.position} - Fitness: {self.multi_fitness} - Velocity: {self.velocity}"

class MultiObjectiveParticleSwarmOptimizer(MultiObjectiveSolver):
    """
    Multi-Objective Particle Swarm Optimizer
    
    This algorithm extends the standard PSO for multi-objective optimization
    using archive management and grid-based selection for leader selection.
    
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
        - w: Inertia weight (default: 1.0)
        - wdamp: Inertia weight damping ratio (default: 0.99)
        - c1: Personal learning coefficient (default: 1.5)
        - c2: Global learning coefficient (default: 2.0)
        - vel_max: Maximum velocity (default: 10% of variable range)
        - vel_min: Minimum velocity (default: -10% of variable range)
    """
    
    def __init__(self, objective_func: Callable, lb: Union[float, np.ndarray], 
                 ub: Union[float, np.ndarray], dim: int, maximize, **kwargs):
        super().__init__(objective_func, lb, ub, dim, maximize, **kwargs)
        
        # Set solver name
        self.name_solver = "Multi-Objective Particle Swarm Optimizer"
        
        # PSO-specific parameters
        self.w = kwargs.get('w', 1.0)  # Inertia weight
        self.wdamp = kwargs.get('wdamp', 0.99)  # Inertia weight damping ratio
        self.c1 = kwargs.get('c1', 1.5)  # Personal learning coefficient
        self.c2 = kwargs.get('c2', 2.0)  # Global learning coefficient
        
        # Velocity limits (10% of variable range)
        vel_range = 0.1 * (self.ub - self.lb)
        self.vel_max = kwargs.get('vel_max', vel_range)
        self.vel_min = kwargs.get('vel_min', -vel_range)

    def _init_population(self, search_agents_no) -> List[ParticleMultiMember]:
        """Initialize multi-objective particle population"""
        population = []
        for _ in range(search_agents_no):
            position = np.random.uniform(self.lb, self.ub, self.dim)
            velocity = np.random.uniform(self.vel_min, self.vel_max, self.dim)
            fitness = self.objective_func(position)
            population.append(ParticleMultiMember(position, fitness, velocity))
        return population

    def _dominates_personal_best(self, particle: ParticleMultiMember) -> bool:
        """Check if current position dominates personal best (Pareto dominance)"""
        return self._dominates(particle, 
                             ParticleMultiMember(particle.personal_best_position, 
                                               particle.personal_best_fitness))

    def solver(self, search_agents_no: int, max_iter: int) -> Tuple[List, List[ParticleMultiMember]]:
        """
        Main optimization method for multi-objective PSO
        
        Parameters:
        -----------
        search_agents_no : int
            Number of search agents (population size)
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        Tuple[List, List[ParticleMultiMember]]
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
            # Update all particles
            for i, particle in enumerate(population):
                # Select leader from archive using grid-based selection
                leader = self._select_leader()
                
                # If no leader in archive, use random particle from population
                if leader is None:
                    leader = np.random.choice(population)
                
                # Update velocity
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                
                cognitive_component = self.c1 * r1 * (particle.personal_best_position - particle.position)
                social_component = self.c2 * r2 * (leader.position - particle.position)
                
                particle.velocity = (self.w * particle.velocity + 
                                   cognitive_component + 
                                   social_component)
                
                # Apply velocity limits
                particle.velocity = np.clip(particle.velocity, self.vel_min, self.vel_max)
                
                # Update position
                new_position = particle.position + particle.velocity
                
                # Apply position limits and velocity mirror effect
                outside_bounds = (new_position < self.lb) | (new_position > self.ub)
                particle.velocity[outside_bounds] = -particle.velocity[outside_bounds]
                new_position = np.clip(new_position, self.lb, self.ub)
                
                # Update particle position
                particle.position = new_position
                
                # Evaluate new fitness
                new_fitness = self.objective_func(new_position)
                particle.multi_fitness = new_fitness
                
                # Update personal best if current position dominates personal best
                if self._dominates_personal_best(particle):
                    particle.personal_best_position = particle.position.copy()
                    particle.personal_best_fitness = particle.multi_fitness.copy()
            
            # Update archive with current population
            self._add_to_archive(population)
            
            # Store archive state for history
            history_archive.append([particle.copy() for particle in self.archive])
            
            # Update inertia weight
            self.w *= self.wdamp
            
            # Update progress
            self._callbacks(iter, max_iter, self.archive[0] if self.archive else None)
        
        # Final processing
        self.history_step_solver = history_archive
        self.best_solver = self.archive
        
        # End solver
        self._end_step_solver()
        
        return history_archive, self.archive
